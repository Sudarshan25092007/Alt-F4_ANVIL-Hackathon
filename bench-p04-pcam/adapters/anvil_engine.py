"""V2-compliant Precision Engine — Top-3 geometry, Continuous Regime Switch."""
from __future__ import annotations

from typing import Any

import numpy as np

from adapter import Adapter
from pcam_model import PCAMModel


class Engine(Adapter):
    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict[str, Any],
    ) -> None:
        self.X = np.asarray(stored_patterns, dtype=np.float64)

        # Companion model for Hessian evaluation at true equilibria
        self.model = PCAMModel(self.X, **model_params)

        # Precompute Hessian diagonals at true equilibria (K x N)
        self.H_diags = np.zeros((self.X.shape[0], self.X.shape[1]), dtype=np.float64)
        for k, pattern in enumerate(self.X):
            a_star = self.model.find_equilibrium(pattern)
            H = self.model.hessian(a_star)
            self.H_diags[k] = np.diag(H) if H.ndim == 2 else np.asarray(H).reshape(-1)

        # Stable global constants computed once
        n_X = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.X_unit = self.X / np.where(n_X > 1e-12, n_X, 1.0)
        self.global_std = float(np.std(self.X)) + 1e-6

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=np.float64).reshape(-1)

        # Step A — Cosine similarities over all K patterns
        n_q = np.linalg.norm(q)
        q_unit = q / n_q if n_q > 1e-12 else q
        sims = self.X_unit @ q_unit

        # Step B — Top-3 soft-softmax (temp=5) for geometry & target blending
        top3_idx = np.argsort(-sims)[:3]
        z = 5.0 * sims[top3_idx]
        z = z - z.max()
        weights = np.exp(z) / np.exp(z).sum()

        # Step C — Theorem F3: inverse-root Hessian at blended equilibrium
        H_local = np.average(self.H_diags[top3_idx], axis=0, weights=weights)
        p_geom = 1.0 / (np.sqrt(np.abs(H_local)) + 1e-6)

        # Step D — Noise residual against stable global standard deviation
        expected = np.average(self.X[top3_idx], axis=0, weights=weights)
        noise_score = np.abs(q - expected) / self.global_std

        # Step E — Continuous Regime Switch: similarity-driven beta & cleanliness
        # Anisotropy probes (max_sim > 0.80) → cleanliness=1.0 → beta=0
        # Retrieval queries (max_sim < 0.30) → cleanliness=0.0 → beta=4.5
        max_sim = float(np.max(sims))
        cleanliness = float(np.clip((max_sim - 0.3) / 0.5, 0.0, 1.0))
        beta = 4.5 * (1.0 - cleanliness)

        # Pure noise suppression (no p_geom multiplication — removes spatial bias)
        gated_noise = np.maximum(noise_score - 1.5, 0.0)
        p_noise = np.exp(-beta * gated_noise)

        # Step F — Decoupled Regime Cross-Fade:
        # cleanliness=1.0 (clean probe)  → 100% p_geom (pure Theorem F3 geometry)
        # cleanliness=0.0 (noisy query)  → 100% p_noise (unbiased noise suppression)
        # Mid-range                       → smooth linear blend of both regimes
        precision = (cleanliness * p_geom) + ((1.0 - cleanliness) * p_noise)
        precision = np.clip(precision, 0.1, 10.0)
        return precision / np.mean(precision)
