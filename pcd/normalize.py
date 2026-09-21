"""Per-objective gradient normalization (Def. 4.5).

This is Adam's bias-corrected second-moment estimate with one scalar per *objective* rather than one per
coordinate:

    v_i <- beta v_i + (1 - beta) ||g_i||^2,    v_hat_i = v_i / (1 - beta^t),    s_i = 1 / sqrt(v_hat_i + eps)

In steady state ||s_i g_i|| ~ 1 for every objective, which is what makes tau a scale-free fraction of the
achievable secondary progress (Thm. B.9).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class EMANormalizer:
    def __init__(self, beta: float = 0.999, eps: float = 1e-8):
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.beta, self.eps = beta, eps
        self.v: Optional[np.ndarray] = None
        self.t = 0

    def __call__(self, sq_norms) -> np.ndarray:
        """Update with this step's squared gradient norms ||g_i||^2 and return the scales s_i."""
        sq = np.asarray(sq_norms, dtype=np.float64)
        if self.v is None:
            self.v = np.zeros_like(sq)
        if sq.shape != self.v.shape:
            raise ValueError(f"expected {self.v.shape[0]} objectives, got {sq.shape[0]}")
        self.t += 1
        self.v = self.beta * self.v + (1.0 - self.beta) * sq
        v_hat = self.v / (1.0 - self.beta ** self.t)
        # The paper's form. The research code used 1 / (sqrt(v_hat) + eps); at eps = 1e-8 the two agree
        # to well below float32 precision for any gradient norm seen in practice.
        return 1.0 / np.sqrt(v_hat + self.eps)

    def state_dict(self) -> dict:
        return {"beta": self.beta, "eps": self.eps, "t": self.t,
                "v": None if self.v is None else self.v.tolist()}

    def load_state_dict(self, state: dict) -> None:
        self.beta, self.eps, self.t = state["beta"], state["eps"], state["t"]
        self.v = None if state["v"] is None else np.asarray(state["v"], dtype=np.float64)
