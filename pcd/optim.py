"""PyTorch front end: turn K losses into one PCD update direction for any optimizer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .normalize import EMANormalizer
from .qp import solve_qp


@dataclass
class PCDInfo:
    """Diagnostics of one PCD step. Index 0 is the primary; lists over secondaries have K-1 entries."""

    mu: List[float]            # KKT multipliers of the normalized QP (Thm. 4.2); 0 when inactive
    active: Tuple[int, ...]    # indices (into ``losses``) of the constraints that bind this step
    feasible: bool             # False only if the constraints were mutually incompatible (K >= 3)
    cosine: List[float]        # cos(g_1, g_j): the conflict between primary and each secondary
    grad_norm: List[float]     # raw gradient norms ||g_i||
    scale: List[float]         # EMA normalization scales s_i


class PCD:
    """Priority-Constrained Descent (Algorithm 1) as a gradient transform.

    ``backward(losses)`` computes one gradient per loss, solves the PCD QP on their normalized Gram
    matrix, rescales the direction to the primary gradient's norm and writes it into ``p.grad``. Any
    optimizer then takes the step::

        pcd = PCD(model.parameters(), tau=0.02)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for x, y in loader:
            task = F.cross_entropy(model(x), y)
            info = pcd.backward([task, penalty(model)])   # primary first, then the secondaries
            opt.step()

    Args:
        params: parameters to optimize (the same ones the optimizer holds).
        tau:    tolerance in [0, 1]. Each secondary must receive at least a tau-fraction of the progress
                a step along its own normalized gradient would give; the update stays as close to the
                primary gradient as that allows. A scalar as in the paper, or one value per secondary.
        beta:   EMA decay of the per-objective gradient-norm normalization (Def. 4.5).
        eps:    stabilizer of the normalization.
    """

    def __init__(self, params: Iterable[torch.Tensor], tau: Union[float, Sequence[float]] = 0.02,
                 beta: float = 0.999, eps: float = 1e-8):
        self.params = [p for p in params if p.requires_grad]
        if not self.params:
            raise ValueError("PCD received no parameters that require grad")
        if np.any(np.asarray(tau, dtype=float) < 0):
            raise ValueError(f"tau must be non-negative, got {tau}")
        self.tau = tau
        self.normalizer = EMANormalizer(beta, eps)

    def backward(self, losses: Sequence[torch.Tensor]) -> PCDInfo:
        """Differentiate each loss (``losses[0]`` is the primary) and set ``p.grad`` to the PCD direction.

        This runs one backward pass per loss, as any gradient-manipulation method must. Existing
        ``.grad`` values are overwritten, so ``zero_grad()`` beforehand is optional.
        """
        grads = []
        for i, loss in enumerate(losses):
            if loss.requires_grad:
                grads.append(torch.autograd.grad(loss, self.params, retain_graph=i < len(losses) - 1,
                                                 allow_unused=True))
            else:                                    # a constant loss has no gradient
                grads.append((None,) * len(self.params))
        return self.apply_gradients(grads)

    @torch.no_grad()
    def apply_gradients(self, grads: Sequence[Sequence[Optional[torch.Tensor]]]) -> PCDInfo:
        """Like ``backward``, but from precomputed gradients: ``grads[i][k]`` is the gradient of loss ``i``
        with respect to ``params[k]`` (``None`` means zero). Use this when the gradients come from
        elsewhere, e.g. after an all-reduce in data-parallel training."""
        K = len(grads)
        if K == 0:
            raise ValueError("need at least one loss")
        used = [any(g[k] is not None for g in grads) for k in range(len(self.params))]
        flat = [self._flatten(g, used) for g in grads]

        # Gram matrix of the raw gradients: the only O(n) work besides forming the final direction.
        pairs = [(i, j) for i in range(K) for j in range(i, K)]
        dots = torch.stack([torch.dot(flat[i], flat[j]) for i, j in pairs]).double().cpu().numpy()
        gram = np.empty((K, K))
        for (i, j), v in zip(pairs, dots):
            gram[i, j] = gram[j, i] = v

        scale = self.normalizer(np.diag(gram))       # updated every step, including the halting branch
        norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
        cosine = [float(gram[0, j] / (norms[0] * norms[j])) if norms[0] * norms[j] > 0 else 0.0
                  for j in range(1, K)]

        if norms[0] == 0.0:                          # g_1 = 0: the deployed update halts (Sec. 4.7)
            self._write(torch.zeros_like(flat[0]), used)
            return PCDInfo([0.0] * (K - 1), (), True, cosine, norms.tolist(), scale.tolist())

        result = solve_qp(gram * np.outer(scale, scale), self.tau)
        coef = result.weights * scale                # d~* = sum_i w_i s_i g_i, in raw gradients
        direction = flat[0] * float(coef[0])
        for i in range(1, K):
            if coef[i] != 0.0:
                direction.add_(flat[i], alpha=float(coef[i]))
        d_norm = direction.norm()
        # Magnitude rescale d* = d~* ||g_1|| / ||d~*||; d~* = 0 gives d* = 0 (Algorithm 1).
        factor = torch.where(d_norm > 0, float(norms[0]) / d_norm, torch.zeros_like(d_norm))
        self._write(direction.mul_(factor), used)
        return PCDInfo(result.weights[1:].tolist(), result.active, result.feasible, cosine,
                       norms.tolist(), scale.tolist())

    def _flatten(self, grads: Sequence[Optional[torch.Tensor]], used: List[bool]) -> torch.Tensor:
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for p, g, u in zip(self.params, grads, used) if u])

    def _write(self, direction: torch.Tensor, used: List[bool]) -> None:
        offset = 0
        for p, u in zip(self.params, used):
            if not u:                                # no loss depends on p: leave it out of the step
                p.grad = None
                continue
            n = p.numel()
            p.grad = direction[offset:offset + n].view_as(p).to(p.dtype)
            offset += n

    def state_dict(self) -> dict:
        return {"tau": self.tau, "normalizer": self.normalizer.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.tau = state["tau"]
        self.normalizer.load_state_dict(state["normalizer"])

    def __repr__(self) -> str:
        return (f"PCD(tau={self.tau}, beta={self.normalizer.beta}, eps={self.normalizer.eps}, "
                f"params={sum(p.numel() for p in self.params):,})")
