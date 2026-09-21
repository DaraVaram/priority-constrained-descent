"""Secondary objectives of the paper's experiments, their proximal steps, and the reported metrics.

Every function takes the list of penalized conv weights (``model.penalized_weights()``). A conv weight
of shape [C_out, C_in, kh, kw] is viewed as a [C_out, C_in*kh*kw] matrix: one group per output filter
for the group lasso, one matrix per layer for the nuclear norm. The losses are unweighted sums; PCD
needs no penalty coefficient.
"""
import math

import torch


def group_lasso(weights):
    """Structured sparsity (Sec. 5.1): sum of the L2 norms of the output filters."""
    return sum(w.flatten(1).norm(dim=1).sum() for w in weights)


def l1(weights):
    """Unstructured sparsity (Sec. 5.2)."""
    return sum(w.abs().sum() for w in weights)


def nuclear(weights):
    """Low rank (Sec. 5.2): sum of singular values of each layer's matrix. Autograd gives U V^T."""
    return sum(torch.linalg.svdvals(w.flatten(1)).sum() for w in weights)


SECONDARIES = {"group_lasso": group_lasso, "l1": l1, "nuclear": nuclear}


# Proximal steps. They are not part of PCD. The paper's runs applied one after every optimizer step with
# threshold lambda * lr; ``train.py --prox LAMBDA`` replicates that (off by default).
@torch.no_grad()
def prox_group_lasso(weights, t):
    for w in weights:
        g = w.view(w.shape[0], -1)
        g.mul_((1 - t / g.norm(dim=1, keepdim=True).clamp(min=1e-12)).clamp(min=0))


@torch.no_grad()
def prox_l1(weights, t):
    for w in weights:
        w.copy_(w.sign() * (w.abs() - t).clamp(min=0))


@torch.no_grad()
def prox_nuclear(weights, t):
    for w in weights:
        try:
            U, S, Vh = torch.linalg.svd(w.flatten(1), full_matrices=False)
        except RuntimeError:                              # SVD did not converge: leave this layer as is
            continue
        w.copy_(((U * (S - t).clamp(min=0)) @ Vh).view_as(w))


PROX = {"group_lasso": prox_group_lasso, "l1": prox_l1, "nuclear": prox_nuclear}


# Metrics, with the thresholds used for every number in the paper.
@torch.no_grad()
def group_sparsity(weights, thr=1e-4):
    """Native group sparsity (%): output filters whose L2 norm training drove below ``thr``."""
    norms = torch.cat([w.flatten(1).norm(dim=1) for w in weights])
    return 100.0 * (norms < thr).float().mean().item()


@torch.no_grad()
def param_sparsity(params, thr=1e-8):
    """Unstructured sparsity (%) over the given tensors: entries with |w| < ``thr``."""
    params = list(params)
    zeros = sum((p.abs() < thr).sum().item() for p in params)
    return 100.0 * zeros / sum(p.numel() for p in params)


@torch.no_grad()
def effective_rank(weights):
    """Mean over layers of exp(entropy of the normalized singular values) (Roy & Vetterli, 2007)."""
    ranks = []
    for w in weights:
        s = torch.linalg.svdvals(w.flatten(1))
        if s.sum() <= 0:
            ranks.append(0.0)
            continue
        p = s / s.sum()
        ranks.append(math.exp(-(p * (p + 1e-12).log()).sum().item()))
    return sum(ranks) / len(ranks)


@torch.no_grad()
def rank_reduction(weights, abs_thr=1e-5, rel_thr=1e-3):
    """Numerical-rank reduction (%): singular values above max(abs_thr, rel_thr * sigma_max) are kept."""
    kept = full = 0
    for w in weights:
        m = w.flatten(1)
        s = torch.linalg.svdvals(m)
        kept += int((s > max(abs_thr, rel_thr * s.max().item())).sum())
        full += min(m.shape)
    return 100.0 * (full - kept) / full
