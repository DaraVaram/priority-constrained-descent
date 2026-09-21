"""Priority-Constrained Descent (PCD) for hierarchical multi-objective optimization.

    Dara Varam and Mohamed I. AlHajri. "Not All Objectives Are Born Equal: Priority-Constrained Descent
    for Hierarchical Multi-Objective Optimization." Transactions on Machine Learning Research, 2026.

``PCD`` is the PyTorch front end. ``solve_qp`` and ``EMANormalizer`` are framework-independent (NumPy)
and are all that is needed to use PCD with another framework.
"""
from .normalize import EMANormalizer
from .optim import PCD, PCDInfo
from .qp import QPResult, solve_qp

__all__ = ["PCD", "PCDInfo", "EMANormalizer", "solve_qp", "QPResult"]
__version__ = "1.0.0"
