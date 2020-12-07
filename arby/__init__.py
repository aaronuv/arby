# __init__.py

"""Model Order Reduction (MOR)
is a technique for reducing the computational complexity of
mathematical models in numerical simulations.
Arby is a fully data-driven Python module to construct reduced bases,
empirical interpolants and surrogate models from training data.
"""

__version__ = "0.1"

__all__ = ["Integration", "ReducedOrderModeling", "gram_schmidt"]

from .integrals import Integration
from .core import ReducedOrderModeling, gram_schmidt
