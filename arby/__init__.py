# __init__.py

__version__ = "0.1"

from .integrals import Integration
from .core import ReducedOrderModeling, gram_schmidt

__all__ = ["Integration", "ReducedOrderModeling", "gram_schmidt"]
