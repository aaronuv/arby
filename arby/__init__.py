# __init__.py

__version__ = "0.1a1"

from .integrals import Integration
from .greedy import ReducedBasis
from .eim import EmpiricalMethods

__all__ = [Integration, ReducedBasis, EmpiricalMethods]