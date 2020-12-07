# __init__.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


"""Model Order Reduction (MOR) is a technique for reducing the computational
the complexity of mathematical models in numerical simulations.

Arby is a fully data-driven Python module to construct reduced bases, empirical
interpolants and surrogate models from training data.

"""

__version__ = "0.1"

__all__ = ["Integration", "ReducedOrderModeling", "gram_schmidt"]

from .core import ReducedOrderModeling, gram_schmidt
from .integrals import Integration
