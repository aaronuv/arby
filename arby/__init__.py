# __init__.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


"""Reduced Order Modeling tools for building surrogate models."""

__version__ = "1.0"

__all__ = [
    "Basis",
    "Integration",
    "ReducedOrderModel",
    "gram_schmidt",
    "reduced_basis",
]

from .basis import Basis, gram_schmidt, reduced_basis
from .integrals import Integration
from .rom import ReducedOrderModel
