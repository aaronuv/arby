# test_rompy_to_arby.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib

import numpy as np

import pytest


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

BESSEL_PATH = PATH / "bessel"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def basis_data():
    path = BESSEL_PATH / "bessel_basis.txt"
    return np.loadtxt(path)


@pytest.fixture
def basis(basis_data):
    return basis_data


@pytest.fixture
def training_space():
    path = BESSEL_PATH / "bessel_training.txt"
    return np.loadtxt(path)


@pytest.fixture
def physical_interval():
    path = BESSEL_PATH / "physical_interval.txt"
    return np.loadtxt(path)


@pytest.fixture
def rom_parameters(training_space, physical_interval):
    parameter_interval = np.linspace(0, 1, len(training_space))
    params = {
        "training_space": training_space,
        "physical_interval": physical_interval,
        "parameter_interval": parameter_interval,
    }
    return params