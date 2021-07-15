#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2021, Juan B Cabral and QuatroPe.
# License: BSD-3-Clause
#   Full Text: https://github.com/quatrope/pyonono/blob/master/LICENSE

# This file is a prototype for the upcoming pyonono benchmark framework

import arby

import numpy as np

import pytest

# =============================================================================
# BENCHMARK
# =============================================================================


@pytest.mark.parametrize("iteration", range(1000))  # repite cada test 100 veces
@pytest.mark.parametrize(
    "training_set_shape",  # b
    [
        (11, 11),
        (101, 101),
        (1001, 1001),
        (10001, 10001)
    ],
)
@pytest.mark.parametrize(
    "integration_rule", ["riemann", "trapezoidal", "euclidean"]
)
@pytest.mark.parametrize("greedy_tol", [1e-12, [1e-14]])
@pytest.mark.parametrize("normalize", [True, False])
def test_benchmark_basis(
    benchmark, training_set_shape, integration_rule, greedy_tol, iteration, normalize
):
    physical_points_size = training_set_shape[1]
    physical_points = np.linspace(0, 1, physical_points_size)

    training_set = np.random.random(training_set_shape)

    benchmark(
        arby.reduced_basis,
        training_set=training_set,
        physical_points=physical_points,
        integration_rule=integration_rule,
        greedy_tol=greedy_tol,
        normalize=normalize
    )
