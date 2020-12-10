# Arby

<img src="res/logo.png" alt="logo" width="42%">

[![PyPI version](https://badge.fury.io/py/arby.svg)](https://badge.fury.io/py/arby)
[![Build Status](https://travis-ci.com/aaronuv/rbpy.svg?branch=master)](https://travis-ci.com/aaronuv/rbpy)
[![Documentation Status](https://readthedocs.org/projects/arby/badge/?version=latest)](https://arby.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/gitlab/aaronuv/arby/badge.svg?branch=master)](https://coveralls.io/gitlab/aaronuv/arby?branch=master)
[![https://github.com/leliel12/diseno_sci_sfw](https://img.shields.io/badge/DiSoftCompCi-FAMAF-ffda00)](https://github.com/leliel12/diseno_sci_sfw)
[![codecov](https://codecov.io/gl/aaronuv/arby/branch/'master'/graph/badge.svg?token=lbQD1dc32z)](https://codecov.io/gl/aaronuv/arby)

[Model Order Reduction (MOR)](https://en.wikipedia.org/wiki/Model_order_reduction)
is a technique for reducing the computational complexity of mathematical models in
numerical simulations.

Arby is a fully data-driven Python module to construct reduced bases,
empirical interpolants and surrogate models from training data.

# Install

    pip install arby

# Quick Usage

Suppose we want to build a surrogate model for a family of real functions $`f_\lambda(x)`$
parametrized by a real number $`\lambda\in[\lambda_{min},\lambda_{max}]`$ and $`x\in[a,b]`$.
We have discretizations of both domains in, say, 101 and 1001 points respectively,
```
lambda_params = np.linspace(lambda_min, lambda_max, 101)
x_samples = np.linspace(a, b, 1001)
```
The next step is to build a training set of functions associated to the discretizations.

```
training_data = [f(lambda, x_samples) for lambda in lambda_params]
```
This is an array of shape $`(101,1001)`$.

Then we can build a surrogate model with `arby` using:

    from arby import ReducedOrderModeling as ROM
    f_model = ROM(training_space=training_data,
                  physical_interval=x_samples,
                  parameter_interval=lambda_params)
    
With our `f_model` we can get function samples for any parameter $`\lambda`$ in the
interval $`\lambda\in[\lambda_{min},\lambda_{max}]`$.

    new_param = 0.554
    f_model_new_param = f_model.surrogate(new_param)
    plt.plot(x_samples, model_new_param)
    plt.show()

# License

MIT

***

(c) 2020 Aaron Villanueva
