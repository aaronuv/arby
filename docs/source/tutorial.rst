Tutorial
========

**Build a surrogate model for Bessel functions of first kind.**

Suppose we want to find surrogates functions for solutions of the Bessel
differential equation with a free parameter :math:`\nu`.

.. math::

    x^2 \frac{d^2f}{dx^2} + x \frac{df}{dx} + (x^2 - \nu^2)y = 0

Suppose we have numerical solutions :math:`J_{\nu}(x)` particular values of
the parameter :math:`\nu`, say, for a discretized interval :math:`[1, 5]` with
101 samples, defined in some interval for :math:`x`. For convention, we will
refer to :math:`\nu` as the parameter variable and :math:`x` as the physical one.
So each one belongs to a parameter space and a physical domain, respectively.

We can use Arby to build a surrogate model for this data set. In this example,
we will generate the sample data using scipy's Bessel special functions.

.. code-block:: python

        from arby import ReducedOrderModeling as ROM
        from scipy.special import jv as BesselJ

        npoints = 101
        
        # Sample parameter nu and variable x
        nu = np.linspace(1, 5, num=npoints)
        x = np.linspace(0, 10, 1001)

        # build traning set
        training = np.array([BesselJ(nn, x) for nn in nu])

        # create a model
        bessel_model = ROM(training, x, nu)

        # build and evaluate the surrogate at some parameter `par`
        bessel_par = bessel_model.surrogate(par)

Note that ``par`` does not necessarily belong to the training parameters.

We can test the accuracy of our model in an arbitrary ``par`` (that belongs to the parameter interval)
in the :math:`L_2`-norm with using ``integration``, an object defined inside our ``bessel_model``
that comprises inner products.

.. code-block:: python

        bessel_model.integration.norm(bessel_par - BesselJ(par, x))

**Build a reduced basis**

Lets go deeper. Reduced Basis _[1] is a reduced order modeling technique to find a
cuasi-optimal basis capable of span the entire training set by means of projection.
Suppose we have a training set :math:`\{f_{\lambda_i}\}_{i=1}^N` of parameterized real
functions. This set may represent a non-linear model, perhaps solution of PDEs. We would
like, if possible, to reduce the dimensionality/complexity of these set traying to find a
compact representation of them in terms of linear combinations of basis elements
:math:`\{e_i\}_{i=1}^n`.

To build a reduced basis, you just provide the training set of functions and the discretization of
the physical variable :math:`x`. The later is to define an integration scheme for inner products.