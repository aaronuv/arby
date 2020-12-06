Tutorial
--------

Build a reduced basis for Bessel functions of first kind.

Suppose we want to find surrogates functions for solutions of the Bessel
differential equation with a free parameter :math:`\nu`.

.. math::

    x^2 \frac{d^2f}{dx^2} + x \frac{df}{dx} + (x^2 - \nu^2)y = 0

Suppose we have numerical solutions :math:`J_{\nu}(x)` particular values of
the parameter :math:`\nu`, say, for a discretized interval :math:`[1, 5]` with
101 samples, defined in some interval for :math:`x`.

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
        bessel_model.surrogate(par)

