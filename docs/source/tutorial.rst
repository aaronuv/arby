Tutorial
--------

Build a reduced basis for the Bessel problem.

Suppose we want to find surrogate functions for the Bessel differential equation,
with a free parameter :math:`\alpha`.

.. math::

    x^2 \frac{d^2f}{dx^2}+...

Suppose we have numerical solutions :math:`J_{\alpha}(x)` for particular values of the parameter :math:`\alpha`,
discretized over the interval :math:`[0, 1]` with 1000 samples.

We can use Arby to build a surrogate model for this data set.
In this example, we will generate the sample data using scipy's Bessel special functions.

.. code-block:: python

        import arby as rb
        from scipy.special import jv as BesselJ

        npoints = 101
        # Sample parameter nu and physical variable x
        nu = np.linspace(0, 10, num=npoints)
        x = np.linspace(0, 1, 101)
        # build traning space
        training = np.array([BesselJ(nn, x) for nn in nu])
        # build reduced basis
        rb = arby.ReducedBasis(training, [0, 1], rule="riemann")
        rb.build_rb(tol=1e-14)

