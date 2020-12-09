Tutorial
========

Example: Bessel functions of first kind
---------------------------------------

Build a surrogate model
^^^^^^^^^^^^^^^^^^^^^^^

Suppose we want to find surrogates functions for solutions of the Bessel
differential equation with a free parameter :math:`\nu`.

.. math::

    x^2 \frac{d^2f}{dx^2} + x \frac{df}{dx} + (x^2 - \nu^2)y = 0

Suppose we have numerical solutions :math:`J_{\nu}(x)` for particular values of
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
        bessel_model = ROM(training_space=training,
                           physical_interval=x,
                           parameter_interval=nu)

        # build and evaluate the surrogate at some parameter `par`
        bessel_par = bessel_model.surrogate(par)

Note that ``par`` does not necessarily belong to the training parameters.

We can test the accuracy of our model in an arbitrary ``par`` (that belongs to the parameter interval)
in the :math:`L_2`-norm with using ``integration``, an object defined inside our ``bessel_model``
that comprises inner products.

.. code-block:: python

        squared_L2_error = bessel_model.integration.norm(bessel_par - BesselJ(par, x))

If we want to improve the surrogate accuracy, we tune the ``greedy_tol`` (Default= ``1e-12``)
or the ``poly_deg`` (Default= ``3``) class parameters. The first one controls the precission of
the underlying basis. The second one controls the degree of the interpolation polynomials used to
build splines that will give us continuity in the parameter space.

Build a reduced basis
^^^^^^^^^^^^^^^^^^^^^

Lets go deeper. The Reduced Basis Method (RBM) [1]_ is a reduced order modeling technique to find a
cuasi-optimal basis of functions capable of span the entire training set. This is a projection
approach, say, we need an inner product to perform projections and construct the approximation.

Suppose we have a training set :math:`\{f_{\lambda_i}\}_{i=1}^N` of parameterized real
functions. This set may represent a non-linear model, perhaps solution of PDEs. We would
like, if possible, to reduce the dimensionality/complexity of these set by traying to find a
compact representation in terms of linear combinations of basis elements
:math:`\{e_i\}_{i=1}^n`, that is,

.. math::

        f \approx \sum_{i=1}^n c_i e_i\,.

f is an arbitrary training function and the :math:`c_i`'s are the projection coefficients
:math:`<e_i,f>` computed in some inner product :math:`<\cdot,\cdot>` on the space of functions.
The RB method choose a set of optimal functions that belongs to the training set to build a
finite dimensional subspace capable to represent the entire training set up to a prefixed tolerance
chosen by the user.

To build a reduced basis with Arby, you just provide the training set of functions and the
discretization of the physical variable :math:`x` to the ``ReducedOrderModeling`` class.
The later is to define the integration scheme used to compute inner products. For the
Bessel example,

.. code-block:: python

        bessel_model = ROM(training_space=training,
                           physical_interval=x, greedy_tol=1e-12)

The ``greedy_tol`` is the accuracy in the :math:`L_2`-norm that our reduced basis is expected
to achieve. To build the basis, just call it:

.. code-block:: python

        reduced_basis = bessel_model.basis

This builds an orthonormalized basis. We can access to the *greedy points* through
``bessel_model.greedy_indices``. These indices mark those functions in the training
set that was selected to span the approximating subspace. For stability reasons,
they are iteratively orthonormalized in the building stage. The number of basis
elements ``bessel_model.Nbasis`` represents the dimension of the subspace and is not
fixed. It changes if we change the greedy tolerance. The lower the tolerance,
the bigger the number of basis elements needed to reach that accuracy. With Arby,
we can tune the accuracy of the reduced basis through ``greedy_tol``.

To measure the effectiveness of the reduced basis in approximatting the training
functions we do

.. code-block:: python

        projected_f = bessel_model.project(f, reduced_basis)
        squared_L2_error = bessel_model.integration,norm(f - projected_f)



References
----------

.. [1] Scott E. Field, Chad R. Galley, Jan S. Hesthaven, Jason Kaye,
       and Manuel Tiglio. Fast Prediction and Evaluation of Gravitational
       Waveforms Using Surrogate Models. Phys. Rev. X 4, 031006
