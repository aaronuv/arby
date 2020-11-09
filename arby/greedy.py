# --- greedy.py ---

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE
"""
Classes for building reduced basis greedy algorithms
"""

__author__ = "Chad Galley <crgalley@tapir.caltech.edu, crgalley@gmail.com>"


import numpy as np
from .integrals import Integration

#############################################
# Class for iterated, modified Gram-Schmidt #
#      orthonormalization of functions      #
#############################################


class _IteratedModifiedGramSchmidt:
    """Iterated modified Gram-Schmidt algorithm for building an orthonormal
    basis. Algorithm from Hoffman, `Iterative Algorithms for Gram-Schmidt
    Orthogonalization`.
    """

    def __init__(self, inner):
        self.inner = inner

    def add_basis(self, h, basis, a=0.5, max_iter=3):
        """Given a function, h, find the corresponding basis function
        orthonormal to all previous ones"""
        norm = self.inner.norm(h)
        e = h / norm

        flag, ctr = 0, 1
        while flag == 0:
            for b in basis:
                e -= b * self.inner.dot(b, e)
            new_norm = self.inner.norm(e)
            if new_norm / norm <= a:
                norm = new_norm
                ctr += 1
                if ctr > max_iter:
                    raise Exception("Gram-Schmidt: max number of "
                                    "iterations reached.")
            else:
                flag = 1

        return [e / new_norm, new_norm]

    def make_basis(self, hs, a=0.5, max_iter=3):
        """Given a set of functions, hs, find the corresponding orthonormal
        set of basis functions."""

        dim = np.shape(hs)
        basis = np.empty_like(hs)
        basis[0] = self.inner.normalize(hs[0])

        for ii in range(1, dim[0]):
            basis[ii], _ = self.add_basis(hs[ii], basis[:ii],
                                          a=a, max_iter=max_iter)

        return np.array(basis)


class GramSchmidt(_IteratedModifiedGramSchmidt):
    """Class for building an orthonormal basis using the
    iterated, modified Gram-Schmidt procedure.

    Input
    -----
    vectors    -- set of vectors to orthonormalize
    inner      -- instance of Integration class
    normsQ     -- norms of input vectors (default is False)

    Methods
    -------
    iter -- one iteration of the iterated, modified
            Gram-Schmidt algorithm
    make -- orthonormalize all the input vectors

    Examples
    --------
    Create an instance of the Basis class for functions with
    unit norm::

    >>> basis = rp.algorithms.Basis(vectors, inner)

    Build an orthonormal basis by running

    >>> basis.make()

    Output is an array of orthonormal basis elements.
    """

    def __init__(self, vectors, integration):
        self.Nbasis, self.Nnodes = np.shape(vectors)
        self.functions = np.asarray(vectors)

        _IteratedModifiedGramSchmidt.__init__(self, integration)

    def iter(self, step, h, a=0.5, max_iter=3):
        """One iteration of the iterated, modified Gram-Schmidt algorithm"""
        ans = self.add_basis(h, self.basis[:step], a=a, max_iter=max_iter)
        self.basis[step], _ = ans

    def build_basis(self, a=0.5, max_iter=3):
        """Find the corresponding orthonormal set of basis functions."""

        _, svds, _ = np.linalg.svd(self.functions)

        if min(svds) < 5e-15:
            raise Exception("Functions are not linearly independent.")

        else:
            self.basis = np.empty(
                (self.Nbasis, self.Nnodes), dtype=self.functions.dtype
            )

            self.basis[0] = self.inner.normalize(self.functions[0])

            for ii in range(1, self.Nbasis):
                self.iter(ii, self.functions[ii], a=a, max_iter=max_iter)

            return np.array(self.basis)


#############################################
# Class for reduced basis greedy algorithms #
#############################################

class ReducedBasis(_IteratedModifiedGramSchmidt):
    """Class for standard reduced basis greedy algorithm.
    """
    def __init__(self, interval, num, rule="riemann"):

        comp_integration = Integration(interval=interval, num=num,
                                       rule=rule)
        self.inner = comp_integration

        _IteratedModifiedGramSchmidt.__init__(self, comp_integration)

    def build_rb(self, training_space, index_seed, tol, verbose=False):
        """Make a reduced basis using the standard greedy algorithm.
        """

        self.loss = self.proj_errors_from_alpha

        self.seed(training_space, index_seed)

        if verbose:
            print("\nStep", "\t", "Error")

        nn = 0
        sigma = 1.
        while sigma > tol:
            nn += 1
            errs = self.loss(self.alpha[: nn], norms=self._norms)
            next_index = np.argmax(errs)

            if next_index in self.indices:
                self.size = nn -1
                self.trim(self.size)
                raise Exception("Index already selected: Exiting greedy "
                      "algorithm.")
            else:
                self.indices[nn] = next_index
                self.errors[nn - 1] = errs[next_index]
                self.basis[nn], self.basisnorms[nn] = self.add_basis(
                    training_space[self.indices[nn]], self.basis[:nn]
                )
                self.alpha[nn] = self.alpha_arr(self.basis[nn],
                                                training_space)
            sigma = errs[next_index]
            if verbose:
                print(nn, "\t", sigma)
        # Trim excess allocated entries
        self.size = nn
        self.trim(self.size)

    def seed(self, training_space, seed):
        """Seed the greedy algorithm.
        """

        # Extract dimensions of training space data
        try:
            Npoints, Nsamples = np.shape(np.asarray(training_space))
        except(ValueError):
            print("Unexpected dimensions for training space.")

        # Validate inputs
        assert Nsamples == np.size(
            self.inner.weights
        ), "Number of samples is inconsistent with quadrature rule."

        # Allocate memory for greedy algorithm arrays
        dtype = type(np.asarray(training_space).flatten()[0])
        self.allocate(Npoints, Nsamples, dtype=dtype)

        # Compute norms of training space data
        self._norms = np.array([self.inner.norm(tt) for tt in training_space])

        # Seed
        self.indices[0] = seed
        self.basis[0] = training_space[seed] / self._norms[seed]
        self.basisnorms[0] = self._norms[seed]
        self.alpha[0] = self.alpha_arr(self.basis[0], training_space)

# --- Aux functions ----------------------------------------------------------

    def allocate(self, Npoints, Nquads, dtype="complex"):
        """Allocate memory for numpy arrays used for making reduced basis"""
        self.errors = np.empty(Npoints, dtype="double")
        self.indices = np.empty(Npoints, dtype="int")
        self.basis = np.empty((Npoints, Nquads), dtype=dtype)
        self.basisnorms = np.empty(Npoints, dtype="double")
        self.alpha = np.empty((Npoints, Npoints), dtype=dtype)

    def alpha_arr(self, e, hs):
        """Inner products of a basis function e with an array of functions
        hs"""
        return np.array([self.inner.dot(e, hh) for hh in hs])

    def proj_error_from_basis(self, basis, h):
        """Square of the projection error of a function h on basis"""
        norm = self.inner.norm(h).real
        dim = len(basis[:, 0])
        ans = 0.0
        for ii in range(dim):
            ans += np.abs(self.inner.dot(basis[ii], h)) ** 2
        return norm ** 2 - ans

    def proj_errors_from_basis(self, basis, hs):
        """Square of the projection error of functions hs on basis"""
        return [self.proj_error_from_basis(basis, hh) for hh in hs]

    def proj_errors_from_alpha(self, alpha, norms=None):
        """Square of the projection error of a function h on basis in terms
        of pre-computed alpha matrix"""
        if norms is None:
            norms = np.ones(len(alpha[0]), dtype="double")
        ans = 0.0
        for aa in alpha:
            ans += np.abs(aa) ** 2
        return norms ** 2 - ans

    def projection_from_basis(self, h, basis):
        """Project a function h onto the basis functions"""
        ans = 0.0
        for ee in basis:
            ans += ee * self.inner.dot(ee, h)
        return ans

    def projection_from_alpha(self, alpha, basis):
        """Project a function h onto the basis functions using the
        precomputed quantity alpha = <basis, h>"""
        ans = 0.0
        for ii, ee in basis:
            ans += ee * alpha[ii]
        return ans

    def _Alpha(self, E, e, alpha):
        return self.inner.dot(E, self.projection_from_alpha(alpha, e))

    def Alpha_arr(self, E, e, alpha):
        return np.array([self._Alpha(EE, e, alpha) for EE in E])

    def trim(self, num):
        """Trim arrays to have size num"""
        self.errors = self.errors[:num]
        self.indices = self.indices[:num]
        self.basis = self.basis[:num]
        self.alpha = self.alpha[:num]
