# - greedy.py -

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE
"""
Classes for building reduced basis greedy algorithms
"""

import numpy as np
from .integrals import Integration
from random import randint

#############################################
# Class for iterated, modified Gram-Schmidt #
#      orthonormalization of functions      #
#############################################


def GS_add_element(h, basis, integration, a, max_iter):
    """
    Iterated modified Gram-Schmidt algorithm for building an orthonormal
    basis. Algorithm from Hoffman, `Iterative Algorithms for Gram-Schmidt
    Orthogonalization`. Given a function, h, find the corresponding basis
    function orthonormal to all previous ones.
    """
    norm = integration.norm(h)
    e = h / norm

    flag, ctr = 0, 1
    while flag == 0:
        for b in basis:
            e -= b * integration.dot(b, e)
        new_norm = integration.norm(e)
        if new_norm / norm <= a:
            norm = new_norm
            ctr += 1
            if ctr > max_iter:
                raise Exception(
                    "Gram-Schmidt: max number of iterations reached."
                )
        else:
            flag = 1

    return [e / new_norm, new_norm]



def GramSchmidt(vectors, integration, a=0.5, max_iter=3):
    """Class for building an orthonormal basis using the
    iterated, modified Gram-Schmidt procedure.

    Input
    -----
    vectors: set of vectors to orthonormalize
    integration: instance of Integration class that defines the inner product

    Output is an array of orthonormal basis elements.
    """
    
    Nbasis, Nnodes = np.shape(vectors)
    functions = np.asarray(vectors)

    _, svds, _ = np.linalg.svd(functions)

    linear_indep_tol = 5e-15
    if min(svds) < linear_indep_tol:
        raise Exception("Functions are not linearly independent.")

    ortho_basis = []
    # First element of the basis is special, it's just normalized
    ortho_basis.append(integration.normalize(functions[0]))
    # For the rest of basis elements add them one by one by extending basis
    for new_basis_elem in functions[1:]:
        projected_element, _ = GS_add_element(new_basis_elem, ortho_basis,
                                              integration, a, max_iter)
        ortho_basis.append(projected_element)
    basis = np.array(ortho_basis)

    return basis


#############################################
# Class for reduced basis greedy algorithms #
#############################################


class ReducedBasis:
    """Class for standard reduced basis greedy algorithm."""

    def __init__(self, training_space, interval, rule="riemann"):
        self.training = np.array(training_space)
        self.Ntrain, self.Nsamples = training_space.shape
        self.loss = self.proj_errors_from_alpha
        self.integration = Integration(interval=interval, num=self.Nsamples,
                                       rule=rule)

    def build_rb(self, index_seed=0, tol=1e-12, verbose=False):
        """Make a reduced basis using the standard greedy algorithm."""
        # In seed gives a null function, iterate to a new seed
        seed_function = self.training[index_seed]
        zero_function = np.zeros_like(seed_function)
        while np.allclose(seed_function, zero_function):
            index_seed = np.randint(1, Ntrain)
            seed_function = self.training[index_seed]

        # ====== Seed the greedy algorithm and allocate memory ================
        # Validate inputs
        assert self.Nsamples == np.size(
            self.integration.weights
        ), "Number of samples is inconsistent with quadrature rule."

        # Allocate memory for greedy algorithm arrays
        dtype = type(np.asarray(self.training).flatten()[0])
        self.allocate(self.Ntrain, self.Nsamples, dtype=dtype)

        # Compute norms of training space data
        self._norms = np.array([self.integration.norm(tt) for tt
                               in self.training])

        # Seed
        self.indices[0] = index_seed
        self.basis[0] = self.training[index_seed] / self._norms[index_seed]
        self.basisnorms[0] = self._norms[index_seed]
        self.alpha[0] = self.alpha_arr(self.basis[0], self.training)
        # =====================================================================

        # ===== Start greedy loops ============================================
        if verbose:
            print("\nStep", "\t", "Error")

        nn = 0
        sigma = 1.0
        while sigma > tol:
            nn += 1
            errs = self.loss(self.alpha[:nn], norms=self._norms)
            next_index = np.argmax(errs)

            if next_index in self.indices:
                self.size = nn - 1
                self.trim(self.size)
                raise Exception(
                    "Index already selected: Exiting greedy algorithm."
                )
            else:
                self.indices[nn] = next_index
                self.errors[nn - 1] = errs[next_index]
                self.basis[nn], self.basisnorms[nn] = GS_add_element(
                    self.training[self.indices[nn]], self.basis[:nn],
                    self.integration, a=0.5, max_iter=3
                )
                self.alpha[nn] = self.alpha_arr(self.basis[nn], self.training)
            sigma = errs[next_index]
            if verbose:
                print(nn, "\t", sigma)
        # Trim excess allocated entries
        self.size = nn
        self.trim(self.size)

        
    # ==== Auxiliary functions ===============================================

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
        return np.array([self.integration.dot(e, hh) for hh in hs])

    def proj_error_from_basis(self, basis, h_vector):
        """Square of the projection error of a function h on basis"""
        h_vector_sqnorm = self.integration.norm(h_vector).real ** 2
        inner_prod = np.array([self.integration.dot(basis_elem, h_vector)**2
                                                for basis_elem in basis])
        return h_vector_sqnorm - np.linalg.norm(inner_prod)


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
            ans += ee * self.integration.dot(ee, h)
        return ans

    def projection_from_alpha(self, alpha, basis):
        """Project a function h onto the basis functions using the
        precomputed quantity alpha = <basis, h>"""
        ans = 0.0
        for ii, ee in basis:
            ans += ee * alpha[ii]
        return ans

    def _Alpha(self, E, e, alpha):
        return self.integration.dot(E, self.projection_from_alpha(alpha, e))

    def Alpha_arr(self, E, e, alpha):
        return np.array([self._Alpha(EE, e, alpha) for EE in E])

    def trim(self, num):
        """Trim arrays to have size num"""
        self.errors = self.errors[:num]
        self.indices = self.indices[:num]
        self.basis = self.basis[:num]
        self.alpha = self.alpha[:num]
