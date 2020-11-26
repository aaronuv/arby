# - greedy.py -

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

import numpy as np
from .integrals import Integration
from random import randint  # noqa: F401
from scipy.interpolate import splrep, splev

# ===================================
#    Iterated-Modified Gram-Schmidt
#      Orthonormalization Method
# ===================================


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
                raise Exception("Gram-Schmidt: max number of iterations"
                                "reached.")
        else:
            flag = 1

    return [e / new_norm, new_norm]


def GramSchmidt(vectors, integration, a=0.5, max_iter=3):
    """Class for building an orthonormal basis using the iterated,
    modified Gram-Schmidt procedure.
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
        projected_element, _ = GS_add_element(
            new_basis_elem, ortho_basis, integration, a, max_iter
        )
        ortho_basis.append(projected_element)
    basis = np.array(ortho_basis)

    return basis


# =================================
# Class for Reduced Order Modeling
# =================================


class ReducedOrderModeling:
    def __init__(
        self,
        training_space=None,
        physical_interval=None,
        parameter_interval=None,
        basis=None,
    ):
        # Check non empty inputs aiming for reduced order model

        if training_space is not None and physical_interval is not None:
            self.training_space = np.asarray(training_space)
            self.Ntrain, self.Nsamples = self.training_space.shape
            self.physical_interval = np.asarray(physical_interval)
            if self.Nsamples != self.physical_interval.size:
                    raise ValueError(
                        "Number of samples for each training function must be "
                        "equal to number of physical points."
                    )
            if parameter_interval is not None:
                self.parameter_interval = np.asarray(parameter_interval)
                if self.Ntrain != self.parameter_interval.size:
                    raise ValueError(
                        "Number of training functions must be "
                        "equal to number of parameter points."
                    )

        self.basis = basis
        self.integration = None

        self.v_matrix = None
        self.eim_nodes = None
        self.interpolant = None

    # ==== Reduced Basis Method ===============================================

    def build_reduced_basis(
        self, rule="riemann", index_seed=0, tol=1e-12, verbose=False
    ):
        phys_min = self.physical_interval.min()
        phys_max = self.physical_interval.max()
        self.integration = Integration(
            interval=[phys_min, phys_max], num=self.Nsamples, rule=rule
        )

        self.loss = self.projection_error  # no me convence este atributo

        # If seed gives a null function, iterate to a new seed
        seed_function = self.training_space[index_seed]
        zero_function = np.zeros_like(seed_function)
        while np.allclose(seed_function, zero_function):
            index_seed = np.randint(1, self.Ntrain)
            seed_function = self.training_space[index_seed]

        # ====== Seed the greedy algorithm and allocate memory ======

        # Validate inputs
        assert self.Nsamples == np.size(
            self.integration.weights
        ), "Number of samples is inconsistent with quadrature rule."
        # Allocate memory for greedy algorithm arrays
        self.allocate(self.Ntrain, self.Nsamples,
                      dtype=self.training_space.dtype)

        # Compute norms of the training space data
        self._norms = np.array(
            [self.integration.norm(tt) for tt in self.training_space]
        )

        # Seed
        self.greedy_indices = [index_seed]
        self.basis[0] = (self.training_space[index_seed] /
                         self._norms[index_seed])
        self.basisnorms[0] = self._norms[index_seed]
        self.proj_matrix[0] = self.integration.dot(self.basis[0],
                                                   self.training_space)

        # ====== Start greedy loop ======

        if verbose:
            print("\n Step", "\t", "Error")

        nn = 0
        sigma = 1.0
        while sigma > tol:
            nn += 1
            errs = self.loss(self.proj_matrix[:nn], norms=self._norms)
            next_index = np.argmax(errs)

            if next_index in self.greedy_indices:
                self.trim(nn - 1)
                raise Exception("Index already selected: exiting "
                                "greedy algorithm.")

            self.greedy_indices.append(next_index)
            self.errors[nn - 1] = errs[next_index]
            self.basis[nn], self.basisnorms[nn] = GS_add_element(
                self.training_space[self.greedy_indices[nn]],
                self.basis[:nn],
                self.integration,
                a=0.5,
                max_iter=3,
            )
            self.proj_matrix[nn] = self.integration.dot(
                self.basis[nn], self.training_space
            )
            sigma = errs[next_index]
            if verbose:
                print(nn, "\t", sigma)
        # Trim excess allocated entries
        self.trim(nn)

    # ====== Empirical Interpolation Method ===================================

    def build_eim(self, verbose=False):
        """Find EIM nodes and build Empirical Interpolant operator."""

        if self.basis is None:
            raise AttributeError("There is no basis to work with.")

        self.Nbasis, _ = self.basis.shape

        nodes = []
        v_matrix = None
        first_node = np.argmax(np.abs(self.basis[0]))
        nodes.append(first_node)

        if verbose:
            print(first_node)

        for i in range(1, self.Nbasis):
            v_matrix = self.next_vandermonde(nodes, v_matrix)
            base_at_nodes = [self.basis[i, t] for t in nodes]
            invV_matrix = np.linalg.inv(v_matrix)
            step_basis = self.basis[:i]
            basis_interpolant = base_at_nodes @ invV_matrix @ step_basis
            residual = self.basis[i] - basis_interpolant
            new_node = np.argmax(abs(residual))

            if verbose:
                print(new_node)
            nodes.append(new_node)

        v_matrix = np.array(self.next_vandermonde(nodes, v_matrix))
        self.v_matrix = v_matrix.transpose()
        invV_matrix = np.linalg.inv(self.v_matrix)
        self.interpolant = self.basis.transpose() @ invV_matrix
        self.eim_nodes = nodes

    def build_splines(
        self,
        rule="riemann",
        index_seed=0,
        tol=1e-12,
        verbose=False,
        built_basis=False,
        poly_deg=3,
    ):

        if not built_basis:
            self.build_reduced_basis(
                rule=rule, index_seed=index_seed, tol=tol, verbose=verbose
            )

        self.build_eim()

        training_compressed = np.empty(
            (self.Ntrain, self.basis.size), dtype=self.training_space.dtype
        )
        for i in range(self.Ntrain):
            for j, node in enumerate(self.eim_nodes):
                training_compressed[i, j] = self.training_space[i, node]
        h_in_nodes_splined = []
        for i in range(self.Nbasis):
            h_in_nodes_splined.append(
                splrep(self.parameter_interval,
                       training_compressed[:, i],
                       k=poly_deg)
                )

        self.spline_model = h_in_nodes_splined

    def surrogate(self, parameter):
        h_surr_at_nodes = np.array(
            [splev(parameter, spline) for spline in self.spline_model]
        )
        h_surrogate = self.interpolant @ h_surr_at_nodes

        return h_surrogate

    # ==== Auxiliary functions ================================================

    def allocate(self, Npoints, Nquads, dtype="complex"):
        """Allocate memory for numpy arrays used for making reduced basis"""
        self.errors = np.empty(Npoints, dtype="double")
        self.basis = np.empty((Npoints, Nquads), dtype=dtype)
        self.basisnorms = np.empty(Npoints, dtype="double")
        self.proj_matrix = np.empty((Npoints, Npoints), dtype=dtype)

    def projection_error(self, proj_matrix, norms):
        """Square of the projection error of a function h on basis in terms
        of pre-computed projection matrix"""
        proj_norms = np.array(
            [np.linalg.norm(proj_matrix[:, i]) for i in range(self.Ntrain)]
        )
        return norms ** 2 - proj_norms ** 2

    def trim(self, num):
        """Trim arrays to have size num"""
        self.errors = self.errors[:num]
        self.basis = self.basis[:num]
        self.proj_matrix = self.proj_matrix[:num]

    def next_vandermonde(self, nodes, vandermonde=None):
        """Build the next V-matrix from the previous one."""
        if vandermonde is None:
            vandermonde = [[self.basis[0, nodes[0]]]]
            return vandermonde

        n = len(vandermonde)
        new_node = nodes[-1]
        for i in range(n):
            vandermonde[i].append(self.basis[i, new_node])
        vertical_vector = [self.basis[n, nodes[j]] for j in range(n)]
        vertical_vector.append(self.basis[n, new_node])
        vandermonde.append(vertical_vector)
        return vandermonde

    # ==== Validation functions ===============================================

    # ~ # This function inherites homology operation property from .dot method
    # ~ # from integrals.py. Then h_vector may be an array of functions.
    # ~ def proj_error_from_basis(self, basis, h_vector):
    # ~ """Square of the projection error of a function h_vector on basis."""
    # ~ h_vector_sqnorm = self.integration.norm(h_vector).real
    # ~ inner_prod = np.array(
    # ~ [self.integration.dot(basis_elem, h_vector)
    # ~ for basis_elem in basis]
    # ~ )
    # ~ return h_vector_sqnorm ** 2 - np.linalg.norm(inner_prod) ** 2

    # ~ def project_on_basis(self, h, basis):
    # ~ """Project a function h onto the basis functions"""
    # ~ projected_function = 0.0
    # ~ for e in basis:
    # ~ projected_function += e * self.integration.dot(e, h)
    # ~ return projected_function

    # ~ def interpolate(self, h):
    # ~ """Interpolate a function h at EIM nodes."""
    # ~ assert len(h) == self.Nsamples, (
    # ~ "Size of vector h doesn't " "match grid size of basis elements."
    # ~ )
    # ~ h_at_nodes = np.array([h[eim_node] for eim_node in self.eim_nodes])
    # ~ h_interpolated = self.interpolant @ h_at_nodes
    # ~ return h_interpolated
