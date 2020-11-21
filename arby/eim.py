# --- eim.py ---

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

import numpy as np

# =========================================
# Class for Empirical Interpolation Method
# =========================================


class EmpiricalMethods:
    """Build an Empirical Interpolant matrix operator from a given basis."""

    def __init__(self, Basis):
        self.Basis = np.array(Basis)
        self.Nbasis, self.Nsamples = self.Basis.shape
        self.V_matrix = None
        self.eim_nodes = None
        self.Interpolant = None

    # ==== Classic EIM ========================================================

    def build_eim(self, verbose=False):
        """Find EIM nodes and build Empirical Interpolant operator."""
        nodes = []
        V_matrix = None
        first_node = np.argmax(np.abs(self.Basis[0]))
        nodes.append(first_node)

        if verbose:
            print(first_node)

        for i in range(1, self.Nbasis):
            V_matrix = self.next_vandermonde(nodes, V_matrix)
            base_at_nodes = [self.Basis[i, t] for t in nodes]
            invV_matrix = np.linalg.inv(V_matrix)
            step_basis = self.Basis[:i]
            basis_interpolant = base_at_nodes @ invV_matrix @ step_basis
            residual = self.Basis[i] - basis_interpolant
            new_node = np.argmax(abs(residual))

            if verbose:
                print(new_node)
            nodes.append(new_node)

        V_matrix = np.array(self.next_vandermonde(nodes, V_matrix))
        self.V_matrix = V_matrix.transpose()
        invV_matrix = np.linalg.inv(self.V_matrix)
        self.Interpolant = self.Basis.transpose() @ invV_matrix
        self.eim_nodes = nodes

    # ==== Auxiliary functions ================================================

    def next_vandermonde(self, nodes, vandermonde=None):
        """Build the next V-matrix from the previous one."""
        if vandermonde is None:
            vandermonde = [[self.Basis[0, nodes[0]]]]
            return vandermonde

        n = len(vandermonde)
        new_node = nodes[-1]
        for i in range(n):
            vandermonde[i].append(self.Basis[i, new_node])
        vertical_vector = [self.Basis[n, nodes[j]] for j in range(n)]
        vertical_vector.append(self.Basis[n, new_node])
        vandermonde.append(vertical_vector)
        return vandermonde

    # ==== Validation functions ===============================================
