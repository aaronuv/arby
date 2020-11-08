# --- eim.py ---

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE

import numpy as np
import copy


def two_norm(v):
    return np.linalg.norm(v, 2)


class EmpiricalMethods:
    """
    A class to select empirical nodes from a basis.

    """

    def __init__(self, B):
        self.B = B
        self.n = len(self.B)
        self.L = len(self.B[0])

    #################################
    # Construct the next Vandermonde matrix with eim nodes:
    # V(T0,...,Tn-1,Tn)

    def Vandermonde(self, van, nodes):
        v = copy.deepcopy(van)
        dim = len(v)
        new_node = nodes[dim]
        for i in range(dim):
            v[i].append(self.B[i][new_node])
        vertical = [self.B[dim][nodes[j]] for j in range(dim)]
        vertical.append(self.B[dim][new_node])
        v.append(vertical)
        return v

    ##########################################################
    # Function for EIM Classic #
    ##########################################################

    def eim(self, verbose=False):

        nodes = []
        lebesgue_norms = []
        k = []
        matrices = []

        new_node = np.argmax(np.abs(self.B[0]))
        if verbose:
            print(new_node)
        nodes.append(new_node)
        V = [[self.B[0][new_node]]]

        for i in range(1, self.n):
            matrices.append(V)
            base_in_nodes = [self.B[i][t] for t in nodes]
            invV = np.linalg.inv(V)
            leb = two_norm(invV)
            norm = two_norm(V)
            k.append(norm * leb)
            lebesgue_norms.append(leb)
            C = np.matmul(base_in_nodes, invV)
            base = self.B[:i]
            interpolant = np.matmul(C, base)
            residual = self.B[i] - interpolant
            new_node = np.argmax(abs(residual))
            if verbose:
                print(new_node)
            nodes.append(new_node)
            V = self.Vandermonde(V, nodes)
        matrices.append(V)
        invV = np.linalg.inv(V)
        leb = two_norm(invV)
        norm = two_norm(V)
        k.append(norm * leb)
        lebesgue_norms.append(leb)

        self.indices = nodes
        self.lebesgue = lebesgue_norms
        self.k = k
        self.matrices = matrices
