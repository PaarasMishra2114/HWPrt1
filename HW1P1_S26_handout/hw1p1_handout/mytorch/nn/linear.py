import numpy as np


class Linear:
    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros (Hint: check np.zeros method)
        Read the writeup (Hint: Linear Layer Section Table) to identify the right shapes for `W` and `b`.
        """
        self.debug = debug
        self.W = np.zeros((out_features, in_features), dtype='f')
        self.b = np.zeros((out_features, 1), dtype='f')

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output of linear layer with shape (N, C1)

        Read the writeup (Hint: Linear Layer Section) for implementation details for `Z`
        """
        self.A = A
        self.N = A.shape[0]

        self.ones = np.ones((self.N, 1))

        Z = A @ self.W.T + self.ones @ self.b.T
        
        if self.debug:
            self.Z = Z
        
        return Z

    def backward(self, dLdZ):
        """
        :param dLdZ: Gradient of loss wrt output Z (N, C1)
        :return: Gradient of loss wrt input A (N, C0)

        Read the writeup (Hint: Linear Layer Section) for implementation details below variables.
        """
        dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = (self.ones.T @ dLdZ).T

        if self.debug:
            self.dLdA = dLdA

        return dLdA
