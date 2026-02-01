from mytorch.nn import Swish
import numpy as np

Z = np.array([
    [-4, -3],
    [-2, -1],
    [0, 1],
    [2, 3]
])

swish = Swish()
A = swish.forward(Z)

dLdA = np.array([
    [1.0,   1.0,],
    [3.0,   1.0,],
    [2.0,   0.0,],
    [0.0,  -1.0,]
])

dLdZ = swish.backward(dLdA)
dLdbeta = swish.dLdbeta

print("dLdbeta =", dLdbeta)
print("dLdbeta type:", type(dLdbeta))
print("Expected dLdbeta = 1.7391382")

# Debug: compute step by step
sigmoid = 1 / (1 + np.exp(-swish.beta * Z))
dLdbeta_manual = np.sum(dLdA * Z * sigmoid * (1 - sigmoid))
print("Manual dLdbeta =", dLdbeta_manual)
