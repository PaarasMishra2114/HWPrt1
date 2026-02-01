import numpy as np

Z = np.array([
    [-4, -3],
    [-2, -1],
    [0, 1],
    [2, 3]
])

beta = 1.0
sigmoid = 1 / (1 + np.exp(-beta * Z))

dLdA = np.array([
    [1.0,   1.0,],
    [3.0,   1.0,],
    [2.0,   0.0,],
    [0.0,  -1.0,]
])

# Method 1: Z * sigmoid * (1-sigmoid)
dLdbeta_v1 = np.sum(dLdA * Z * sigmoid * (1 - sigmoid))
print("Method 1 (Z * sigmoid * (1-sigmoid)):", dLdbeta_v1)

# Method 2: Maybe it's Z^2 * sigmoid * (1-sigmoid)
dLdbeta_v2 = np.sum(dLdA * Z * Z * sigmoid * (1 - sigmoid))
print("Method 2 (Z^2 * sigmoid * (1-sigmoid)):", dLdbeta_v2)

# Check expected
print("Expected: 1.7391382")

# Method 3: What if it's the opposite sign?
dLdbeta_v3 = -dLdbeta_v1
print("Method 3 (-dLdbeta_v1):", dLdbeta_v3)
