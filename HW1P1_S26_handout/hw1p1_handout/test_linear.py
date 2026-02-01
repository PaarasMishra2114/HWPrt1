import numpy as np
from mytorch.nn.linear import Linear

A = np.array([
    [-4., -3.],
    [-2., -1.],
    [0., 1.],
    [2., 3.]], dtype="f")

W = np.array([
    [-2., -1.],
    [0., 1.],
    [2., 3.]], dtype="f")

b = np.array([
    [-1.],
    [0.],
    [1.]], dtype="f")

linear = Linear(2, 3, debug=True)
print("Initial W shape:", linear.W.shape)
print("Initial b shape:", linear.b.shape)

linear.W = W
linear.b = b

print("After assignment W shape:", linear.W.shape)
print("After assignment b shape:", linear.b.shape)

try:
    Z = linear.forward(A)
    print("Z =\n", Z)
except Exception as e:
    print("Error:", e)
    print("A shape:", A.shape)
    print("W shape:", linear.W.shape)
    print("b shape:", linear.b.shape)
