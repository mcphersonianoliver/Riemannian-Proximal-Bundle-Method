import numpy as np
import sys
sys.path.append('src')
from RiemannianProximalBundle import RProximalBundle

# Quick test to see if the basic functionality works
print("Testing basic imports and functionality...")

# Test SPD manifold basic operations
class SPDManifold:
    def __init__(self, n):
        self.n = n

    def inner_product(self, X, U, V):
        from scipy.linalg import inv
        X_inv = inv(X)
        return np.trace(X_inv @ U @ X_inv @ V)

    def norm(self, X, V):
        return np.sqrt(self.inner_product(X, V, V))

# Test basic manifold operations
manifold = SPDManifold(2)
X = np.eye(2) * 2
V = np.array([[1, 0.5], [0.5, 1]])

print(f"Inner product test: {manifold.inner_product(X, V, V)}")
print(f"Norm test: {manifold.norm(X, V)}")

print("Basic functionality working!")