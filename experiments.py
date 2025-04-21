# %% Experiment 0: checking if manopt is installed correctly - running simple example
import autograd.numpy as anp
import pymanopt

anp.random.seed(42)

dim = 3
manifold = pymanopt.manifolds.Sphere(dim)

matrix = anp.random.normal(size=(dim, dim))
matrix = 0.5 * (matrix + matrix.T)

@pymanopt.function.autograd(manifold)
def cost(point):
    return -point @ matrix @ point

problem = pymanopt.Problem(manifold, cost)

optimizer = pymanopt.optimizers.SteepestDescent()
result = optimizer.run(problem)

eigenvalues, eigenvectors = anp.linalg.eig(matrix)
dominant_eigenvector = eigenvectors[:, eigenvalues.argmax()]

print("Dominant eigenvector:", dominant_eigenvector)
print("Pymanopt solution:", result.point)

# %% Experiment 1: PSD matrices
from pymanopt.manifolds import SymmetricPositiveDefinite

# set the manifold a priori
dim = 3
manifold = SymmetricPositiveDefinite(dim)

samples = 10 
# generate |samples| random PSD matrices
data = []
for _ in range(samples):
    data_point = manifold.random_point()
    data.append(data_point)

# Riemannian median cost function
def cost(point, data):
    # riemannian median cost function given points

    return sum([manifold.dist(point, x) for x in data]) / len(data)

# %%

print("Data shape:", len(data), data[0].shape)
# check if the matrices are PSD
for i, matrix in enumerate(data):
    eigenvalues, _ = anp.linalg.eig(matrix)
    print(f"Matrix {i} eigenvalues:", eigenvalues)
    assert all(eigenvalues >= 0), f"Matrix {i} is not PSD"

# print matrices
for i, matrix in enumerate(data):
    print(f"Matrix {i}:\n", matrix)

# %%
point_a, point_b = data[0], data[1]

dist_ab = manifold.dist(point_a, point_b)
print("Distance between point_a and point_b:", dist_ab)
# %%


# %%
# create a random point on the manifold
point = manifold.random_point()
print("Random point on manifold:", point)

objective = cost(point,data)
print("Objective value:", objective)

# %%
