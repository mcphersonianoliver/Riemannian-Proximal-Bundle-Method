# %% Experiment 1: PSD Matrices
import autograd.numpy as anp
from pymanopt.manifolds import SymmetricPositiveDefinite
import numpy as np
from src.RiemannianProximalBundle import RProximalBundle

# %%
# set the manifold a priori
dim = 50
manifold = SymmetricPositiveDefinite(dim)

# generate true median
true_median = manifold.random_point()
# generate random PSD matrices to take median of
num_of_perturbations = 100
data = []
for _ in range(num_of_perturbations):
    random_tangent_vector = manifold.random_tangent_vector(true_median)
    ranndom_scale = np.random.uniform(0.5, 10)
    random_tangent_vector *= ranndom_scale
    perturbation_1 = manifold.exp(true_median, random_tangent_vector)
    perturbation_2 = manifold.exp(true_median, -random_tangent_vector)
    data.append(perturbation_1)
    data.append(perturbation_2)

# %% Setting up functions
# Riemannian median cost function
def cost_set_up(point, data):
    # riemannian median cost function given points

    return sum([manifold.dist(point, x) for x in data]) / len(data)

def cost(point):
    # compute the cost function
    return cost_set_up(point, data)

# Riemannian subgradient operator
def subgradient_set_up(point, data):
    # compute the Riemannian subgradient of the cost function
    grad = anp.zeros_like(point)
    for x in data:
        log_point = manifold.log(point, x)
        grad += - (log_point)/ manifold.norm(point, log_point)
    return grad / len(data)

def subgradient(point):
    # compute the subgradient
    return subgradient_set_up(point, data)

# %% Initialize points and set up the optimizer
# Initialize a random point on the manifold
initial_point = manifold.random_point()
initial_objective = cost(initial_point)
initial_subgradient = subgradient(initial_point)
true_objective = cost(true_median)

# Set up the optimizer
optimizer = RProximalBundle(
    manifold=manifold,
    retraction_map = manifold.exp,
    transport_map = manifold.transport,
    objective_function=cost,
    subgradient = subgradient, true_min_obj=true_objective,
    initial_point=initial_point,
    initial_objective=initial_objective,
    initial_subgradient=initial_subgradient,
    trust_parameter=0.2,
    transport_error =0.5,
    retraction_error= 1,
)
# %%
optimizer.run()
# %%
optimizer.plot_objective_versus_iter()
# %%
