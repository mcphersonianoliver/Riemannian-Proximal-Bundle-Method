# %% general imports and set up Riemannian median function and subgradients
import autograd.numpy as anp
from pymanopt.manifolds import SymmetricPositiveDefinite
import numpy as np
from src.RiemannianProximalBundle import RProximalBundle
import matplotlib.pyplot as plt

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

# %% Experiment 1: PSD Matrices of differing dimensions
dimensions_of_interest = [15, 30, 50, 75, 100]
relative_objective_gaps_vs_iterations = []
objective_gaps_vs_iterations = []
initial_objectives = []
descent_step_indices = []
null_step_indices = []
num_of_perturbations = 50
preset_scalings = []
for i in range(num_of_perturbations):
    preset_scalings.append(np.random.uniform(0.5, 10))
initialization_scaling = 20

for dim in dimensions_of_interest:
    manifold = SymmetricPositiveDefinite(dim)

    # generate true median
    base_point = manifold.random_point()
    # generate random PSD matrices to take median of
    data = []
    for _ in range(num_of_perturbations):
        random_tangent_vector = manifold.random_tangent_vector(base_point)
        # normalize the tangent vector
        random_tangent_vector /= manifold.norm(base_point, random_tangent_vector)
        ranndom_scale = preset_scalings[_]
        random_tangent_vector *= ranndom_scale
        perturbation = manifold.exp(base_point, random_tangent_vector)
        data.append(perturbation)

    # Initialize a random point on the manifold
    random_initial_vector = manifold.random_tangent_vector(base_point)
    random_initial_vector /= manifold.norm(base_point, random_initial_vector)
    random_initial_vector *= initialization_scaling
    initial_point = manifold.exp(base_point, random_initial_vector)
    
    # Compute the initial objective and subgradient
    initial_objective = cost(initial_point)
    initial_objectives.append(initial_objective)
    initial_subgradient = subgradient(initial_point)
    print(' ')
    print(f"Dimension: {dim}")

    # Set up the rBundAlg
    rBundAlg = RProximalBundle(
        manifold=manifold,
        retraction_map = manifold.exp,
        transport_map = manifold.transport,
        objective_function=cost,
        subgradient = subgradient,
        initial_point=initial_point,
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        proximal_parameter=0.2,
        trust_parameter=0.05,
        transport_error =0.5,
        retraction_error=0,
        know_minimizer=False
    )

    # Run optimization scheme
    rBundAlg.run()

    # Store objective gaps vs iterations
    descent_step_indices.append(rBundAlg.indices_of_descent_steps)
    print('Number of descent steps taken: ', len(rBundAlg.indices_of_descent_steps))
    relative_objective_gaps_vs_iterations.append(rBundAlg.relative_objective_history)
    objective_gaps_vs_iterations.append(rBundAlg.objective_history)
    null_step_indices.append(rBundAlg.indices_of_null_steps)
    print('Number of null steps taken: ', len(rBundAlg.indices_of_null_steps))
    print('Converged.')

# %% Plot results of experiment
# we plot the relative objective gaps vs iterations for each dimension as a line plot, and specify which steps were descent steps as
# a scatter plot on top of the line plot - this is done as a log plot: the values will be objective gaps divided by the initial objective

plt.figure(figsize=(10, 6))
for i, dim in enumerate(dimensions_of_interest):
    plt.plot(objective_gaps_vs_iterations[i], label=f"Dimension {dim}")
    # Plot descent steps
    for j in range(len(descent_step_indices[i])):
        plt.scatter(descent_step_indices[i][j], objective_gaps_vs_iterations[i][descent_step_indices[i][j]], color='red', s=10)
    # Plot null steps
    for j in range(len(null_step_indices[i])):
        plt.scatter(null_step_indices[i][j], objective_gaps_vs_iterations[i][null_step_indices[i][j]], color='blue', s=10)
plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Objective Gap (log scale)")
plt.title("Objective Gaps vs Iterations for Different Dimensions")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i, dim in enumerate(dimensions_of_interest):
    plt.plot(relative_objective_gaps_vs_iterations[i], label=f"Dimension {dim}")
    # Plot descent steps
    for j in range(len(descent_step_indices[i])):
        plt.scatter(descent_step_indices[i][j], relative_objective_gaps_vs_iterations[i][descent_step_indices[i][j]], color='red', s=10)
    # Plot null steps
    for j in range(len(null_step_indices[i])):
        plt.scatter(null_step_indices[i][j], relative_objective_gaps_vs_iterations[i][null_step_indices[i][j]], color='blue', s=10)
plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Relative Objective Gap (log scale)")
plt.title("Relative Objective Gaps vs Iterations for Different Dimensions")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %% Experiment 2: PSD Matrices with Different Proximal Parameters
# we fix one manifold, initial point, and data set
dimension = 50
manifold = SymmetricPositiveDefinite(dimension)
# generate true median
base_point = manifold.random_point()
# generate random PSD matrices to take median of
num_of_perturbations = 50
data = []
for _ in range(num_of_perturbations):
    random_tangent_vector = manifold.random_tangent_vector(base_point)
    ranndom_scale = np.random.uniform(0.5, 10)
    random_tangent_vector *= ranndom_scale
    perturbation_1 = manifold.exp(base_point, random_tangent_vector)
    data.append(perturbation_1)


# Initialize a random point on the manifold
random_initial_vector = manifold.random_tangent_vector(base_point)
random_initial_vector /= manifold.norm(base_point, random_initial_vector)
random_initial_vector *= initialization_scaling
initial_point = manifold.exp(base_point, random_initial_vector)
# Compute the initial objective and subgradient
initial_objective = cost(initial_point)
initial_subgradient = subgradient(initial_point)
print(' ')
print(f"Dimension: {dimension}")
print(f"Initial Cost: {initial_objective}")
print(' ')

# Preset proximal parameters
proximal_parameters = [0.1, 0.2, 0.5, 1, 2, 5]
relative_objective_gaps_vs_iterations = []
objective_gaps_vs_iterations = []
descent_step_indices = []
null_step_indices = []
for proximal_parameter in proximal_parameters:
    # Set up the rBundAlg
    rBundAlg = RProximalBundle(
        manifold=manifold,
        retraction_map = manifold.exp,
        transport_map = manifold.transport,
        objective_function=cost,
        subgradient = subgradient,
        initial_point=initial_point,
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        proximal_parameter=proximal_parameter,
        transport_error = 0.5,
        retraction_error= 0,
        trust_parameter=0.05
    )

    # Run optimization scheme
    rBundAlg.run()
    # Store objective gaps vs iterations
    descent_step_indices.append(rBundAlg.indices_of_descent_steps)
    null_step_indices.append(rBundAlg.indices_of_null_steps)
    relative_objective_gaps_vs_iterations.append(rBundAlg.relative_objective_history)
    objective_gaps_vs_iterations.append(rBundAlg.objective_history)
    print('Converged.')
    # Print the number of descent steps taken
    print(f"Number of descent steps taken with proximal parameter {proximal_parameter}: {len(rBundAlg.indices_of_descent_steps)}")

    print(' ')

# %%  Plot results of experiment
# we plot the objective gaps vs iterations for each proximal parameter as a line plot, and specify which steps were descent steps as
# a scatter plot on top of the line plot - this is done as a log plot
plt.figure(figsize=(10, 6))
for i, proximal_parameter in enumerate(proximal_parameters):
    plt.plot(objective_gaps_vs_iterations[i], label=f"Proximal Parameter {proximal_parameter}")
    for j in range(len(descent_step_indices[i])):
        plt.scatter(descent_step_indices[i][j], objective_gaps_vs_iterations[i][descent_step_indices[i][j]], color='red', s=10)
    # Plot null steps
    for j in range(len(null_step_indices[i])):
        plt.scatter(null_step_indices[i][j], objective_gaps_vs_iterations[i][null_step_indices[i][j]], color='blue', s=10)    

plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Objective Gap (log scale)")
plt.title("Objective Gaps vs Iterations for Different Proximal Parameters")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# we plot the relative objective gaps vs iterations for each proximal parameter as a line plot, and specify which steps were descent steps as
# a scatter plot on top of the line plot - this is done as a log plot
plt.figure(figsize=(10, 6))
for i, proximal_parameter in enumerate(proximal_parameters):
    plt.plot(relative_objective_gaps_vs_iterations[i], label=f"Proximal Parameter {proximal_parameter}")
    for j in range(len(descent_step_indices[i])):
        plt.scatter(descent_step_indices[i][j], objective_gaps_vs_iterations[i][descent_step_indices[i][j]], color='red', s=10)
    # Plot null steps
    for j in range(len(null_step_indices[i])):
        plt.scatter(null_step_indices[i][j], objective_gaps_vs_iterations[i][null_step_indices[i][j]], color='blue', s=10)
   
plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Relative Objective Gap (log scale)")
plt.title("Relative Objective Gaps vs Iterations for Different Proximal Parameters")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %% Make a plot without the descent steps
plt.figure(figsize=(10, 6))
for i, proximal_parameter in enumerate(proximal_parameters):
    plt.plot(objective_gaps_vs_iterations[i], label=f"Proximal Parameter {proximal_parameter}")
    # Plot null steps
    for j in range(len(null_step_indices[i])):
        plt.scatter(null_step_indices[i][j], objective_gaps_vs_iterations[i][null_step_indices[i][j]], color='blue', s=10)

plt.yscale('log')
plt.xlabel("Iterations")
plt.ylabel("Objective Gap (log scale)")
plt.title("Objective Gaps vs Iterations for Different Proximal Parameters")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# %% Experiment 3: PSD Matrices with and without adaptive step sizes for small initial proximal parameters 
# we fix one manifold, initial point, and data set
dimension = 30
manifold = SymmetricPositiveDefinite(dimension)
# generate true median
base_point = manifold.random_point()
# generate random PSD matrices to take median of
num_of_perturbations = 50
data = []
for _ in range(num_of_perturbations):
    random_tangent_vector = manifold.random_tangent_vector(base_point)
    ranndom_scale = np.random.uniform(0.5, 10)
    random_tangent_vector *= ranndom_scale
    perturbation_1 = manifold.exp(base_point, random_tangent_vector)
    data.append(perturbation_1)

# Initialize a random point on the manifold
random_initial_vector = manifold.random_tangent_vector(base_point)
random_initial_vector /= manifold.norm(base_point, random_initial_vector)
random_initial_vector *= initialization_scaling
initial_point = manifold.exp(base_point, random_initial_vector)
# Compute the initial objective and subgradient
initial_objective = cost(initial_point)
initial_subgradient = subgradient(initial_point)
print(' ')
print(f"Dimension: {dimension}")
print(f"Initial Cost: {initial_objective}")
print(' ')

# Preset proximal parameters
proximal_parameters = [0.05, 0.1, 0.15, 0.2]

# trackers for adaptive and non-adaptive step sizes
objective_gaps_vs_iterations_adaptive = []
relative_objective_gaps_vs_iterations_adaptive = []
descent_step_indices_adaptive = []
null_step_indices_adaptive = []
adaptive_step_indices_adaptive = []

objective_gaps_vs_iterations_non_adaptive = []
relative_objective_gaps_vs_iterations_non_adaptive = []
descent_step_indices_non_adaptive = []
null_step_indices_non_adaptive = []
adaptive_step_indices_non_adaptive = []

for proximal_parameter in proximal_parameters:
    # Set up the rBundAlg
    rBundAlg = RProximalBundle(
        manifold=manifold,
        retraction_map = manifold.exp,
        transport_map = manifold.transport,
        objective_function=cost,
        subgradient = subgradient,
        initial_point=initial_point,
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        proximal_parameter=proximal_parameter,
        transport_error = 2,
        retraction_error= 0,
        trust_parameter=0.05,
        adaptive_proximal=True
    )

    # Run optimization scheme
    rBundAlg.run()
    # Store objective gaps vs iterations
    descent_step_indices_adaptive.append(rBundAlg.indices_of_descent_steps)
    null_step_indices_adaptive.append(rBundAlg.indices_of_null_steps)
    relative_objective_gaps_vs_iterations_adaptive.append(rBundAlg.relative_objective_history)
    objective_gaps_vs_iterations_adaptive.append(rBundAlg.objective_history)
    adaptive_step_indices_adaptive.append(rBundAlg.indices_of_proximal_doubling_steps)
    print('Adaptive Converged.')

    rBundAlg_non_adaptive = RProximalBundle(
        manifold=manifold,
        retraction_map = manifold.exp,
        transport_map = manifold.transport,
        objective_function=cost,
        subgradient = subgradient,
        initial_point=initial_point,
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        proximal_parameter=proximal_parameter,
        transport_error = 2,
        retraction_error= 0,
        trust_parameter=0.05,
        adaptive_proximal=False
    )

    # Run optimization scheme
    rBundAlg_non_adaptive.run()
    # Store objective gaps vs iterations
    descent_step_indices_non_adaptive.append(rBundAlg_non_adaptive.indices_of_descent_steps)
    null_step_indices_non_adaptive.append(rBundAlg_non_adaptive.indices_of_null_steps)
    relative_objective_gaps_vs_iterations_non_adaptive.append(rBundAlg_non_adaptive.relative_objective_history)
    objective_gaps_vs_iterations_non_adaptive.append(rBundAlg_non_adaptive.objective_history)
    adaptive_step_indices_non_adaptive.append(rBundAlg_non_adaptive.indices_of_proximal_doubling_steps)
    print('Non-Adaptive Converged.')


# %% Plot results of experiment
# we plot the objective gaps vs iterations for each proximal parameter as a line plot, and specify which steps were descent steps as
# a scatter plot on top of the line plot - this is done as a log plot
plt.figure(figsize=(10, 6))
for i, proximal_parameter in enumerate(proximal_parameters):
    plt.plot(objective_gaps_vs_iterations_adaptive[i], label=f"Adaptive Proximal Parameter {proximal_parameter}")
    # Plot null steps
    for j in range(len(null_step_indices_adaptive[i])):
        plt.scatter(null_step_indices_adaptive[i][j], objective_gaps_vs_iterations_adaptive[i][null_step_indices_adaptive[i][j]], color='blue', s=10)
    # Plot adaptive steps
    for j in range(len(adaptive_step_indices_adaptive[i])):
        plt.scatter(adaptive_step_indices_adaptive[i][j], objective_gaps_vs_iterations_adaptive[i][adaptive_step_indices_adaptive[i][j]], color='green', s=10)
plt.yscale('log')
plt.xlabel("Iterations")
# plt.xlim(0, 7)
plt.ylabel("Objective Gap (log scale)")
plt.title("Objective Gaps vs Iterations for Different Proximal Parameters (Adaptive)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i, proximal_parameter in enumerate(proximal_parameters):
    plt.plot(objective_gaps_vs_iterations_non_adaptive[i], label=f"Non-Adaptive Proximal Parameter {proximal_parameter}")
    # Plot null steps
    for j in range(len(null_step_indices_non_adaptive[i])):
        plt.scatter(null_step_indices_non_adaptive[i][j], objective_gaps_vs_iterations_non_adaptive[i][null_step_indices_non_adaptive[i][j]], color='blue', s=10)
    # Plot adaptive steps
    for j in range(len(adaptive_step_indices_non_adaptive[i])):
        plt.scatter(adaptive_step_indices_non_adaptive[i][j], objective_gaps_vs_iterations_non_adaptive[i][adaptive_step_indices_non_adaptive[i][j]], color='green', s=10)
plt.yscale('log')
plt.xlabel("Iterations")
# plt.xlim(0, 7)
plt.ylabel("Objective Gap (log scale)")
plt.title("Objective Gaps vs Iterations for Different Proximal Parameters (Non-Adaptive)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# we plot the relative objective gaps vs iterations for each proximal parameter as a line plot, and specify which steps were descent steps as
# a scatter plot on top of the line plot - this is done as a log plot
plt.figure(figsize=(10, 6))
for i, proximal_parameter in enumerate(proximal_parameters):
    plt.plot(relative_objective_gaps_vs_iterations_adaptive[i], label=f"Adaptive Proximal Parameter {proximal_parameter}")
    # Plot null steps
    for j in range(len(null_step_indices_adaptive[i])):
        plt.scatter(null_step_indices_adaptive[i][j], relative_objective_gaps_vs_iterations_adaptive[i][null_step_indices_adaptive[i][j]], color='blue', s=10)
    # Plot adaptive steps
    for j in range(len(adaptive_step_indices_adaptive[i])):
        plt.scatter(adaptive_step_indices_adaptive[i][j], relative_objective_gaps_vs_iterations_adaptive[i][adaptive_step_indices_adaptive[i][j]], color='green', s=10)
plt.yscale('log')
plt.xlabel("Iterations")
plt.xlim(0, 7)
plt.ylabel("Relative Objective Gap (log scale)")
plt.title("Relative Objective Gaps vs Iterations for Different Proximal Parameters (Adaptive)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for i, proximal_parameter in enumerate(proximal_parameters):
    plt.plot(relative_objective_gaps_vs_iterations_non_adaptive[i], label=f"Non-Adaptive Proximal Parameter {proximal_parameter}")
    # Plot null steps
    for j in range(len(null_step_indices_non_adaptive[i])):
        plt.scatter(null_step_indices_non_adaptive[i][j], relative_objective_gaps_vs_iterations_non_adaptive[i][null_step_indices_non_adaptive[i][j]], color='blue', s=10)
    # Plot adaptive steps
    for j in range(len(adaptive_step_indices_non_adaptive[i])):
        plt.scatter(adaptive_step_indices_non_adaptive[i][j], relative_objective_gaps_vs_iterations_non_adaptive[i][adaptive_step_indices_non_adaptive[i][j]], color='green', s=10)
plt.yscale('log')
plt.xlabel("Iterations")
# plt.xlim(0, 7)
plt.ylabel("Relative Objective Gap (log scale)")
plt.title("Relative Objective Gaps vs Iterations for Different Proximal Parameters (Non-Adaptive)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# %%
