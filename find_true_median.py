import numpy as np
import sys
sys.path.append('src')
from test_python_implementation import *
from scipy.optimize import minimize

# Let's find the actual Riemannian median more carefully
np.random.seed(42)

# Problem setup
n = 2
num_data_points = 5
manifold = SPDManifold(n)

# Generate test data
data_points, true_center = generate_test_data(manifold, num_data_points)
median_problem = RiemannianMedianProblem(manifold, data_points)

print("Data points:")
for i, point in enumerate(data_points):
    obj_at_point = median_problem.objective_function(point)
    print(f"Point {i}: obj = {obj_at_point:.6f}")

# Try gradient descent with much smaller step sizes and more careful implementation
def careful_gradient_descent(problem, start_point, max_iter=2000, base_step_size=0.01):
    X = start_point.copy()

    for iter_num in range(max_iter):
        grad = problem.subgradient_function(X)
        grad_norm = manifold.norm(X, grad)

        if grad_norm < 1e-10:
            print(f"Converged at iteration {iter_num}")
            break

        # Adaptive step size with backtracking
        step_size = base_step_size / (1 + iter_num * 0.001)

        # Try the step
        X_new = manifold.exp_map(X, -step_size * grad)

        # Make sure we're making progress
        current_obj = problem.objective_function(X)
        new_obj = problem.objective_function(X_new)

        # If we're not improving, reduce step size
        backtrack_count = 0
        while new_obj > current_obj and backtrack_count < 10:
            step_size *= 0.5
            X_new = manifold.exp_map(X, -step_size * grad)
            new_obj = problem.objective_function(X_new)
            backtrack_count += 1

        X = X_new

        if iter_num % 200 == 0:
            print(f"Iter {iter_num}: obj = {new_obj:.8f}, grad_norm = {grad_norm:.8f}")

    return X

print(f"\n--- Careful gradient descent from data point 2 (best so far) ---")
best_start = data_points[2]  # This had the lowest objective
result = careful_gradient_descent(median_problem, best_start)
obj_at_result = median_problem.objective_function(result)
grad_at_result = median_problem.subgradient_function(result)
grad_norm_at_result = manifold.norm(result, grad_at_result)

print(f"\nFinal result:")
print(f"Point: \n{result}")
print(f"Objective: {obj_at_result:.10f}")
print(f"Gradient norm: {grad_norm_at_result:.10f}")

# Let's also try starting from the geometric mean of the data points
print(f"\n--- Starting from geometric mean ---")
# For SPD matrices, geometric mean is (∏ᵢ Xᵢ)^(1/n)
# We'll approximate this by averaging in the log space
log_sum = np.zeros((n, n))
for point in data_points:
    log_sum += manifold.log_map(np.eye(n), point)
geometric_mean_log = log_sum / len(data_points)
geometric_mean = manifold.exp_map(np.eye(n), geometric_mean_log)

print(f"Geometric mean: \n{geometric_mean}")
obj_at_geom_mean = median_problem.objective_function(geometric_mean)
print(f"Objective at geometric mean: {obj_at_geom_mean:.10f}")

result_from_geom = careful_gradient_descent(median_problem, geometric_mean)
obj_at_result_geom = median_problem.objective_function(result_from_geom)
grad_norm_result_geom = manifold.norm(result_from_geom, median_problem.subgradient_function(result_from_geom))

print(f"Result from geometric mean:")
print(f"Objective: {obj_at_result_geom:.10f}")
print(f"Gradient norm: {grad_norm_result_geom:.10f}")

# Compare all results
print(f"\n=== COMPARISON ===")
print(f"Best data point (point 2): {median_problem.objective_function(data_points[2]):.10f}")
print(f"GD from best data point: {obj_at_result:.10f}")
print(f"GD from geometric mean: {obj_at_result_geom:.10f}")
print(f"Original 'true center': {median_problem.objective_function(true_center):.10f}")