"""
Python test script to validate the Riemannian median computation
and provide comparison data for the Julia implementation.
"""

import numpy as np
import sys
import os
from scipy.linalg import expm, logm
sys.path.append('./src')

from RiemannianProximalBundle import RProximalBundle

class SPDManifold:
    """Simple SPD manifold for testing"""
    def __init__(self, n):
        self.n = n

    def inner_product(self, X, U, V):
        """Riemannian inner product"""
        X_inv = np.linalg.inv(X)
        return np.trace(X_inv @ U @ X_inv @ V)

    def norm(self, X, V):
        """Riemannian norm"""
        return np.sqrt(self.inner_product(X, V, V))

def exp_map(X, V):
    """Exponential map for SPD manifold"""
    sqrt_X = np.linalg.cholesky(X)
    inv_sqrt_X = np.linalg.inv(sqrt_X)
    return sqrt_X @ expm(inv_sqrt_X @ V @ inv_sqrt_X.T) @ sqrt_X.T

def log_map(X, Y):
    """Logarithmic map for SPD manifold"""
    sqrt_X = np.linalg.cholesky(X)
    inv_sqrt_X = np.linalg.inv(sqrt_X)
    return sqrt_X @ logm(inv_sqrt_X @ Y @ inv_sqrt_X.T) @ sqrt_X.T

def parallel_transport(X, Y, V):
    """Simple parallel transport (identity for testing)"""
    return V

def distance_spd(manifold, X, Y):
    """Riemannian distance between SPD matrices"""
    return manifold.norm(X, log_map(X, Y))

class RiemannianMedianProblem:
    """Riemannian median problem on SPD manifold"""
    def __init__(self, manifold, data_points, weights=None):
        self.manifold = manifold
        self.data_points = data_points
        if weights is None:
            weights = np.ones(len(data_points)) / len(data_points)
        self.weights = weights

    def objective_function(self, X):
        """Compute median objective function"""
        total = 0.0
        for i, Y in enumerate(self.data_points):
            dist = distance_spd(self.manifold, X, Y)
            total += self.weights[i] * dist
        return total

    def subgradient_function(self, X):
        """Compute subgradient of median objective"""
        subgrad = np.zeros_like(X)

        for i, Y in enumerate(self.data_points):
            log_XY = log_map(X, Y)
            norm_log_XY = self.manifold.norm(X, log_XY)

            if norm_log_XY > 1e-12:
                unit_log_XY = log_XY / norm_log_XY
                subgrad += self.weights[i] * unit_log_XY

        return subgrad

def generate_test_data(manifold, num_points, center_matrix=None):
    """Generate test data around a center"""
    if center_matrix is None:
        center_matrix = 2.0 * np.eye(manifold.n)

    data_points = []
    np.random.seed(42)  # For reproducibility

    for i in range(num_points):
        # Generate random symmetric tangent vector
        V = np.random.randn(manifold.n, manifold.n)
        V = (V + V.T) / 2
        V = V * 0.5  # Scale down

        # Retract to manifold
        point = exp_map(center_matrix, V)
        data_points.append(point)

    return data_points, center_matrix

def test_python_implementation():
    print("=" * 60)
    print("Testing Python Riemannian Proximal Bundle Implementation")
    print("=" * 60)

    # Problem setup
    n = 2
    num_data_points = 5
    manifold = SPDManifold(n)

    print(f"Problem Setup:")
    print(f"  Matrix dimension: {n}×{n}")
    print(f"  Number of data points: {num_data_points}")

    # Generate test data
    data_points, true_center = generate_test_data(manifold, num_data_points)
    print(f"  True center:\n{true_center}")

    # Create median problem
    median_problem = RiemannianMedianProblem(manifold, data_points)

    # Initial point
    initial_point = np.eye(n) + 0.1 * np.random.randn(n, n)
    initial_point = (initial_point + initial_point.T) / 2 + 0.1 * np.eye(n)
    initial_objective = median_problem.objective_function(initial_point)
    initial_subgradient = median_problem.subgradient_function(initial_point)

    print(f"\nInitial Setup:")
    print(f"  Initial point:\n{initial_point}")
    print(f"  Initial objective: {initial_objective}")
    print(f"  Initial subgradient norm: {manifold.norm(initial_point, initial_subgradient)}")

    # Simple gradient descent to find true minimum
    X = initial_point.copy()
    for _ in range(100):
        grad = median_problem.subgradient_function(X)
        grad_norm = manifold.norm(X, grad)
        if grad_norm < 1e-8:
            break
        step_size = 0.1 / (_ + 1)
        X = exp_map(X, -step_size * grad)

    true_min_obj = median_problem.objective_function(X)
    print(f"\nTrue minimum objective (gradient descent): {true_min_obj}")

    # Create proximal bundle algorithm
    rpb = RProximalBundle(
        manifold=manifold,
        retraction_map=exp_map,
        transport_map=parallel_transport,
        objective_function=median_problem.objective_function,
        subgradient=median_problem.subgradient_function,
        initial_point=initial_point,
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        true_min_obj=true_min_obj,
        proximal_parameter=0.1,
        trust_parameter=0.2,
        max_iter=50,
        tolerance=1e-8,
        adaptive_proximal=True,
        know_minimizer=True
    )

    print("\n" + "=" * 40)
    print("Running Proximal Bundle Algorithm")
    print("=" * 40)

    # Run algorithm
    rpb.run()

    print(f"\nAlgorithm Results:")
    print(f"  Final point:\n{rpb.current_proximal_center}")
    print(f"  Final objective: {rpb.raw_objective_history[-1]}")
    print(f"  Final gap: {rpb.objective_history[-1]}")
    print(f"  Number of iterations: {len(rpb.objective_history) - 1}")
    print(f"  Descent steps: {len(rpb.indices_of_descent_steps)}")
    print(f"  Null steps: {len(rpb.indices_of_null_steps)}")
    print(f"  Proximal doubling steps: {len(rpb.indices_of_proximal_doubling_steps)}")

    # Convergence check
    if rpb.objective_history[-1] < 1e-6:
        print("  ✅ Algorithm converged successfully!")
    else:
        print("  ⚠️  Algorithm may not have fully converged")

    print("\n" + "=" * 40)
    print("Convergence Summary")
    print("=" * 40)

    print("Iteration | Objective Gap | Step Type")
    print("-" * 40)
    for i, gap in enumerate(rpb.objective_history[:min(10, len(rpb.objective_history))]):
        step_type = "Initial"
        if i > 0:
            iter_idx = i
            if iter_idx in rpb.indices_of_descent_steps:
                step_type = "Descent"
            elif iter_idx in rpb.indices_of_null_steps:
                step_type = "Null"
            elif iter_idx in rpb.indices_of_proximal_doubling_steps:
                step_type = "Doubling"
        print(f"{i:8d} | {gap:12.6e} | {step_type}")

    if len(rpb.objective_history) > 10:
        print("...")
        final_i = len(rpb.objective_history) - 1
        final_gap = rpb.objective_history[-1]
        print(f"{final_i:8d} | {final_gap:12.6e} | Final")

    # Save test data for Julia comparison
    test_data = {
        'initial_point': initial_point,
        'initial_objective': initial_objective,
        'true_min_obj': true_min_obj,
        'data_points': data_points,
        'final_objective': rpb.raw_objective_history[-1],
        'final_gap': rpb.objective_history[-1],
        'iterations': len(rpb.objective_history) - 1,
        'descent_steps': len(rpb.indices_of_descent_steps),
        'null_steps': len(rpb.indices_of_null_steps),
        'doubling_steps': len(rpb.indices_of_proximal_doubling_steps)
    }

    np.savez('python_test_results.npz', **test_data)
    print(f"\n📁 Test results saved to 'python_test_results.npz'")

    return rpb, median_problem

if __name__ == "__main__":
    rpb, problem = test_python_implementation()
    print("\n" + "=" * 60)
    print("Python test completed! You can now compare with Julia implementation.")
    print("=" * 60)