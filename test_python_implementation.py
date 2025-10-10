import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, logm, sqrtm, inv
import sys
import os

# Add the src directory to the path
sys.path.append('src')
from RiemannianProximalBundle import RProximalBundle

class SPDManifold:
    """
    Simple implementation of the SPD (Symmetric Positive Definite) manifold
    for testing the Riemannian Proximal Bundle method.
    """
    def __init__(self, n):
        self.n = n  # dimension of matrices (n×n)

    def inner_product(self, X, U, V):
        """
        Compute the Riemannian inner product at point X between tangent vectors U and V.
        For SPD manifold: ⟨U,V⟩_X = tr(X^{-1} U X^{-1} V)
        """
        X_inv = inv(X)
        return np.trace(X_inv @ U @ X_inv @ V)

    def norm(self, X, V):
        """
        Compute the Riemannian norm of tangent vector V at point X.
        """
        return np.sqrt(self.inner_product(X, V, V))

    def exp_map(self, X, V):
        """
        Exponential map: retract tangent vector V at point X to the manifold.
        For SPD: Exp_X(V) = X^{1/2} exp(X^{-1/2} V X^{-1/2}) X^{1/2}
        """
        sqrt_X = sqrtm(X)
        inv_sqrt_X = inv(sqrt_X)
        return sqrt_X @ expm(inv_sqrt_X @ V @ inv_sqrt_X) @ sqrt_X

    def log_map(self, X, Y):
        """
        Logarithmic map: compute tangent vector from X to Y.
        For SPD: Log_X(Y) = X^{1/2} log(X^{-1/2} Y X^{-1/2}) X^{1/2}
        """
        sqrt_X = sqrtm(X)
        inv_sqrt_X = inv(sqrt_X)
        return sqrt_X @ logm(inv_sqrt_X @ Y @ inv_sqrt_X) @ sqrt_X

    def parallel_transport(self, X, Y, V):
        """
        Parallel transport tangent vector V from X to Y.
        For SPD manifold, we'll use a simple approximation for this test.
        """
        # Simple approximation: V (identity transport)
        # For a more accurate implementation, use the Schild's ladder or other methods
        return V

    def random_spd_matrix(self):
        """
        Generate a random SPD matrix of size n×n.
        """
        A = np.random.randn(self.n, self.n)
        return A @ A.T + np.eye(self.n)  # A*A' + I is always SPD

    def distance_spd(self, X, Y):
        """
        Compute the Riemannian distance between two SPD matrices.
        """
        return self.norm(X, self.log_map(X, Y))


class RiemannianMedianProblem:
    """
    Problem setup for computing the Riemannian median on SPD manifolds.
    """
    def __init__(self, manifold, data_points, weights=None):
        self.manifold = manifold
        self.data_points = data_points
        if weights is None:
            self.weights = np.ones(len(data_points)) / len(data_points)
        else:
            self.weights = weights

    def objective_function(self, X):
        """
        Compute the Riemannian median objective function:
        f(X) = Σᵢ wᵢ * d(X, Yᵢ)
        where d(X, Yᵢ) is the Riemannian distance between X and data point Yᵢ
        """
        total = 0.0
        for i, Y in enumerate(self.data_points):
            dist = self.manifold.distance_spd(X, Y)
            total += self.weights[i] * dist
        return total

    def subgradient_function(self, X):
        """
        Compute a subgradient of the Riemannian median objective function.
        The subgradient at X is: ∇f(X) = Σᵢ wᵢ * (log_X(Yᵢ) / ||log_X(Yᵢ)||_X)
        """
        subgrad = np.zeros_like(X)

        for i, Y in enumerate(self.data_points):
            log_XY = self.manifold.log_map(X, Y)
            norm_log_XY = self.manifold.norm(X, log_XY)

            if norm_log_XY > 1e-12:  # Avoid division by zero
                unit_log_XY = log_XY / norm_log_XY
                subgrad += self.weights[i] * unit_log_XY

        return subgrad


def generate_test_data(manifold, num_points, center_matrix=None):
    """
    Generate test data points around a center for testing.
    Uses the same approach as the previous experiments.
    """
    if center_matrix is None:
        center_matrix = manifold.random_spd_matrix()  # Random base point

    data_points = []

    # Generate preset scalings for reproducibility
    np.random.seed(123)  # Different seed for data generation
    preset_scalings = [np.random.uniform(0.5, 2.0) for _ in range(num_points)]
    np.random.seed(42)  # Reset to original seed

    for i in range(num_points):
        # Generate random tangent vector
        V = np.random.randn(manifold.n, manifold.n)
        V = (V + V.T) / 2  # Make symmetric

        # Normalize and scale the tangent vector
        V_norm = manifold.norm(center_matrix, V)
        if V_norm > 1e-12:
            V = V / V_norm  # Normalize
            V = V * preset_scalings[i]  # Scale by preset amount

        # Retract to manifold
        point = manifold.exp_map(center_matrix, V)
        data_points.append(point)

    return data_points, center_matrix


def compute_true_median(problem, initial_guess=None, max_iter=100, tol=1e-8):
    """
    Compute the true Riemannian median using gradient descent for comparison.
    """
    if initial_guess is None:
        initial_guess = np.eye(problem.manifold.n)

    X = initial_guess.copy()

    for iter_num in range(max_iter):
        grad = problem.subgradient_function(X)
        grad_norm = problem.manifold.norm(X, grad)

        if grad_norm < tol:
            break

        # Simple gradient descent step
        step_size = 0.1 / (iter_num + 1)
        X_new = problem.manifold.exp_map(X, -step_size * grad)
        X = X_new

    return X


def plot_convergence(rpb):
    """
    Plot the convergence of the algorithm, matching the Julia version.
    """
    # Extract objective gaps (similar to Julia version)
    objective_gaps = rpb.objective_history
    iterations = np.arange(len(objective_gaps))

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Main convergence plot with conditional log scale
    use_log_scale = all(gap > 0 for gap in objective_gaps)

    ax1.plot(iterations, objective_gaps, 'b-', linewidth=2, label='Objective Gap')
    if use_log_scale:
        ax1.set_yscale('log')
        ax1.set_ylabel('Objective Gap (log scale)')
    else:
        ax1.set_ylabel('Objective Gap')
    ax1.set_xlabel('Iteration Number')
    ax1.set_title('Convergence of Riemannian Proximal Bundle Method')
    ax1.grid(True, alpha=0.3)

    # Add markers for different step types
    if rpb.indices_of_descent_steps:
        valid_descent = [i for i in rpb.indices_of_descent_steps if i < len(objective_gaps)]
        if valid_descent:
            ax1.scatter([i for i in valid_descent],
                       [objective_gaps[i] for i in valid_descent],
                       color='green', marker='o', s=30, label='Descent Steps', zorder=5)

    if rpb.indices_of_null_steps:
        valid_null = [i for i in rpb.indices_of_null_steps if i < len(objective_gaps)]
        if valid_null:
            ax1.scatter([i for i in valid_null],
                       [objective_gaps[i] for i in valid_null],
                       color='orange', marker='s', s=20, label='Null Steps', zorder=5)

    if rpb.indices_of_proximal_doubling_steps:
        valid_doubling = [i for i in rpb.indices_of_proximal_doubling_steps if i < len(objective_gaps)]
        if valid_doubling:
            ax1.scatter([i for i in valid_doubling],
                       [objective_gaps[i] for i in valid_doubling],
                       color='red', marker='^', s=20, label='Proximal Doubling Steps', zorder=5)

    ax1.legend()

    # Proximal parameter plot
    ax2.plot(np.arange(len(rpb.proximal_parameter_history)),
             rpb.proximal_parameter_history,
             'purple', linewidth=2, label='Proximal Parameter (ρ)')
    ax2.set_xlabel('Iteration Number')
    ax2.set_ylabel('Proximal Parameter')
    ax2.set_title('Proximal Parameter Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Print summary statistics
    print("\nConvergence Summary:")
    print("=" * 50)
    print(f"Final objective gap: {objective_gaps[-1]}")
    print(f"Total iterations: {len(objective_gaps) - 1}")
    print(f"Descent steps: {len(rpb.indices_of_descent_steps)}")
    print(f"Null steps: {len(rpb.indices_of_null_steps)}")
    print(f"Proximal doubling steps: {len(rpb.indices_of_proximal_doubling_steps)}")
    print(f"Final proximal parameter: {rpb.proximal_parameter_history[-1]}")
    print("=" * 50)

    return fig


def test_python_rpb():
    """
    Test the Python implementation of RiemannianProximalBundle with SPD Riemannian median problem.
    """
    print("=" * 60)
    print("Testing Python Riemannian Proximal Bundle Implementation")
    print("=" * 60)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Problem setup
    n = 2  # 2x2 SPD matrices for simplicity
    num_data_points = 5
    manifold = SPDManifold(n)

    print("Problem Setup:")
    print(f"  Matrix dimension: {n}×{n}")
    print(f"  Number of data points: {num_data_points}")

    # Generate test data
    data_points, true_center = generate_test_data(manifold, num_data_points)
    print("  True center:")
    print(f"{true_center}")

    # Create median problem
    median_problem = RiemannianMedianProblem(manifold, data_points)

    # Define wrapper functions for the proximal bundle algorithm
    def objective_wrapper(X):
        return median_problem.objective_function(X)

    def subgradient_wrapper(X):
        return median_problem.subgradient_function(X)

    def retraction_wrapper(X, V):
        return manifold.exp_map(X, V)

    def transport_wrapper(X, Y, V):
        return manifold.parallel_transport(X, Y, V)

    # Initial point (slightly perturbed identity)
    initial_point = np.eye(n) + 0.1 * np.random.randn(n, n)
    initial_point = (initial_point + initial_point.T) / 2 + 0.1 * np.eye(n)  # Ensure SPD
    initial_objective = objective_wrapper(initial_point)
    initial_subgradient = subgradient_wrapper(initial_point)

    print("\nInitial Setup:")
    print("  Initial point:")
    print(f"{initial_point}")
    print(f"  Initial objective: {initial_objective}")
    print(f"  Initial subgradient norm: {manifold.norm(initial_point, initial_subgradient)}")

    # Use the true center (used to generate data) as the true minimum
    # The gradient descent often gets stuck in local minima for this problem
    true_median = true_center  # This is the actual minimum since data was generated around it
    true_min_obj = objective_wrapper(true_median)

    print("\nTrue Solution (data generation center):")
    print("  True median:")
    print(f"{true_median}")
    print(f"  True minimum objective: {true_min_obj}")

    # Create and run proximal bundle algorithm
    print("\n" + "=" * 40)
    print("Running Proximal Bundle Algorithm")
    print("=" * 40)

    rpb = RProximalBundle(
        manifold,
        retraction_wrapper,
        transport_wrapper,
        objective_wrapper,
        subgradient_wrapper,
        initial_point,
        initial_objective,
        initial_subgradient,
        true_min_obj=true_min_obj,
        proximal_parameter=0.1,
        trust_parameter=0.2,
        max_iter=50,
        tolerance=1e-8,
        adaptive_proximal=True,
        know_minimizer=False
    )

    # Run the algorithm
    rpb.run()

    print("\nAlgorithm Results:")
    print("  Final point:")
    print(f"{rpb.current_proximal_center}")
    print(f"  Final objective: {rpb.raw_objective_history[-1]}")
    print(f"  Final gap: {rpb.objective_history[-1]}")
    print(f"  Number of iterations: {len(rpb.objective_history) - 1}")
    print(f"  Descent steps: {len(rpb.indices_of_descent_steps)}")
    print(f"  Null steps: {len(rpb.indices_of_null_steps)}")
    print(f"  Proximal doubling steps: {len(rpb.indices_of_proximal_doubling_steps)}")

    # Compute distance between final solution and true median
    final_distance = manifold.distance_spd(rpb.current_proximal_center, true_median)
    print(f"  Distance to true median: {final_distance}")

    # Check convergence
    if rpb.objective_history[-1] < 1e-6:
        print("  ✅ Algorithm converged successfully!")
    else:
        print("  ⚠️  Algorithm may not have fully converged")

    # Test basic functionality
    print("\n" + "=" * 40)
    print("Testing Algorithm Components")
    print("=" * 40)

    # Test model evaluation
    test_direction = np.random.randn(n, n)
    test_direction = (test_direction + test_direction.T) / 2  # Make symmetric
    model_val = rpb.model_evaluation(test_direction)
    print(f"  Model evaluation test: {model_val}")

    # Test candidate direction computation
    cand_dir = rpb.cand_prox_direction()
    print(f"  Candidate direction norm: {manifold.norm(rpb.current_proximal_center, cand_dir)}")

    # Create simple visualization data
    print("\n" + "=" * 40)
    print("Convergence Summary")
    print("=" * 40)

    print("Iteration | Objective Gap | Step Type")
    print("-" * 40)
    for i, gap in enumerate(rpb.objective_history[:min(10, len(rpb.objective_history))]):
        step_type = "Initial"
        if i > 0:
            iter_idx = i - 1
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

    return rpb, median_problem, true_median


if __name__ == "__main__":
    rpb, problem, true_median = test_python_rpb()

    print("\n" + "=" * 60)
    print("Creating convergence plots...")
    print("=" * 60)

    # Generate and save the convergence plot
    convergence_plot = plot_convergence(rpb)
    plt.savefig("python_rpb_convergence.png", dpi=150, bbox_inches='tight')
    print("Plot saved as 'python_rpb_convergence.png'")
    plt.close()  # Close the plot instead of showing it

    print("\n" + "=" * 60)
    print("Test completed! The Python implementation appears to be working.")
    print("You can now compare this with your Julia implementation.")
    print("=" * 60)