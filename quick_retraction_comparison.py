"""
Quick Retraction Comparison - Dimension 10
Streamlined version for fast execution and clean log-log plot.
"""

import autograd.numpy as anp
import numpy as np
import matplotlib.pyplot as plt
import time
from pymanopt.manifolds import SymmetricPositiveDefinite
from src.RiemannianProximalBundle import RProximalBundle
import warnings
warnings.filterwarnings('ignore')


def first_order_retraction(manifold, x, xi):
    """First-order retraction with projection."""
    candidate = x + xi
    candidate = 0.5 * (candidate + candidate.T)

    # Simple eigenvalue clipping
    eigenvals, eigenvecs = anp.linalg.eigh(candidate)
    eigenvals_clipped = anp.maximum(eigenvals, 1e-12)
    result = eigenvecs @ anp.diag(eigenvals_clipped) @ eigenvecs.T

    return 0.5 * (result + result.T)


class QuickSubgradientMethod:
    """Quick subgradient method implementation."""

    def __init__(self, manifold, objective_function, subgradient, initial_point,
                 initial_objective, max_iter=30):
        self.manifold = manifold
        self.compute_objective = objective_function
        self.compute_subgradient = subgradient
        self.current_point = initial_point
        self.max_iter = max_iter
        self.objective_history = [initial_objective]

    def run(self):
        for k in range(self.max_iter):
            subgrad = self.compute_subgradient(self.current_point)
            subgrad_norm = self.manifold.norm(self.current_point, subgrad)

            if subgrad_norm > 1e-12:
                step_size = 1.0 / (k + 1)  # Diminishing step size
                direction = -step_size * subgrad / subgrad_norm
                self.current_point = self.manifold.exp(self.current_point, direction)

            obj = self.compute_objective(self.current_point)
            self.objective_history.append(obj)

            if subgrad_norm < 1e-8:
                break


def generate_simple_data(manifold, n_points, dim):
    """Generate simple well-conditioned data."""
    anp.random.seed(42)
    data = []
    for i in range(n_points):
        A = anp.random.randn(dim, dim) * 0.5
        spd_matrix = A @ A.T + anp.eye(dim) * 0.5
        data.append(spd_matrix)
    return data


def setup_simple_problem(data, manifold):
    """Simple median problem setup."""
    def cost_function(point):
        return sum(manifold.dist(point, x) for x in data) / len(data)

    def subgradient_function(point):
        grad = anp.zeros_like(point)
        for x in data:
            log_point = manifold.log(point, x)
            norm_log = manifold.norm(point, log_point)
            if norm_log > 1e-12:
                grad -= log_point / norm_log
        return grad / len(data)

    return cost_function, subgradient_function


def run_quick_comparison():
    """Quick comparison of retraction methods."""

    print("Quick Retraction Comparison - Dimension 10")
    print("="*45)

    # Reduced problem size for speed
    dim = 10
    n_points = 10
    max_iter = 25

    print(f"SPD({dim}) median, {n_points} points, {max_iter} iterations")
    print()

    # Setup
    manifold = SymmetricPositiveDefinite(dim)
    data = generate_simple_data(manifold, n_points, dim)
    cost_function, subgradient_function = setup_simple_problem(data, manifold)

    # Initial point
    initial_point = manifold.random_point()
    initial_objective = cost_function(initial_point)
    initial_subgradient = subgradient_function(initial_point)

    print(f"Initial objective: {initial_objective:.6f}")
    print()

    results = {}

    # 1. RPB with Exponential Maps
    print("1. RPB-Exponential...")
    rpb_exp = RProximalBundle(
        manifold=manifold,
        retraction_map=manifold.exp,
        transport_map=manifold.transport,
        objective_function=cost_function,
        subgradient=subgradient_function,
        initial_point=initial_point,
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        true_min_obj=0,
        adaptive_proximal=True,
        trust_parameter=0.2,
        transport_error=0,
        retraction_error=0,
        max_iter=max_iter
    )
    rpb_exp.run()
    results['RPB-Exponential'] = rpb_exp.raw_objective_history
    print(f"   Final: {rpb_exp.raw_objective_history[-1]:.6f}")

    # 2. RPB with Second-Order Retractions
    print("2. RPB-Second-Order...")
    rpb_second = RProximalBundle(
        manifold=manifold,
        retraction_map=manifold.retraction,
        transport_map=manifold.transport,
        objective_function=cost_function,
        subgradient=subgradient_function,
        initial_point=initial_point,
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        true_min_obj=0,
        adaptive_proximal=True,
        trust_parameter=0.2,
        transport_error=2,
        retraction_error=1,
        max_iter=max_iter
    )
    rpb_second.run()
    results['RPB-Second-Order'] = rpb_second.raw_objective_history
    print(f"   Final: {rpb_second.raw_objective_history[-1]:.6f}")

    # 3. RPB with First-Order Retractions
    print("3. RPB-First-Order...")
    first_order_map = lambda x, xi: first_order_retraction(manifold, x, xi)
    rpb_first = RProximalBundle(
        manifold=manifold,
        retraction_map=first_order_map,
        transport_map=manifold.transport,
        objective_function=cost_function,
        subgradient=subgradient_function,
        initial_point=initial_point,
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        true_min_obj=0,
        adaptive_proximal=True,
        trust_parameter=0.2,
        transport_error=2,
        retraction_error=2,
        max_iter=max_iter
    )
    rpb_first.run()
    results['RPB-First-Order'] = rpb_first.raw_objective_history
    print(f"   Final: {rpb_first.raw_objective_history[-1]:.6f}")

    # 4. Subgradient Method
    print("4. SGM...")
    sgm = QuickSubgradientMethod(
        manifold=manifold,
        objective_function=cost_function,
        subgradient=subgradient_function,
        initial_point=initial_point,
        initial_objective=initial_objective,
        max_iter=max_iter
    )
    sgm.run()
    results['SGM'] = sgm.objective_history
    print(f"   Final: {sgm.objective_history[-1]:.6f}")

    return results, initial_objective


def create_clean_loglog_plot(results, initial_objective):
    """Create the requested log-log objective gap plot."""

    # Find best objective across all methods
    best_objective = min(min(history) for history in results.values())

    # Create figure
    plt.figure(figsize=(10, 7))

    # Colors and styles
    styles = {
        'RPB-Exponential': {'color': 'blue', 'linestyle': '-', 'linewidth': 3, 'alpha': 0.8},
        'RPB-Second-Order': {'color': 'red', 'linestyle': '--', 'linewidth': 2.5, 'alpha': 0.8},
        'RPB-First-Order': {'color': 'green', 'linestyle': '-.', 'linewidth': 2.5, 'alpha': 0.8},
        'SGM': {'color': 'orange', 'linestyle': ':', 'linewidth': 2.5, 'alpha': 0.8}
    }

    for method_name, obj_history in results.items():
        # Compute gaps
        gaps = [max(obj - best_objective, 1e-16) for obj in obj_history]

        # Iterations (start from 1 for log scale)
        iterations = list(range(1, len(gaps) + 1))

        # Plot
        style = styles[method_name]
        plt.plot(iterations, gaps,
                color=style['color'],
                linestyle=style['linestyle'],
                linewidth=style['linewidth'],
                alpha=style['alpha'],
                label=method_name)

    # Formatting
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Objective Gap', fontsize=14)
    plt.title('Log-Log Objective Gap - Dimension 10 SPD Riemannian Median', fontsize=15)

    # Log-log scale
    plt.xscale('log')
    plt.yscale('log')

    # Grid and legend
    plt.grid(True, alpha=0.4, which="both")
    plt.legend(fontsize=12)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig('loglog_objective_gap_dim10.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    print(f"\nBest objective achieved: {best_objective:.8f}")
    print(f"Initial objective: {initial_objective:.6f}")
    print(f"Total improvement: {initial_objective - best_objective:.6f}")

    print("\nFinal gaps from best:")
    for method, history in results.items():
        final_gap = history[-1] - best_objective
        print(f"  {method:<20s}: {final_gap:.2e}")


if __name__ == "__main__":
    print("Running quick retraction comparison...")

    results, initial_obj = run_quick_comparison()
    create_clean_loglog_plot(results, initial_obj)

    print("\n✓ Comparison completed!")
    print("✓ Log-log plot saved as 'loglog_objective_gap_dim10.png'")