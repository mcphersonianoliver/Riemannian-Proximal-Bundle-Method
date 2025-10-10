"""
Corrected Python test with proper Phase 1 median finding using pymanopt approach
"""

import numpy as np
import matplotlib.pyplot as plt
from pymanopt.manifolds import SymmetricPositiveDefinite
import sys
sys.path.append('src')
from RiemannianProximalBundle import RProximalBundle

def cost_function(point, data, manifold):
    """Riemannian median cost function"""
    return sum([manifold.dist(point, x) for x in data]) / len(data)

def subgradient_function(point, data, manifold):
    """Riemannian median subgradient"""
    import autograd.numpy as anp
    grad = anp.zeros_like(point)
    for x in data:
        log_point = manifold.log(point, x)
        norm_log = manifold.norm(point, log_point)
        if norm_log > 1e-12:
            grad += -(log_point) / norm_log
    return grad / len(data)

def robust_gradient_descent(cost, subgradient, start_point, manifold, max_iter=500):
    """Robust gradient descent using pymanopt"""
    X = start_point.copy()

    for iter_num in range(max_iter):
        grad = subgradient(X)
        grad_norm = manifold.norm(X, grad)

        if grad_norm < 1e-10:
            break

        # Adaptive step size with backtracking
        step_size = 0.1 / (1 + iter_num * 0.01)
        current_obj = cost(X)

        for alpha in [step_size, step_size/2, step_size/4]:
            try:
                X_new = manifold.exp(X, -alpha * grad)
                new_obj = cost(X_new)
                if new_obj < current_obj:
                    X = X_new
                    break
            except:
                continue

    return X

def phase1_find_median(data, manifold):
    """Phase 1: Find the true Riemannian median"""
    print("=== PHASE 1: Finding True Median ===")

    def cost(point):
        return cost_function(point, data, manifold)

    def subgradient(point):
        return subgradient_function(point, data, manifold)

    candidates = []

    # Try starting from each data point
    for i, start_point in enumerate(data):
        result = robust_gradient_descent(cost, subgradient, start_point, manifold)
        obj_val = cost(result)
        candidates.append((obj_val, result, f"data_point_{i}"))
        print(f"  From data point {i}: objective = {obj_val:.10f}")

    # Try starting from random points
    for i in range(3):
        start_point = manifold.random_point()
        result = robust_gradient_descent(cost, subgradient, start_point, manifold)
        obj_val = cost(result)
        candidates.append((obj_val, result, f"random_{i}"))
        print(f"  From random point {i}: objective = {obj_val:.10f}")

    # Find the best
    candidates.sort(key=lambda x: x[0])
    best_obj, best_point, best_method = candidates[0]

    print(f"\\nBest result: objective = {best_obj:.10f} (from {best_method})")

    # Check gradient norm
    grad = subgradient(best_point)
    grad_norm = manifold.norm(best_point, grad)
    print(f"Gradient norm at best point: {grad_norm:.10f}")

    return best_point, best_obj

def generate_test_data_matching_julia():
    """Generate test data using exact same data points as Julia version"""
    # Exact base point from Julia run
    base_point = np.array([
        [1.5553775484093073, 0.0881444897500597],
        [0.0881444897500597, 1.5243065153294948]
    ])

    # Exact data points from Julia run
    data = [
        np.array([
            [2.2846743337457487, -1.0747056674130149],
            [-1.0747056674130151, 1.7783705028792500]
        ]),
        np.array([
            [1.3588126157670670, 0.9825774429878000],
            [0.9825774429878000, 2.7617298741925764]
        ]),
        np.array([
            [2.6638202248799034, 1.1000816268898084],
            [1.1000816268898084, 1.9854209837702750]
        ]),
        np.array([
            [4.6453882099991661, 1.7991740173838338],
            [1.7991740173838338, 2.8041800201508371]
        ]),
        np.array([
            [0.9831861042132830, 0.4125300954631763],
            [0.4125300954631763, 3.4743182284258554]
        ])
    ]

    return data, base_point

def get_exact_initial_point():
    """Get exact initial point from Julia version"""
    return np.array([
        [1.5223718755744191, -0.0086383303086675],
        [-0.0086383303086675, 1.5856486774143124]
    ])

def plot_convergence(rpb):
    """Plot convergence with proper positive gaps"""
    objective_gaps = rpb.objective_history
    iterations = np.arange(len(objective_gaps))

    # Check if all gaps are positive
    use_log_scale = all(gap > 0 for gap in objective_gaps)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Main convergence plot
    ax1.plot(iterations, objective_gaps, 'b-', linewidth=2, label='Objective Gap')
    if use_log_scale:
        ax1.set_yscale('log')
        ax1.set_ylabel('Objective Gap (log scale)')
    else:
        ax1.set_ylabel('Objective Gap')
    ax1.set_xlabel('Iteration Number')
    ax1.set_title('Convergence of Riemannian Proximal Bundle Method')
    ax1.grid(True, alpha=0.3)

    # Add step type markers
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

    # Print summary
    print("\\nConvergence Summary:")
    print("=" * 50)
    print(f"Final objective gap: {objective_gaps[-1]}")
    print(f"Total iterations: {len(objective_gaps) - 1}")
    print(f"Descent steps: {len(rpb.indices_of_descent_steps)}")
    print(f"Null steps: {len(rpb.indices_of_null_steps)}")
    print(f"Proximal doubling steps: {len(rpb.indices_of_proximal_doubling_steps)}")
    print(f"Final proximal parameter: {rpb.proximal_parameter_history[-1]}")
    print("=" * 50)

    return fig

def test_corrected_python():
    """Test corrected Python implementation with proper Phase 1"""
    print("=" * 60)
    print("Testing Corrected Python Implementation")
    print("=" * 60)

    # Problem setup
    dim = 2
    num_data_points = 5
    manifold = SymmetricPositiveDefinite(dim)

    print("Problem Setup:")
    print(f"  Matrix dimension: {dim}×{dim}")
    print(f"  Number of data points: {num_data_points}")

    # Generate data using exact same points as Julia
    data, base_point = generate_test_data_matching_julia()
    print(f"  Base point:\\n{base_point}")

    # Phase 1: Find true median
    true_median, true_min_obj = phase1_find_median(data, manifold)

    # Setup initial point using exact same point as Julia
    initial_point = get_exact_initial_point()

    def cost(point):
        return cost_function(point, data, manifold)

    def subgradient(point):
        return subgradient_function(point, data, manifold)

    initial_objective = cost(initial_point)
    initial_subgradient = subgradient(initial_point)

    print(f"\\nInitial Setup:")
    print(f"  Initial objective: {initial_objective}")
    print(f"  True minimum objective: {true_min_obj}")
    print(f"  Initial gap: {initial_objective - true_min_obj}")

    # Run proximal bundle algorithm
    print("\\n" + "=" * 40)
    print("Running Proximal Bundle Algorithm")
    print("=" * 40)

    rpb = RProximalBundle(
        manifold=manifold,
        retraction_map=manifold.exp,
        transport_map=manifold.transport,
        objective_function=cost,
        subgradient=subgradient,
        initial_point=initial_point,
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        true_min_obj=true_min_obj,
        proximal_parameter=0.1,  # Same as Julia corrected version
        trust_parameter=0.2,     # Same as Julia corrected version
        max_iter=50,
        tolerance=1e-8,
        adaptive_proximal=True,  # Same as Julia corrected version
        know_minimizer=True
    )

    rpb.run()

    print("\\nAlgorithm Results:")
    print(f"  Final objective: {rpb.raw_objective_history[-1]}")
    print(f"  Final gap: {rpb.objective_history[-1]}")
    print(f"  Number of iterations: {len(rpb.objective_history) - 1}")
    print(f"  Descent steps: {len(rpb.indices_of_descent_steps)}")
    print(f"  Null steps: {len(rpb.indices_of_null_steps)}")
    print(f"  Proximal doubling steps: {len(rpb.indices_of_proximal_doubling_steps)}")

    # Check convergence
    if rpb.objective_history[-1] < 1e-6:
        print("  ✅ Algorithm converged successfully!")
    else:
        print("  ⚠️  Algorithm may not have fully converged")

    return rpb, data, true_median

if __name__ == "__main__":
    rpb, data, true_median = test_corrected_python()

    print("\\n" + "=" * 60)
    print("Creating convergence plots...")
    print("=" * 60)

    # Generate and save the convergence plot
    convergence_plot = plot_convergence(rpb)
    plt.savefig("corrected_python_rpb_convergence.png", dpi=150, bbox_inches='tight')
    print("Plot saved as 'corrected_python_rpb_convergence.png'")
    plt.close()

    print("\\n" + "=" * 60)
    print("Test completed! Corrected Python implementation with proper Phase 1.")
    print("=" * 60)