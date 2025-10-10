# %%
# RMedian Experiment: Exponential Maps vs Retractions on SPD Manifolds
import autograd.numpy as anp
import matplotlib.pyplot as plt
from pymanopt.manifolds import SymmetricPositiveDefinite
from src.RiemannianProximalBundle import RProximalBundle

# ============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE VALUES AS NEEDED
# ============================================================================

# Dimensions to test
DIMENSIONS_TO_TEST = [10, 25, 50, 100]

# Problem parameters
NUM_DATA_POINTS = 200
PHASE1_ITERATIONS = 300  # For estimating minimum
PHASE2_ITERATIONS = 200  # For convergence analysis
TRUST_PARAMETER = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42

print(f"Configured to test dimensions: {DIMENSIONS_TO_TEST}")
print(f"Data points per dimension: {NUM_DATA_POINTS}")
print(f"Random seed: {RANDOM_SEED}")

# ============================================================================

# Setting up functions
def first_order_retraction(manifold, x, xi):  # pylint: disable=unused-argument
    """
    First-order retraction for SPD manifolds: R_X(ξ) = X + ξ
    followed by projection to ensure the result is positive definite.

    Args:
        manifold: The SPD manifold (unused but kept for API consistency)
        x: Base point (symmetric positive definite matrix)
        xi: Tangent vector (symmetric matrix)

    Returns:
        Retracted point on the manifold (guaranteed to be positive definite)
    """
    # First-order approximation: simply add the tangent vector
    candidate = x + xi

    # Ensure the result is symmetric (it should be, but numerical errors can occur)
    candidate = 0.5 * (candidate + candidate.T)

    # Project to positive definite cone using eigenvalue clipping
    eigenvals, eigenvecs = anp.linalg.eigh(candidate)

    # Clip eigenvalues to ensure positive definiteness
    # Use a small positive threshold to maintain numerical stability
    min_eigenval = 1e-12
    eigenvals_clipped = anp.maximum(eigenvals, min_eigenval)

    # Reconstruct the matrix with clipped eigenvalues
    result = eigenvecs @ anp.diag(eigenvals_clipped) @ eigenvecs.T

    # Ensure the result is symmetric (numerical stability)
    result = 0.5 * (result + result.T)

    return result

# Riemannian median cost function with numerical safeguards
def cost_set_up(point, data):
    # riemannian median cost function given points
    distances = []
    numerical_issues = 0

    for i, x in enumerate(data):
        try:
            dist = manifold.dist(point, x)
            if anp.isfinite(dist) and dist >= 0:
                distances.append(dist)
            else:
                numerical_issues += 1
        except Exception:
            numerical_issues += 1

    if numerical_issues > 0:
        print(f"⚠ Cost computation: {numerical_issues}/{len(data)} points had numerical issues")

    if len(distances) == 0:
        print("⚠ Cost computation: All distances were invalid!")
        return float('inf')

    return sum(distances) / len(distances)

def cost(point):
    # compute the cost function
    return cost_set_up(point, data)

# Riemannian subgradient operator with numerical diagnostics
def subgradient_set_up(point, data):
    # compute the Riemannian subgradient of the cost function
    grad = anp.zeros_like(point)
    numerical_issues = []
    skipped_points = 0

    for i, x in enumerate(data):
        try:
            log_point = manifold.log(point, x)

            # Check for NaN/inf in log_point
            if not anp.all(anp.isfinite(log_point)):
                numerical_issues.append(f"Data point {i}: log_point contains NaN/inf")
                skipped_points += 1
                continue

            norm_log_point = manifold.norm(point, log_point)

            # Check for NaN/inf in norm
            if not anp.isfinite(norm_log_point):
                numerical_issues.append(f"Data point {i}: norm is NaN/inf")
                skipped_points += 1
                continue

            # Skip very small norms to avoid division issues
            if norm_log_point > 1e-12:
                contribution = - (log_point) / norm_log_point

                # Check contribution for numerical issues
                if not anp.all(anp.isfinite(contribution)):
                    numerical_issues.append(f"Data point {i}: contribution contains NaN/inf")
                    skipped_points += 1
                    continue

                grad += contribution
            else:
                skipped_points += 1

        except Exception as e:
            numerical_issues.append(f"Data point {i}: Exception {e}")
            skipped_points += 1
            continue

    # Report numerical issues if any occurred
    if numerical_issues:
        print(f"\n⚠ NUMERICAL ISSUES in subgradient computation:")
        print(f"  Skipped {skipped_points}/{len(data)} data points")
        for issue in numerical_issues[:5]:  # Show first 5 issues
            print(f"  - {issue}")
        if len(numerical_issues) > 5:
            print(f"  ... and {len(numerical_issues) - 5} more issues")

    result = grad / len(data)

    # Final check on the result
    if not anp.all(anp.isfinite(result)):
        print(f"⚠ Final subgradient contains NaN/inf values!")
        print(f"  Result norm: {anp.linalg.norm(result)}")
        # Return a zero gradient as fallback
        return anp.zeros_like(point)

    return result

def subgradient(point):
    # compute the subgradient
    return subgradient_set_up(point, data)

# Main Experiment Loop Over Dimensions
print("="*80)
print("RIEMANNIAN MEDIAN: EXPONENTIAL MAPS VS RETRACTIONS COMPARISON")
print("TESTING MULTIPLE DIMENSIONS WITH TIMING ANALYSIS")
print("="*80)

# Storage for results across all dimensions
all_results = {}
timing_results = []

# Choose dimension for the experiment
dim = DIMENSIONS_TO_TEST[0]  # Use first dimension from the list
print(f"\n" + "="*100)
print(f"DIMENSION {dim}: THREE-WAY COMPARISON EXPERIMENT")
print("="*100)

# Set random seed for reproducibility
anp.random.seed(RANDOM_SEED)

# set the manifold for current dimension
manifold = SymmetricPositiveDefinite(dim)

# Generate n truly random points for median computation with numerical checks
n = NUM_DATA_POINTS
data = []
max_attempts_per_point = 50

for i in range(n):
    attempts = 0
    while attempts < max_attempts_per_point:
        random_point = manifold.random_point()

        # Check if the point is numerically stable
        eigenvals = anp.linalg.eigvals(random_point)
        min_eigenval = anp.min(eigenvals)
        max_eigenval = anp.max(eigenvals)
        condition_number = max_eigenval / min_eigenval

        # Accept points that are well-conditioned and have reasonable eigenvalues
        if (min_eigenval > 1e-10 and
            condition_number < 1e12 and
            anp.all(anp.isfinite(random_point))):
            data.append(random_point)
            break

        attempts += 1

    if attempts >= max_attempts_per_point:
        print(f"⚠ Warning: Could not generate numerically stable point {i+1} after {max_attempts_per_point} attempts")
        # Use a well-conditioned fallback
        fallback = anp.eye(dim) + 0.1 * anp.random.randn(dim, dim)
        fallback = 0.5 * (fallback + fallback.T)
        fallback += (1e-6 + abs(anp.min(anp.linalg.eigvals(fallback)))) * anp.eye(dim)
        data.append(fallback)

print(f"Generated {len(data)} data points with numerical stability checks")

print(f"Problem setup:")
print(f"  Manifold: SPD({dim})")
print(f"  Number of data points: {n}")
print(f"  Problem: Riemannian median computation")

# Initialize points and set up common parameters
print(f"\nFinding suitable initial point...")

# Sample random points to understand the typical objective range
sample_objectives = []
for i in range(20):
    sample_point = manifold.random_point()
    sample_obj = cost(sample_point)
    sample_objectives.append(sample_obj)

min_obj = min(sample_objectives)
max_obj = max(sample_objectives)
avg_obj = sum(sample_objectives) / len(sample_objectives)

print(f"Sample statistics from 20 random points:")
print(f"  Min objective: {min_obj:.4f}")
print(f"  Max objective: {max_obj:.4f}")
print(f"  Average objective: {avg_obj:.4f}")

# Adjust target based on observed range
if max_obj < 5.0:
    target_objective = max_obj * 1.2  # Target 20% above the max we've seen
    print(f"Adjusting target to {target_objective:.4f} (20% above observed max)")
else:
    target_objective = 5.0

# Keep drawing random points until initial objective > target or max attempts reached
max_attempts = 300
attempt = 0
initial_point = None
initial_objective = 0
all_objectives = []

print(f"\nSearching for initial point with objective > {target_objective:.4f}...")

while attempt < max_attempts and initial_objective <= target_objective:
    candidate_point = manifold.random_point()
    candidate_objective = cost(candidate_point)
    all_objectives.append(candidate_objective)

    if candidate_objective > initial_objective:
        initial_point = candidate_point
        initial_objective = candidate_objective

    attempt += 1

    # Print progress every 50 attempts
    if attempt % 50 == 0:
        recent_max = max(all_objectives[-50:]) if len(all_objectives) >= 50 else max(all_objectives)
        print(f"Attempt {attempt}: Best overall = {initial_objective:.4f}, Recent max = {recent_max:.4f}")

initial_subgradient = subgradient(initial_point)

print(f"\nInitial point selected:")
print(f"Initial objective value: {initial_objective:.4f} (after {attempt} attempts)")
print(f"Range of all {attempt} attempts: [{min(all_objectives):.4f}, {max(all_objectives):.4f}]")
if initial_objective > target_objective:
    print(f"✓ Successfully found initial point with objective > {target_objective:.4f}")
else:
    print(f"⚠ Reached maximum attempts without finding objective > {target_objective:.4f}")
    print(f"  Using the best point found: {initial_objective:.4f}")

# EXPERIMENT 1: EXPONENTIAL MAPS (retraction_error=0, transport_error=0)
print("\n" + "="*80)
print("EXPERIMENT 1: EXPONENTIAL MAPS AND PARALLEL TRANSPORT")
print("="*80)
print("Phase 1: Estimating true minimum (300 iterations)")
print("-" * 80)

# Phase 1: Long run to estimate minimum with exponential maps
optimizer_exp_phase1 = RProximalBundle(
    manifold=manifold,
    retraction_map=manifold.exp,
    transport_map=manifold.transport,
    objective_function=cost,
    subgradient=subgradient,
    true_min_obj=0,  # Use 0 as baseline for first run
    initial_point=initial_point,
    initial_objective=initial_objective,
    initial_subgradient=initial_subgradient,
    adaptive_proximal=True,
    trust_parameter=0.2,
    transport_error=2,  # Exact exponential maps
    retraction_error=0,  
    know_minimizer=False,
    max_iter=300
)

print("Starting Experiment 1 Phase 1 optimization...")
optimizer_exp_phase1.run()
print("✓ Experiment 1 Phase 1 completed")

# Get estimated minimum for Experiment 1
estimated_minimum_exp = optimizer_exp_phase1.raw_objective_history[-1]
print(f"Experiment 1 Phase 1 Results:")
print(f"  Estimated minimum: {estimated_minimum_exp:.8f}")
print(f"  Initial objective: {initial_objective:.8f}")
print(f"  Improvement: {initial_objective - estimated_minimum_exp:.8f}")


# EXPERIMENT 2: SECOND-ORDER RETRACTIONS (retraction_error=1, transport_error=2)
print("\n" + "="*80)
print("EXPERIMENT 2: SECOND-ORDER RETRACTIONS AND VECTOR TRANSPORT")
print("="*80)
print("Phase 1: Estimating true minimum (300 iterations)")
print("-" * 80)

# Phase 1: Long run to estimate minimum with retractions
optimizer_ret_phase1 = RProximalBundle(
    manifold=manifold,
    retraction_map=manifold.retraction,
    transport_map=manifold.transport,
    objective_function=cost,
    subgradient=subgradient,
    true_min_obj=0,  # Use 0 as baseline for first run
    initial_point=initial_point,  # Same initial point for fair comparison
    initial_objective=initial_objective,
    initial_subgradient=initial_subgradient,
    adaptive_proximal=True,
    trust_parameter=0.2,
    transport_error=2,  # First-order retractions
    retraction_error=1,  # First-order vector transport
    know_minimizer=False,
    max_iter=300
)

print("Starting Experiment 2 Phase 1 optimization...")
optimizer_ret_phase1.run()
print("✓ Experiment 2 Phase 1 completed")

# Get estimated minimum for Experiment 2
estimated_minimum_ret = optimizer_ret_phase1.raw_objective_history[-1]
print(f"Experiment 2 Phase 1 Results:")
print(f"  Estimated minimum: {estimated_minimum_ret:.8f}")
print(f"  Initial objective: {initial_objective:.8f}")
print(f"  Improvement: {initial_objective - estimated_minimum_ret:.8f}")


# EXPERIMENT 3: FIRST-ORDER RETRACTIONS (custom implementation)
print("\n" + "="*80)
print("EXPERIMENT 3: FIRST-ORDER RETRACTIONS (CUSTOM IMPLEMENTATION)")
print("="*80)
print("Phase 1: Estimating true minimum (300 iterations)")
print("-" * 80)

# Create first-order retraction map
first_order_retraction_map = lambda x, xi: first_order_retraction(manifold, x, xi)

# Phase 1: Long run to estimate minimum with first-order retractions
optimizer_first_phase1 = RProximalBundle(
    manifold=manifold,
    retraction_map=first_order_retraction_map,
    transport_map=manifold.transport,
    objective_function=cost,
    subgradient=subgradient,
    true_min_obj=0,  # Use 0 as baseline for first run
    initial_point=initial_point,  # Same initial point for fair comparison
    initial_objective=initial_objective,
    initial_subgradient=initial_subgradient,
    adaptive_proximal=True,
    trust_parameter=0.2,
    transport_error=2,  # No transport error for custom retraction
    retraction_error=2,  # No retraction error for custom retraction
    know_minimizer=False,
    max_iter=300
)

print("Starting Experiment 3 Phase 1 optimization...")
optimizer_first_phase1.run()
print("✓ Experiment 3 Phase 1 completed")

# Get estimated minimum for Experiment 3
estimated_minimum_first = optimizer_first_phase1.raw_objective_history[-1]
print(f"Experiment 3 Phase 1 Results:")
print(f"  Estimated minimum: {estimated_minimum_first:.8f}")
print(f"  Initial objective: {initial_objective:.8f}")
print(f"  Improvement: {initial_objective - estimated_minimum_first:.8f}")

# FIND BEST ESTIMATED MINIMUM ACROSS ALL THREE EXPERIMENTS
print("\n" + "="*80)
print("FINDING BEST ESTIMATED MINIMUM ACROSS ALL EXPERIMENTS")
print("="*80)

# Use the best (lowest) estimated minimum across all three experiments
best_estimated_minimum = min(estimated_minimum_exp, estimated_minimum_ret, estimated_minimum_first)
print(f"Best estimated minimum found: {best_estimated_minimum:.8f}")
print(f"  From Exponential Maps: {estimated_minimum_exp:.8f}")
print(f"  From Second-Order Retractions: {estimated_minimum_ret:.8f}")
print(f"  From First-Order Retractions: {estimated_minimum_first:.8f}")
print("This will be used as the true minimum for all Phase 2 experiments")

# Show which experiment achieved the best minimum
if estimated_minimum_exp == best_estimated_minimum:
    print("Best minimum achieved by: Exponential Maps")
elif estimated_minimum_ret == best_estimated_minimum:
    print("Best minimum achieved by: Second-Order Retractions")
else:
    print("Best minimum achieved by: First-Order Retractions")

# PHASE 2 EXPERIMENTS WITH UNIFIED MINIMUM
print("\n" + "="*80)
print("PHASE 2: CONVERGENCE ANALYSIS WITH UNIFIED MINIMUM")
print("="*80)

# EXPERIMENT 1 PHASE 2: EXPONENTIAL MAPS
print("\n" + "-"*80)
print("EXPERIMENT 1 PHASE 2: EXPONENTIAL MAPS")
print(f"Using unified minimum: {best_estimated_minimum:.8f}")
print("-" * 80)

# Phase 2: Short run with unified minimum
optimizer_exp_phase2 = RProximalBundle(
    manifold=manifold,
    retraction_map=manifold.exp,
    transport_map=manifold.transport,
    objective_function=cost,
    subgradient=subgradient,
    true_min_obj=best_estimated_minimum,  # Use unified best minimum
    initial_point=initial_point,  # Same initial point
    initial_objective=initial_objective,
    initial_subgradient=initial_subgradient,
    adaptive_proximal=True,
    trust_parameter=0.2,
    transport_error=2,  # Exact exponential maps
    retraction_error=0,  # Exact parallel transport
    know_minimizer=True,
    max_iter=200
)

print(f"Initial gap for Phase 2: {initial_objective - best_estimated_minimum:.8f}")
optimizer_exp_phase2.run()
print("✓ Experiment 1 Phase 2 completed")

print(f"Experiment 1 Phase 2 Results:")
print(f"  Final gap: {optimizer_exp_phase2.objective_history[-1]:.8f}")

# EXPERIMENT 2 PHASE 2: SECOND-ORDER RETRACTIONS
print("\n" + "-"*80)
print("EXPERIMENT 2 PHASE 2: SECOND-ORDER RETRACTIONS")
print(f"Using unified minimum: {best_estimated_minimum:.8f}")
print("-" * 80)

# Phase 2: Short run with unified minimum
optimizer_ret_phase2 = RProximalBundle(
    manifold=manifold,
    retraction_map=manifold.retraction,
    transport_map=manifold.transport,
    objective_function=cost,
    subgradient=subgradient,
    true_min_obj=best_estimated_minimum,  # Use unified best minimum
    initial_point=initial_point,  # Same initial point
    initial_objective=initial_objective,
    initial_subgradient=initial_subgradient,
    adaptive_proximal=True,
    trust_parameter=0.2,
    transport_error=2,  # First-order retractions
    retraction_error=1,  # Second-order vector transport
    know_minimizer=True,
    max_iter=200
)

print(f"Initial gap for Phase 2: {initial_objective - best_estimated_minimum:.8f}")
optimizer_ret_phase2.run()
print("✓ Experiment 2 Phase 2 completed")

print(f"Experiment 2 Phase 2 Results:")
print(f"  Final gap: {optimizer_ret_phase2.objective_history[-1]:.8f}")

# EXPERIMENT 3 PHASE 2: FIRST-ORDER RETRACTIONS
print("\n" + "-"*80)
print("EXPERIMENT 3 PHASE 2: FIRST-ORDER RETRACTIONS")
print(f"Using unified minimum: {best_estimated_minimum:.8f}")
print("-" * 80)

# Phase 2: Short run with unified minimum
optimizer_first_phase2 = RProximalBundle(
    manifold=manifold,
    retraction_map=first_order_retraction_map,
    transport_map=manifold.transport,
    objective_function=cost,
    subgradient=subgradient,
    true_min_obj=best_estimated_minimum,  # Use unified best minimum
    initial_point=initial_point,  # Same initial point
    initial_objective=initial_objective,
    initial_subgradient=initial_subgradient,
    adaptive_proximal=True,
    trust_parameter=0.2,
    transport_error=2,  # No transport error for custom retraction
    retraction_error=2,  # No retraction error for custom retraction
    know_minimizer=True,
    max_iter=200
)

print(f"Initial gap for Phase 2: {initial_objective - best_estimated_minimum:.8f}")
optimizer_first_phase2.run()
print("✓ Experiment 3 Phase 2 completed")

print(f"Experiment 3 Phase 2 Results:")
print(f"  Final gap: {optimizer_first_phase2.objective_history[-1]:.8f}")

# Individual Plots for Each Experiment
print("\n" + "="*80)
print("GENERATING INDIVIDUAL PLOTS")
print("="*80)

# Plot Experiment 1 (Exponential Maps)
print("Plotting Experiment 1 Phase 1 convergence...")
optimizer_exp_phase1.plot_objective_versus_iter()

print("Plotting Experiment 1 Phase 2 convergence...")
optimizer_exp_phase2.plot_objective_versus_iter()

print("Plotting Experiment 1 Phase 2 log-log convergence...")
optimizer_exp_phase2.plot_objective_versus_iter(log_log=True)

# Plot Experiment 2 (Second-Order Retractions)
print("Plotting Experiment 2 Phase 1 convergence...")
optimizer_ret_phase1.plot_objective_versus_iter()

print("Plotting Experiment 2 Phase 2 convergence...")
optimizer_ret_phase2.plot_objective_versus_iter()

print("Plotting Experiment 2 Phase 2 log-log convergence...")
optimizer_ret_phase2.plot_objective_versus_iter(log_log=True)

# Plot Experiment 3 (First-Order Retractions)
print("Plotting Experiment 3 Phase 1 convergence...")
optimizer_first_phase1.plot_objective_versus_iter()

print("Plotting Experiment 3 Phase 2 convergence...")
optimizer_first_phase2.plot_objective_versus_iter()

print("Plotting Experiment 3 Phase 2 log-log convergence...")
optimizer_first_phase2.plot_objective_versus_iter(log_log=True)

# COMPARISON PLOTS: Use the better estimated minimum for fair comparison
print("\n" + "="*80)
print("GENERATING COMPARISON PLOTS")
print("="*80)

# Use the best (lowest) estimated minimum for fair comparison
best_estimated_minimum = min(estimated_minimum_exp, estimated_minimum_ret, estimated_minimum_first)
print(f"Using best estimated minimum for comparison: {best_estimated_minimum:.8f}")
print(f"  From Exponential Maps: {estimated_minimum_exp:.8f}")
print(f"  From Second-Order Retractions: {estimated_minimum_ret:.8f}")
print(f"  From First-Order Retractions: {estimated_minimum_first:.8f}")

# Recompute gaps using the same reference minimum
exp_gaps = [obj - best_estimated_minimum for obj in optimizer_exp_phase2.raw_objective_history]
ret_gaps = [obj - best_estimated_minimum for obj in optimizer_ret_phase2.raw_objective_history]
first_gaps = [obj - best_estimated_minimum for obj in optimizer_first_phase2.raw_objective_history]

# Ensure all gap arrays have the same length by trimming to the shortest one
min_length = min(len(exp_gaps), len(ret_gaps), len(first_gaps))
exp_gaps = exp_gaps[:min_length]
ret_gaps = ret_gaps[:min_length]
first_gaps = first_gaps[:min_length]

print(f"Plotting {min_length} iterations (trimmed to match all three experiments)")

# Comparison Plot 1: Linear Scale
plt.figure(figsize=(15, 10))

# Subplot 1: Linear scale comparison
plt.subplot(2, 2, 1)
iterations = range(min_length)
plt.plot(iterations, exp_gaps, 'b-', linewidth=2, label='Exponential Maps', alpha=0.8)
plt.plot(iterations, ret_gaps, 'r-', linewidth=2, label='Second-Order Retractions', alpha=0.8)
plt.plot(iterations, first_gaps, 'g-', linewidth=2, label='First-Order Retractions', alpha=0.8)

# Initialize variables to avoid NameError in later subplots
valid_exp_descent_indices = []
valid_exp_null_indices = []
valid_ret_descent_indices = []
valid_ret_null_indices = []
valid_first_descent_indices = []
valid_first_null_indices = []
exp_descent_values = []
exp_null_values = []
ret_descent_values = []
ret_null_values = []
first_descent_values = []
first_null_values = []

# Add scatter points for step types - Exponential Maps
if optimizer_exp_phase2.indices_of_descent_steps:
    valid_exp_descent_indices = [i for i in optimizer_exp_phase2.indices_of_descent_steps if i < min_length]
    if valid_exp_descent_indices:
        exp_descent_values = [exp_gaps[i] for i in valid_exp_descent_indices]
        plt.scatter(valid_exp_descent_indices, exp_descent_values,
                    color='darkblue', marker='o', s=8, alpha=0.7, zorder=5)

if optimizer_exp_phase2.indices_of_null_steps:
    valid_exp_null_indices = [i for i in optimizer_exp_phase2.indices_of_null_steps if i < min_length]
    if valid_exp_null_indices:
        exp_null_values = [exp_gaps[i] for i in valid_exp_null_indices]
        plt.scatter(valid_exp_null_indices, exp_null_values,
                    color='lightblue', marker='s', s=6, alpha=0.7, zorder=5)

# Add scatter points for step types - Second-Order Retractions
if optimizer_ret_phase2.indices_of_descent_steps:
    valid_ret_descent_indices = [i for i in optimizer_ret_phase2.indices_of_descent_steps if i < min_length]
    if valid_ret_descent_indices:
        ret_descent_values = [ret_gaps[i] for i in valid_ret_descent_indices]
        plt.scatter(valid_ret_descent_indices, ret_descent_values,
                    color='darkred', marker='o', s=8, alpha=0.7, zorder=5)

if optimizer_ret_phase2.indices_of_null_steps:
    valid_ret_null_indices = [i for i in optimizer_ret_phase2.indices_of_null_steps if i < min_length]
    if valid_ret_null_indices:
        ret_null_values = [ret_gaps[i] for i in valid_ret_null_indices]
        plt.scatter(valid_ret_null_indices, ret_null_values,
                    color='lightcoral', marker='s', s=6, alpha=0.7, zorder=5)

# Add scatter points for step types - First-Order Retractions
if optimizer_first_phase2.indices_of_descent_steps:
    valid_first_descent_indices = [i for i in optimizer_first_phase2.indices_of_descent_steps if i < min_length]
    if valid_first_descent_indices:
        first_descent_values = [first_gaps[i] for i in valid_first_descent_indices]
        plt.scatter(valid_first_descent_indices, first_descent_values,
                    color='darkgreen', marker='o', s=8, alpha=0.7, zorder=5)

if optimizer_first_phase2.indices_of_null_steps:
    valid_first_null_indices = [i for i in optimizer_first_phase2.indices_of_null_steps if i < min_length]
    if valid_first_null_indices:
        first_null_values = [first_gaps[i] for i in valid_first_null_indices]
        plt.scatter(valid_first_null_indices, first_null_values,
                    color='lightgreen', marker='s', s=6, alpha=0.7, zorder=5)

plt.title('Three-Way Comparison - Linear Scale')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Log scale comparison
plt.subplot(2, 2, 2)
plt.plot(iterations, exp_gaps, 'b-', linewidth=2, label='Exponential Maps', alpha=0.8)
plt.plot(iterations, ret_gaps, 'r-', linewidth=2, label='Second-Order Retractions', alpha=0.8)
plt.plot(iterations, first_gaps, 'g-', linewidth=2, label='First-Order Retractions', alpha=0.8)

# Add scatter points for step types - Exponential Maps (reuse filtered indices)
if valid_exp_descent_indices:
    plt.scatter(valid_exp_descent_indices, exp_descent_values,
                color='darkblue', marker='o', s=8, alpha=0.7, zorder=5)

if valid_exp_null_indices:
    plt.scatter(valid_exp_null_indices, exp_null_values,
                color='lightblue', marker='s', s=6, alpha=0.7, zorder=5)

# Add scatter points for step types - Second-Order Retractions (reuse filtered indices)
if valid_ret_descent_indices:
    plt.scatter(valid_ret_descent_indices, ret_descent_values,
                color='darkred', marker='o', s=8, alpha=0.7, zorder=5)

if valid_ret_null_indices:
    plt.scatter(valid_ret_null_indices, ret_null_values,
                color='lightcoral', marker='s', s=6, alpha=0.7, zorder=5)

# Add scatter points for step types - First-Order Retractions (reuse filtered indices)
if valid_first_descent_indices:
    plt.scatter(valid_first_descent_indices, first_descent_values,
                color='darkgreen', marker='o', s=8, alpha=0.7, zorder=5)

if valid_first_null_indices:
    plt.scatter(valid_first_null_indices, first_null_values,
                color='lightgreen', marker='s', s=6, alpha=0.7, zorder=5)

plt.title('Three-Way Comparison - Log Scale')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 3: Proximal parameter evolution comparison
plt.subplot(2, 2, 3)
plt.plot(range(len(optimizer_exp_phase2.proximal_parameter_history)),
         optimizer_exp_phase2.proximal_parameter_history, 'b-', linewidth=2,
         label='Exponential Maps', alpha=0.8)
plt.plot(range(len(optimizer_ret_phase2.proximal_parameter_history)),
         optimizer_ret_phase2.proximal_parameter_history, 'r-', linewidth=2,
         label='Second-Order Retractions', alpha=0.8)
plt.plot(range(len(optimizer_first_phase2.proximal_parameter_history)),
         optimizer_first_phase2.proximal_parameter_history, 'g-', linewidth=2,
         label='First-Order Retractions', alpha=0.8)
plt.title('Proximal Parameter Evolution')
plt.xlabel('Iteration Number')
plt.ylabel('Proximal Parameter (ρ)')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 4: Log-log scale comparison
plt.subplot(2, 2, 4)
plt.plot(iterations, exp_gaps, 'b-', linewidth=2, label='Exponential Maps', alpha=0.8)
plt.plot(iterations, ret_gaps, 'r-', linewidth=2, label='Second-Order Retractions', alpha=0.8)
plt.plot(iterations, first_gaps, 'g-', linewidth=2, label='First-Order Retractions', alpha=0.8)
plt.title('Three-Way Comparison - Log-Log Scale')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('RMedianExpVsRetraction_Comparison.png', dpi=150, bbox_inches='tight')
print("✓ Comparison plots saved as 'RMedianExpVsRetraction_Comparison.png'")

# PADDED COMPARISON PLOTS (showing full convergence dynamics)
print("\n" + "="*80)
print("GENERATING PADDED COMPARISON PLOTS (FULL DYNAMICS)")
print("="*80)

# Recompute gaps using the same reference minimum (without trimming)
exp_gaps_full = [obj - best_estimated_minimum for obj in optimizer_exp_phase2.raw_objective_history]
ret_gaps_full = [obj - best_estimated_minimum for obj in optimizer_ret_phase2.raw_objective_history]
first_gaps_full = [obj - best_estimated_minimum for obj in optimizer_first_phase2.raw_objective_history]

# Pad the shorter arrays by repeating their last values
max_length = max(len(exp_gaps_full), len(ret_gaps_full), len(first_gaps_full))
if len(exp_gaps_full) < max_length:
    last_exp_value = exp_gaps_full[-1]
    exp_gaps_full.extend([last_exp_value] * (max_length - len(exp_gaps_full)))
if len(ret_gaps_full) < max_length:
    last_ret_value = ret_gaps_full[-1]
    ret_gaps_full.extend([last_ret_value] * (max_length - len(ret_gaps_full)))
if len(first_gaps_full) < max_length:
    last_first_value = first_gaps_full[-1]
    first_gaps_full.extend([last_first_value] * (max_length - len(first_gaps_full)))

print(f"Padded all arrays to {max_length} iterations")
print(f"  Original lengths: Exp={len(optimizer_exp_phase2.raw_objective_history)}, Ret={len(optimizer_ret_phase2.raw_objective_history)}, First={len(optimizer_first_phase2.raw_objective_history)}")

# Initialize variables for step type plotting
valid_exp_descent_indices = []
valid_exp_null_indices = []
valid_ret_descent_indices = []
valid_ret_null_indices = []
valid_ret_doubling_indices = []
valid_first_descent_indices = []
valid_first_null_indices = []
valid_first_doubling_indices = []
exp_descent_values = []
exp_null_values = []
ret_descent_values = []
ret_null_values = []
ret_doubling_values = []
first_descent_values = []
first_null_values = []
first_doubling_values = []

# Create final comparison plots: Proximal Parameter Evolution and Log Scale
plt.figure(figsize=(15, 6))

# Subplot 1: Proximal parameter evolution comparison
plt.subplot(1, 2, 1)
plt.plot(range(len(optimizer_exp_phase2.proximal_parameter_history)),
         optimizer_exp_phase2.proximal_parameter_history, 'b-', linewidth=2,
         label='Exponential Maps', alpha=0.8)
plt.plot(range(len(optimizer_ret_phase2.proximal_parameter_history)),
         optimizer_ret_phase2.proximal_parameter_history, 'r-', linewidth=2,
         label='Second-Order Retractions', alpha=0.8)
plt.plot(range(len(optimizer_first_phase2.proximal_parameter_history)),
         optimizer_first_phase2.proximal_parameter_history, 'g-', linewidth=2,
         label='First-Order Retractions', alpha=0.8)
plt.title('Proximal Parameter Evolution - Full Dynamics')
plt.xlabel('Iteration Number')
plt.ylabel('Proximal Parameter (ρ)')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Log scale comparison (padded)
plt.subplot(1, 2, 2)
iterations_full = range(max_length)
plt.plot(iterations_full, exp_gaps_full, 'b-', linewidth=2, label='Exponential Maps', alpha=0.8)
plt.plot(iterations_full, ret_gaps_full, 'r-', linewidth=2, label='Second-Order Retractions', alpha=0.8)
plt.plot(iterations_full, first_gaps_full, 'g-', linewidth=2, label='First-Order Retractions', alpha=0.8)

# Add scatter points for step types - Exponential Maps
if optimizer_exp_phase2.indices_of_descent_steps:
    valid_exp_descent_indices = [i for i in optimizer_exp_phase2.indices_of_descent_steps if i < len(exp_gaps_full)]
    if valid_exp_descent_indices:
        exp_descent_values = [exp_gaps_full[i] for i in valid_exp_descent_indices]
        plt.scatter(valid_exp_descent_indices, exp_descent_values,
                    color='darkblue', marker='o', s=12, alpha=0.8, zorder=5, label='Exp: Descent')

if optimizer_exp_phase2.indices_of_null_steps:
    valid_exp_null_indices = [i for i in optimizer_exp_phase2.indices_of_null_steps if i < len(exp_gaps_full)]
    if valid_exp_null_indices:
        exp_null_values = [exp_gaps_full[i] for i in valid_exp_null_indices]
        plt.scatter(valid_exp_null_indices, exp_null_values,
                    color='lightblue', marker='s', s=8, alpha=0.8, zorder=5, label='Exp: Null')

# Add scatter points for step types - Second-Order Retractions
if optimizer_ret_phase2.indices_of_descent_steps:
    valid_ret_descent_indices = [i for i in optimizer_ret_phase2.indices_of_descent_steps if i < len(ret_gaps_full)]
    if valid_ret_descent_indices:
        ret_descent_values = [ret_gaps_full[i] for i in valid_ret_descent_indices]
        plt.scatter(valid_ret_descent_indices, ret_descent_values,
                    color='darkred', marker='o', s=12, alpha=0.8, zorder=5, label='Ret: Descent')

if optimizer_ret_phase2.indices_of_null_steps:
    valid_ret_null_indices = [i for i in optimizer_ret_phase2.indices_of_null_steps if i < len(ret_gaps_full)]
    if valid_ret_null_indices:
        ret_null_values = [ret_gaps_full[i] for i in valid_ret_null_indices]
        plt.scatter(valid_ret_null_indices, ret_null_values,
                    color='lightcoral', marker='s', s=8, alpha=0.8, zorder=5, label='Ret: Null')

if optimizer_ret_phase2.indices_of_proximal_doubling_steps:
    valid_ret_doubling_indices = [i for i in optimizer_ret_phase2.indices_of_proximal_doubling_steps if i < len(ret_gaps_full)]
    if valid_ret_doubling_indices:
        ret_doubling_values = [ret_gaps_full[i] for i in valid_ret_doubling_indices]
        plt.scatter(valid_ret_doubling_indices, ret_doubling_values,
                    color='purple', marker='^', s=10, alpha=0.8, zorder=5, label='Ret: Doubling')

# Add scatter points for step types - First-Order Retractions
if optimizer_first_phase2.indices_of_descent_steps:
    valid_first_descent_indices = [i for i in optimizer_first_phase2.indices_of_descent_steps if i < len(first_gaps_full)]
    if valid_first_descent_indices:
        first_descent_values = [first_gaps_full[i] for i in valid_first_descent_indices]
        plt.scatter(valid_first_descent_indices, first_descent_values,
                    color='darkgreen', marker='o', s=12, alpha=0.8, zorder=5, label='First: Descent')

if optimizer_first_phase2.indices_of_null_steps:
    valid_first_null_indices = [i for i in optimizer_first_phase2.indices_of_null_steps if i < len(first_gaps_full)]
    if valid_first_null_indices:
        first_null_values = [first_gaps_full[i] for i in valid_first_null_indices]
        plt.scatter(valid_first_null_indices, first_null_values,
                    color='lightgreen', marker='s', s=8, alpha=0.8, zorder=5, label='First: Null')

if optimizer_first_phase2.indices_of_proximal_doubling_steps:
    valid_first_doubling_indices = [i for i in optimizer_first_phase2.indices_of_proximal_doubling_steps if i < len(first_gaps_full)]
    if valid_first_doubling_indices:
        first_doubling_values = [first_gaps_full[i] for i in valid_first_doubling_indices]
        plt.scatter(valid_first_doubling_indices, first_doubling_values,
                    color='orange', marker='^', s=10, alpha=0.8, zorder=5, label='First: Doubling')

# Add vertical lines to show where padding starts for each method
orig_exp_len = len(optimizer_exp_phase2.raw_objective_history)
orig_ret_len = len(optimizer_ret_phase2.raw_objective_history)
orig_first_len = len(optimizer_first_phase2.raw_objective_history)
if orig_exp_len < max_length:
    plt.axvline(x=orig_exp_len-1, color='blue', linestyle='--', alpha=0.5, linewidth=1)
if orig_ret_len < max_length:
    plt.axvline(x=orig_ret_len-1, color='red', linestyle='--', alpha=0.5, linewidth=1)
if orig_first_len < max_length:
    plt.axvline(x=orig_first_len-1, color='green', linestyle='--', alpha=0.5, linewidth=1)

plt.title('Three-Way Comparison - Full Dynamics (Log Scale)')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('RMedianExpVsRetraction_FullDynamics.png', dpi=150, bbox_inches='tight')
print("✓ Padded comparison plots saved as 'RMedianExpVsRetraction_FullDynamics.png'")

# FINAL SUMMARY
print("\n" + "="*80)
print("EXPERIMENT SUMMARY")
print("="*80)

print("EXPERIMENT 1 - EXPONENTIAL MAPS AND PARALLEL TRANSPORT:")
print(f"  Phase 1 estimated minimum: {estimated_minimum_exp:.8f}")
print(f"  Phase 2 final gap: {optimizer_exp_phase2.objective_history[-1]:.8f}")
print(f"  Phase 2 iterations: {len(optimizer_exp_phase2.objective_history)}")
print(f"  Phase 2 descent steps: {len(optimizer_exp_phase2.indices_of_descent_steps)}")
print(f"  Phase 2 null steps: {len(optimizer_exp_phase2.indices_of_null_steps)}")

print("\nEXPERIMENT 2 - SECOND-ORDER RETRACTIONS AND VECTOR TRANSPORT:")
print(f"  Phase 1 estimated minimum: {estimated_minimum_ret:.8f}")
print(f"  Phase 2 final gap: {optimizer_ret_phase2.objective_history[-1]:.8f}")
print(f"  Phase 2 iterations: {len(optimizer_ret_phase2.objective_history)}")
print(f"  Phase 2 descent steps: {len(optimizer_ret_phase2.indices_of_descent_steps)}")
print(f"  Phase 2 null steps: {len(optimizer_ret_phase2.indices_of_null_steps)}")

print("\nEXPERIMENT 3 - FIRST-ORDER RETRACTIONS (CUSTOM IMPLEMENTATION):")
print(f"  Phase 1 estimated minimum: {estimated_minimum_first:.8f}")
print(f"  Phase 2 final gap: {optimizer_first_phase2.objective_history[-1]:.8f}")
print(f"  Phase 2 iterations: {len(optimizer_first_phase2.objective_history)}")
print(f"  Phase 2 descent steps: {len(optimizer_first_phase2.indices_of_descent_steps)}")
print(f"  Phase 2 null steps: {len(optimizer_first_phase2.indices_of_null_steps)}")

print("\nCOMPARISON (using best estimated minimum):")
print(f"  Best estimated minimum used: {best_estimated_minimum:.8f}")
print(f"  Exponential Maps final gap: {exp_gaps[-1]:.8f}")
print(f"  Second-Order Retractions final gap: {ret_gaps[-1]:.8f}")
print(f"  First-Order Retractions final gap: {first_gaps[-1]:.8f}")
print(f"  Performance ratio (Second/Exp): {ret_gaps[-1]/exp_gaps[-1]:.4f}")
print(f"  Performance ratio (First/Exp): {first_gaps[-1]/exp_gaps[-1]:.4f}")
print(f"  Performance ratio (First/Second): {first_gaps[-1]/ret_gaps[-1]:.4f}")

print("="*80)
print("EXPERIMENT COMPLETED SUCCESSFULLY!")
print("="*80)

# Save the final comparison plot
print("\n" + "="*80)
print("SAVING PLOTS")
print("="*80)
print("✓ All individual plots already displayed above")
print("✓ Final comparison plot already saved as 'RMedianExpVsRetraction_Comparison.png'")
print("="*80)
print("EXPERIMENT COMPLETED!")
print("All plots have been saved as image files.")
print("="*80)