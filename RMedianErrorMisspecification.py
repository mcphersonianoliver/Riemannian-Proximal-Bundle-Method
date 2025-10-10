# %%
# Riemannian Median Experiment: Testing Error Constant Misspecification
import autograd.numpy as anp
import matplotlib.pyplot as plt
from pymanopt.manifolds import SymmetricPositiveDefinite
from src.RiemannianProximalBundle import RProximalBundle

# ============================================================================
# EXPERIMENT CONFIGURATION - MODIFY THESE VALUES AS NEEDED
# ============================================================================

# Define constant settings to test with each retraction type
CONSTANT_SETTINGS = [0, 1, 5, 2000]

# Error constant combinations to test: (retraction_type, retraction_error, transport_error, description)
ERROR_SETTINGS = []

# Add first-order retraction experiments
for constant in CONSTANT_SETTINGS:
    ERROR_SETTINGS.append((
        "first_order",
        constant,
        constant,
        f"First-Order Retraction (error = {constant})"
    ))

# Add second-order retraction experiments
for constant in CONSTANT_SETTINGS:
    ERROR_SETTINGS.append((
        "second_order",
        constant,
        constant,
        f"Second-Order Retraction (error = {constant})"
    ))

# You can modify this list to test different combinations, for example:
# ERROR_SETTINGS = [
#     (0, 0, "Perfect Setting"),
#     (0.5, 0.5, "Small Misspecification"),
#     (1, 1, "Standard Setting"),
#     (2, 2, "Moderate Misspecification"),
#     (5, 5, "High Misspecification"),
#     (10, 10, "Very High Misspecification")
# ]

# Problem parameters
MANIFOLD_DIM = 10
NUM_DATA_POINTS = 200
PHASE1_ITERATIONS = 300  # For estimating minimum
PHASE2_ITERATIONS = 200  # For convergence analysis
TRUST_PARAMETER = 0.2

# Random seed for reproducibility
RANDOM_SEED = 50  # Change this value to get different (but repeatable) experiments

print(f"Configured to test {len(ERROR_SETTINGS)} error settings:")
for ret_type, ret_err, trans_err, desc in ERROR_SETTINGS:
    print(f"  - {desc}: retraction_error={ret_err}, transport_error={trans_err}")
print(f"Random seed: {RANDOM_SEED} (for reproducibility)")

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

# Problem Setup
print("="*80)
print("RIEMANNIAN MEDIAN: ERROR CONSTANT MISSPECIFICATION EXPERIMENT")
print("="*80)

# Set random seed for reproducibility
anp.random.seed(RANDOM_SEED)

# set the manifold a priori
dim = MANIFOLD_DIM
manifold = SymmetricPositiveDefinite(dim)

# Generate n truly random points for median computation (reproducible)
n = NUM_DATA_POINTS
data = []
for _ in range(n):
    random_point = manifold.random_point()
    data.append(random_point)

print(f"Problem setup:")
print(f"  Manifold: SPD({dim})")
print(f"  Number of data points: {n}")
print(f"  Problem: Riemannian median computation")

# Initialize points and set up common parameters
print(f"\nFinding suitable initial point (using seed {RANDOM_SEED})...")

# Sample random points to understand the typical objective range (reproducible)
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

# ERROR CONSTANT MISSPECIFICATION EXPERIMENTS
print("\n" + "="*80)
print("PHASE 1: FINDING BEST ESTIMATED MINIMUM ACROSS ALL SETTINGS")
print("="*80)

# Store Phase 1 results to find the best minimum
phase1_results = {}

print("Running Phase 1 for all error settings to find the best estimated minimum...")

for retraction_type, retraction_error, transport_error, description in ERROR_SETTINGS:
    print(f"\n" + "-"*80)
    print(f"EXPERIMENT: {description}")
    print(f"  Retraction Type: {retraction_type}")
    print(f"  Retraction Error: {retraction_error}")
    print(f"  Transport Error: {transport_error}")
    print("-"*80)

    # Choose retraction map based on type
    if retraction_type == "first_order":
        # Use custom first-order retraction
        retraction_map = lambda x, xi: first_order_retraction(manifold, x, xi)
    elif retraction_type == "second_order":
        # Use second-order retraction (with specified error constants)
        retraction_map = manifold.retraction
    else:
        raise ValueError(f"Unknown retraction type: {retraction_type}")

    # Phase 1: Long run to estimate minimum
    print(f"Running Phase 1 ({PHASE1_ITERATIONS} iterations) to estimate minimum...")

    optimizer_phase1 = RProximalBundle(
        manifold=manifold,
        retraction_map=retraction_map,
        transport_map=manifold.transport,
        objective_function=cost,
        subgradient=subgradient,
        true_min_obj=0,  # Use 0 as baseline for first run
        initial_point=initial_point,  # Same initial point for fair comparison
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        adaptive_proximal=True,
        trust_parameter=TRUST_PARAMETER,
        transport_error=transport_error,
        retraction_error=retraction_error,
        know_minimizer=False,
        max_iter=PHASE1_ITERATIONS
    )

    optimizer_phase1.run()

    # Get estimated minimum
    estimated_minimum = optimizer_phase1.raw_objective_history[-1]
    print(f"  Estimated minimum: {estimated_minimum:.8f}")
    print(f"  Improvement: {initial_objective - estimated_minimum:.8f}")

    # Store Phase 1 results
    phase1_results[(retraction_type, retraction_error, transport_error)] = {
        'description': description,
        'optimizer': optimizer_phase1,
        'estimated_minimum': estimated_minimum
    }

# Find the best (lowest) estimated minimum across all settings
best_estimated_minimum = min([results['estimated_minimum'] for results in phase1_results.values()])
print(f"\n" + "="*60)
print(f"BEST ESTIMATED MINIMUM FOUND: {best_estimated_minimum:.8f}")
print("This will be used as the true minimum for all Phase 2 experiments")
print("="*60)

# Show which setting achieved the best minimum
for (ret_type, ret_err, trans_err), results in phase1_results.items():
    if results['estimated_minimum'] == best_estimated_minimum:
        print(f"Best minimum achieved by: {results['description']}")
        break

# PHASE 2: CONVERGENCE ANALYSIS WITH UNIFIED MINIMUM
print("\n" + "="*80)
print("PHASE 2: CONVERGENCE ANALYSIS WITH UNIFIED MINIMUM")
print("="*80)

# Store final results for comparison
experiment_results = {}

for retraction_type, retraction_error, transport_error, description in ERROR_SETTINGS:
    print(f"\n" + "-"*80)
    print(f"PHASE 2: {description}")
    print(f"  Retraction Type: {retraction_type}")
    print(f"  Retraction Error: {retraction_error}")
    print(f"  Transport Error: {transport_error}")
    print(f"  Using unified minimum: {best_estimated_minimum:.8f}")
    print("-"*80)

    # Choose retraction map (same logic as Phase 1)
    if retraction_type == "first_order":
        # Use custom first-order retraction
        retraction_map = lambda x, xi: first_order_retraction(manifold, x, xi)
    elif retraction_type == "second_order":
        # Use second-order retraction (with specified error constants)
        retraction_map = manifold.retraction
    else:
        raise ValueError(f"Unknown retraction type: {retraction_type}")

    # Phase 2: Convergence analysis with unified minimum
    optimizer_phase2 = RProximalBundle(
        manifold=manifold,
        retraction_map=retraction_map,
        transport_map=manifold.transport,
        objective_function=cost,
        subgradient=subgradient,
        true_min_obj=best_estimated_minimum,  # Use unified best minimum
        initial_point=initial_point,  # Same initial point
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        adaptive_proximal=True,
        trust_parameter=TRUST_PARAMETER,
        transport_error=transport_error,
        retraction_error=retraction_error,
        know_minimizer=True,
        max_iter=PHASE2_ITERATIONS
    )

    print(f"Running Phase 2 ({PHASE2_ITERATIONS} iterations) with unified minimum...")
    print(f"Initial gap: {initial_objective - best_estimated_minimum:.8f}")
    optimizer_phase2.run()

    print(f"Phase 2 Results:")
    print(f"  Final gap: {optimizer_phase2.objective_history[-1]:.8f}")
    print(f"  Iterations: {len(optimizer_phase2.objective_history)}")
    print(f"  Descent steps: {len(optimizer_phase2.indices_of_descent_steps)}")
    print(f"  Null steps: {len(optimizer_phase2.indices_of_null_steps)}")
    print(f"  Proximal doubling steps: {len(optimizer_phase2.indices_of_proximal_doubling_steps)}")

    # Store results
    experiment_results[(retraction_type, retraction_error, transport_error)] = {
        'description': description,
        'retraction_type': retraction_type,
        'optimizer_phase1': phase1_results[(retraction_type, retraction_error, transport_error)]['optimizer'],
        'optimizer_phase2': optimizer_phase2,
        'estimated_minimum': phase1_results[(retraction_type, retraction_error, transport_error)]['estimated_minimum'],
        'unified_minimum': best_estimated_minimum
    }

# INDIVIDUAL PLOTS FOR EACH EXPERIMENT
print("\n" + "="*80)
print("GENERATING INDIVIDUAL PLOTS")
print("="*80)

for (ret_type, ret_err, trans_err), results in experiment_results.items():
    print(f"Plotting {results['description']}...")

    # Plot Phase 1 convergence
    print(f"  Phase 1 convergence...")
    results['optimizer_phase1'].plot_objective_versus_iter()

    # Plot Phase 2 convergence
    print(f"  Phase 2 convergence...")
    results['optimizer_phase2'].plot_objective_versus_iter()

    # Plot Phase 2 log-log convergence
    print(f"  Phase 2 log-log convergence...")
    results['optimizer_phase2'].plot_objective_versus_iter(log_log=True)

# COMPARISON PLOTS
print("\n" + "="*80)
print("GENERATING COMPARISON PLOTS")
print("="*80)

# Use the unified minimum that was already computed
print(f"Using unified minimum for comparison plots: {best_estimated_minimum:.8f}")

# Compute gaps for all experiments using the same reference
comparison_data = {}
for (ret_type, ret_err, trans_err), results in experiment_results.items():
    gaps = [obj - best_estimated_minimum for obj in results['optimizer_phase2'].raw_objective_history]
    comparison_data[(ret_type, ret_err, trans_err)] = {
        'gaps': gaps,
        'optimizer': results['optimizer_phase2'],
        'description': results['description'],
        'retraction_type': ret_type
    }

# Find maximum length for padding
max_length = max([len(data['gaps']) for data in comparison_data.values()])

# Pad all gap arrays
for key, data in comparison_data.items():
    gaps = data['gaps']
    if len(gaps) < max_length:
        last_value = gaps[-1]
        gaps.extend([last_value] * (max_length - len(gaps)))
    data['gaps_padded'] = gaps

print(f"Padded all experiments to {max_length} iterations")

# Separate data by retraction type
first_order_data = {k: v for k, v in comparison_data.items() if v['retraction_type'] == 'first_order'}
second_order_data = {k: v for k, v in comparison_data.items() if v['retraction_type'] == 'second_order'}

# Colors for constant settings
constant_colors = {
    0: '#1f77b4',   # Blue
    1: '#ff7f0e',   # Orange
    5: '#2ca02c',   # Green
    2000: '#d62728',  # Red
}

# SEPARATE GRAPHS FOR EACH RETRACTION TYPE
print("Creating separate graphs for each retraction type...")

# First-order retraction graph
plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
iterations_full = range(max_length)

for (ret_type, ret_err, trans_err), data in first_order_data.items():
    color = constant_colors[ret_err]
    plt.plot(iterations_full, data['gaps_padded'], color=color, linewidth=2,
             label=f'Error = {ret_err}', alpha=0.8)

    # Add scatter points for step types
    optimizer = data['optimizer']
    orig_len = len([obj - best_estimated_minimum for obj in optimizer.raw_objective_history])

    # Descent steps
    if optimizer.indices_of_descent_steps:
        valid_indices = [j for j in optimizer.indices_of_descent_steps if j < orig_len]
        if valid_indices:
            values = [data['gaps_padded'][j] for j in valid_indices]
            plt.scatter(valid_indices, values, color=color, marker='o', s=12, alpha=0.7, zorder=5)

plt.title('First-Order Retraction')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Second-order retraction graph
plt.subplot(1, 3, 2)

for (ret_type, ret_err, trans_err), data in second_order_data.items():
    color = constant_colors[ret_err]
    plt.plot(iterations_full, data['gaps_padded'], color=color, linewidth=2,
             label=f'Error = {ret_err}', alpha=0.8)

    # Add scatter points for step types
    optimizer = data['optimizer']
    orig_len = len([obj - best_estimated_minimum for obj in optimizer.raw_objective_history])

    # Descent steps
    if optimizer.indices_of_descent_steps:
        valid_indices = [j for j in optimizer.indices_of_descent_steps if j < orig_len]
        if valid_indices:
            values = [data['gaps_padded'][j] for j in valid_indices]
            plt.scatter(valid_indices, values, color=color, marker='o', s=12, alpha=0.7, zorder=5)

plt.title('Second-Order Retraction')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Combined graph with both retraction types
plt.subplot(1, 3, 3)

# Plot first-order with solid lines
for (ret_type, ret_err, trans_err), data in first_order_data.items():
    color = constant_colors[ret_err]
    plt.plot(iterations_full, data['gaps_padded'], color=color, linewidth=2, linestyle='-',
             label=f'1st Order, Error = {ret_err}', alpha=0.8)

# Plot second-order with dashed lines
for (ret_type, ret_err, trans_err), data in second_order_data.items():
    color = constant_colors[ret_err]
    plt.plot(iterations_full, data['gaps_padded'], color=color, linewidth=2, linestyle='--',
             label=f'2nd Order, Error = {ret_err}', alpha=0.8)

plt.title('Both Retractions Combined')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('RMedianErrorMisspecification_RetractionsComparison.png', dpi=150, bbox_inches='tight')
print("✓ Retraction comparison saved as 'RMedianErrorMisspecification_RetractionsComparison.png'")

# Additional detailed analysis plot
plt.figure(figsize=(12, 5))

# Subplot 1: Combined comparison with better legend
plt.subplot(1, 2, 1)

for (ret_type, ret_err, trans_err), data in comparison_data.items():
    color = constant_colors[ret_err]
    linestyle = '-' if ret_type == 'first_order' else '--'
    label = f'{ret_type.replace("_", "-").title()}, Error = {ret_err}'
    plt.plot(iterations_full, data['gaps_padded'], color=color, linewidth=2,
             linestyle=linestyle, label=label, alpha=0.8)

plt.title('Error Constant Misspecification - Both Retractions')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Proximal parameter evolution
plt.subplot(1, 2, 2)

for (ret_type, ret_err, trans_err), data in comparison_data.items():
    color = constant_colors[ret_err]
    linestyle = '-' if ret_type == 'first_order' else '--'
    label = f'{ret_type.replace("_", "-").title()}, Error = {ret_err}'
    optimizer = data['optimizer']
    plt.plot(range(len(optimizer.proximal_parameter_history)), optimizer.proximal_parameter_history,
             color=color, linewidth=2, linestyle=linestyle, label=label, alpha=0.8)

plt.title('Proximal Parameter Evolution')
plt.xlabel('Iteration Number')
plt.ylabel('Proximal Parameter (ρ)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('RMedianErrorMisspecification_Analysis.png', dpi=150, bbox_inches='tight')
print("✓ Detailed analysis saved as 'RMedianErrorMisspecification_Analysis.png'")

# FINAL SUMMARY
print("\n" + "="*80)
print("ERROR MISSPECIFICATION EXPERIMENT SUMMARY")
print("="*80)

for (ret_type, ret_err, trans_err), results in experiment_results.items():
    print(f"\n{results['description']} - {ret_type.title()}, Error=({ret_err},{trans_err}):")
    print(f"  Phase 1 estimated minimum: {results['estimated_minimum']:.8f}")
    print(f"  Phase 2 final gap: {results['optimizer_phase2'].objective_history[-1]:.8f}")
    print(f"  Phase 2 iterations: {len(results['optimizer_phase2'].objective_history)}")
    print(f"  Descent steps: {len(results['optimizer_phase2'].indices_of_descent_steps)}")
    print(f"  Null steps: {len(results['optimizer_phase2'].indices_of_null_steps)}")
    print(f"  Proximal doubling steps: {len(results['optimizer_phase2'].indices_of_proximal_doubling_steps)}")

print("\nCOMPARISON ANALYSIS:")
print(f"  Unified minimum used for all experiments: {best_estimated_minimum:.8f}")

# Show individual Phase 1 estimates vs unified minimum
print("\n  Individual Phase 1 estimated minimums:")
for (ret_type, ret_err, trans_err), results in experiment_results.items():
    individual_min = results['estimated_minimum']
    gap_from_unified = individual_min - best_estimated_minimum
    print(f"    {ret_type.title()}, Error=({ret_err},{trans_err}): {individual_min:.8f} (gap from unified: {gap_from_unified:.8f})")

print("\n  Phase 2 performance with unified minimum:")
for (ret_type, ret_err, trans_err), data in comparison_data.items():
    final_gap = data['gaps_padded'][-1]
    print(f"    {ret_type.title()}, Error=({ret_err},{trans_err}): Final gap={final_gap:.8f}")

print("="*80)
print("ERROR MISSPECIFICATION EXPERIMENT COMPLETED!")
print(f"Note: Experiment is reproducible using random seed {RANDOM_SEED}")
print("="*80)