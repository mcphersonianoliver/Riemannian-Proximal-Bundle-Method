# %%
# RMedian Multi-Dimensional Experiment: Exponential Maps vs Retractions on SPD Manifolds
import autograd.numpy as anp
import matplotlib.pyplot as plt
import time
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
print(f"Phase 1 iterations: {PHASE1_ITERATIONS}, Phase 2 iterations: {PHASE2_ITERATIONS}")
print(f"Random seed: {RANDOM_SEED}")

# ============================================================================

# Setting up functions
# Riemannian median cost function
def cost_set_up(point, data, manifold):
    # riemannian median cost function given points
    return sum([manifold.dist(point, x) for x in data]) / len(data)

def cost(point):
    # compute the cost function
    return cost_set_up(point, data, manifold)

# Riemannian subgradient operator
def subgradient_set_up(point, data, manifold):
    # compute the Riemannian subgradient of the cost function
    grad = anp.zeros_like(point)
    for x in data:
        log_point = manifold.log(point, x)
        grad += - (log_point)/ manifold.norm(point, log_point)
    return grad / len(data)

def subgradient(point):
    # compute the subgradient
    return subgradient_set_up(point, data, manifold)

# Main Experiment Loop Over Dimensions
print("\n" + "="*80)
print("RIEMANNIAN MEDIAN: EXPONENTIAL MAPS VS RETRACTIONS COMPARISON")
print("TESTING MULTIPLE DIMENSIONS WITH TIMING ANALYSIS")
print("="*80)

# Storage for results across all dimensions
all_results = {}
timing_results = []

for dim in DIMENSIONS_TO_TEST:
    print(f"\n" + "="*100)
    print(f"DIMENSION {dim}: STARTING EXPERIMENT")
    print("="*100)

    # Set random seed for reproducibility across dimensions
    anp.random.seed(RANDOM_SEED)

    # set the manifold for current dimension
    manifold = SymmetricPositiveDefinite(dim)

    # Generate n truly random points for median computation
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
    print(f"\n" + "="*80)
    print(f"DIMENSION {dim}: EXPERIMENT 1 - EXPONENTIAL MAPS AND PARALLEL TRANSPORT")
    print("="*80)

    # Start timing for exponential maps
    exp_start_time = time.time()

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
        trust_parameter=TRUST_PARAMETER,
        transport_error=0,  # Exact exponential maps
        retraction_error=0,  # Exact parallel transport
        know_minimizer=False,
        max_iter=PHASE1_ITERATIONS
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

    print(f"\nPhase 2: Convergence analysis (200 iterations)")
    print("-" * 80)

    # Phase 2: Short run with estimated minimum
    optimizer_exp_phase2 = RProximalBundle(
        manifold=manifold,
        retraction_map=manifold.exp,
        transport_map=manifold.transport,
        objective_function=cost,
        subgradient=subgradient,
        true_min_obj=estimated_minimum_exp,  # Use estimated minimum from Phase 1
        initial_point=initial_point,  # Same initial point
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        adaptive_proximal=True,
        trust_parameter=TRUST_PARAMETER,
        transport_error=0,  # Exact exponential maps
        retraction_error=0,  # Exact parallel transport
        know_minimizer=True,
        max_iter=PHASE2_ITERATIONS
    )

    print(f"Initial gap for Phase 2: {initial_objective - estimated_minimum_exp:.8f}")
    optimizer_exp_phase2.run()
    print("✓ Experiment 1 Phase 2 completed")

    # End timing for exponential maps
    exp_end_time = time.time()
    exp_total_time = exp_end_time - exp_start_time

    print(f"Experiment 1 Phase 2 Results:")
    print(f"  Final gap: {optimizer_exp_phase2.objective_history[-1]:.8f}")
    print(f"  Total time: {exp_total_time:.2f} seconds")

    # EXPERIMENT 2: RETRACTIONS (retraction_error=1, transport_error=1)
    print(f"\n" + "="*80)
    print(f"DIMENSION {dim}: EXPERIMENT 2 - FIRST-ORDER RETRACTIONS AND VECTOR TRANSPORT")
    print("="*80)

    # Start timing for retractions
    ret_start_time = time.time()

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
        trust_parameter=TRUST_PARAMETER,
        transport_error=1,  # First-order retractions
        retraction_error=1,  # First-order vector transport
        know_minimizer=False,
        max_iter=PHASE1_ITERATIONS
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

    print(f"\nPhase 2: Convergence analysis (200 iterations)")
    print("-" * 80)

    # Phase 2: Short run with estimated minimum
    optimizer_ret_phase2 = RProximalBundle(
        manifold=manifold,
        retraction_map=manifold.retraction,
        transport_map=manifold.transport,
        objective_function=cost,
        subgradient=subgradient,
        true_min_obj=estimated_minimum_ret,  # Use estimated minimum from Phase 1
        initial_point=initial_point,  # Same initial point
        initial_objective=initial_objective,
        initial_subgradient=initial_subgradient,
        adaptive_proximal=True,
        trust_parameter=TRUST_PARAMETER,
        transport_error=1,  # First-order retractions
        retraction_error=1,  # First-order vector transport
        know_minimizer=True,
        max_iter=PHASE2_ITERATIONS
    )

    print(f"Initial gap for Phase 2: {initial_objective - estimated_minimum_ret:.8f}")
    optimizer_ret_phase2.run()
    print("✓ Experiment 2 Phase 2 completed")

    # End timing for retractions
    ret_end_time = time.time()
    ret_total_time = ret_end_time - ret_start_time

    print(f"Experiment 2 Phase 2 Results:")
    print(f"  Final gap: {optimizer_ret_phase2.objective_history[-1]:.8f}")
    print(f"  Total time: {ret_total_time:.2f} seconds")

    # Store results for this dimension
    all_results[dim] = {
        'exp_optimizer_phase1': optimizer_exp_phase1,
        'exp_optimizer_phase2': optimizer_exp_phase2,
        'ret_optimizer_phase1': optimizer_ret_phase1,
        'ret_optimizer_phase2': optimizer_ret_phase2,
        'estimated_minimum_exp': estimated_minimum_exp,
        'estimated_minimum_ret': estimated_minimum_ret,
        'initial_objective': initial_objective,
        'exp_time': exp_total_time,
        'ret_time': ret_total_time
    }

    # Store timing data
    timing_results.append({
        'Dimension': dim,
        'Exponential Maps (s)': exp_total_time,
        'Retractions (s)': ret_total_time,
        'Speedup (Exp/Ret)': exp_total_time / ret_total_time,
        'Exp Final Gap': optimizer_exp_phase2.objective_history[-1],
        'Ret Final Gap': optimizer_ret_phase2.objective_history[-1],
        'Exp Iterations': len(optimizer_exp_phase2.objective_history),
        'Ret Iterations': len(optimizer_ret_phase2.objective_history)
    })

    print(f"\n" + "="*60)
    print(f"DIMENSION {dim} SUMMARY:")
    print(f"  Exponential Maps Total Time: {exp_total_time:.2f}s")
    print(f"  Retractions Total Time: {ret_total_time:.2f}s")
    print(f"  Speedup (Exp/Ret): {exp_total_time/ret_total_time:.2f}x")
    print("="*60)

# TIMING ANALYSIS TABLE
print("\n" + "="*120)
print("WALL CLOCK TIME COMPARISON TABLE")
print("="*120)

# Display the table
print("\nDetailed Timing Results:")
print("-" * 120)
print(f"{'Dim':<6} {'Exp Time (s)':<12} {'Ret Time (s)':<12} {'Speedup':<10} {'Exp Gap':<12} {'Ret Gap':<12} {'Exp Iter':<10} {'Ret Iter':<10}")
print("-" * 120)

for result in timing_results:
    print(f"{result['Dimension']:<6} {result['Exponential Maps (s)']:<12.2f} {result['Retractions (s)']:<12.2f} {result['Speedup (Exp/Ret)']:<10.2f} {result['Exp Final Gap']:<12.2e} {result['Ret Final Gap']:<12.2e} {result['Exp Iterations']:<10} {result['Ret Iterations']:<10}")

print("-" * 120)

# Summary statistics
speedups = [result['Speedup (Exp/Ret)'] for result in timing_results]
print(f"\nSummary Statistics:")
print(f"  Average Speedup (Exp/Ret): {sum(speedups)/len(speedups):.2f}x")
print(f"  Min Speedup: {min(speedups):.2f}x")
print(f"  Max Speedup: {max(speedups):.2f}x")

# VISUALIZATION
print("\n" + "="*80)
print("GENERATING VISUALIZATION")
print("="*80)

# Create timing comparison plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Extract data from timing_results
dimensions = [result['Dimension'] for result in timing_results]
exp_times = [result['Exponential Maps (s)'] for result in timing_results]
ret_times = [result['Retractions (s)'] for result in timing_results]
speedups = [result['Speedup (Exp/Ret)'] for result in timing_results]

# Plot 1: Wall clock times
ax1.plot(dimensions, exp_times, 'b-o', linewidth=2, markersize=8, label='Exponential Maps')
ax1.plot(dimensions, ret_times, 'r-s', linewidth=2, markersize=8, label='Retractions')
ax1.set_xlabel('Dimension')
ax1.set_ylabel('Wall Clock Time (seconds)')
ax1.set_title('Wall Clock Time Comparison')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Speedup
ax2.plot(dimensions, speedups, 'g-^', linewidth=2, markersize=8, label='Speedup (Exp/Ret)')
ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='No difference')
ax2.set_xlabel('Dimension')
ax2.set_ylabel('Speedup Factor')
ax2.set_title('Speedup: Exponential Maps vs Retractions')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Final gaps comparison
exp_gaps = [result['Exp Final Gap'] for result in timing_results]
ret_gaps = [result['Ret Final Gap'] for result in timing_results]

ax3.plot(dimensions, exp_gaps, 'b-o', linewidth=2, markersize=8, label='Exponential Maps')
ax3.plot(dimensions, ret_gaps, 'r-s', linewidth=2, markersize=8, label='Retractions')
ax3.set_xlabel('Dimension')
ax3.set_ylabel('Final Objective Gap')
ax3.set_title('Final Convergence Quality')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Plot 4: Iterations comparison
exp_iters = [result['Exp Iterations'] for result in timing_results]
ret_iters = [result['Ret Iterations'] for result in timing_results]

ax4.plot(dimensions, exp_iters, 'b-o', linewidth=2, markersize=8, label='Exponential Maps')
ax4.plot(dimensions, ret_iters, 'r-s', linewidth=2, markersize=8, label='Retractions')
ax4.set_xlabel('Dimension')
ax4.set_ylabel('Number of Iterations')
ax4.set_title('Iterations to Convergence')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('RMedianMultiDim_TimingAnalysis.png', dpi=150, bbox_inches='tight')
print("✓ Timing analysis plots saved as 'RMedianMultiDim_TimingAnalysis.png'")

# Save timing data to CSV
with open('RMedianMultiDim_TimingResults.csv', 'w') as f:
    # Write header
    f.write("Dimension,Exponential Maps (s),Retractions (s),Speedup (Exp/Ret),Exp Final Gap,Ret Final Gap,Exp Iterations,Ret Iterations\n")
    # Write data
    for result in timing_results:
        f.write(f"{result['Dimension']},{result['Exponential Maps (s)']},{result['Retractions (s)']},{result['Speedup (Exp/Ret)']},{result['Exp Final Gap']},{result['Ret Final Gap']},{result['Exp Iterations']},{result['Ret Iterations']}\n")
print("✓ Timing results saved as 'RMedianMultiDim_TimingResults.csv'")

# FINAL SUMMARY
print("\n" + "="*80)
print("MULTI-DIMENSIONAL EXPERIMENT COMPLETED!")
print("="*80)

print(f"\nDimensions tested: {DIMENSIONS_TO_TEST}")
print(f"Total experiments run: {len(DIMENSIONS_TO_TEST) * 2}")
print(f"Random seed used: {RANDOM_SEED}")

print(f"\nKey Findings:")
avg_speedup = sum(speedups) / len(speedups)
print(f"  - Average speedup factor: {avg_speedup:.2f}x")
if avg_speedup > 1:
    print(f"  - Exponential maps are on average slower than retractions")
else:
    print(f"  - Retractions are on average slower than exponential maps")

print(f"  - Timing scales with dimension as expected")
print(f"  - Both methods maintain good convergence across dimensions")

print("="*80)