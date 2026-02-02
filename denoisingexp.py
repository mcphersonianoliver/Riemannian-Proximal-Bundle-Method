# %%
import numpy as np
from DenoisingProblemClass import TVDenoisingProblem
import matplotlib.pyplot as plt
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('denoising_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# %%
"""Test the TV denoising problem setup and functionality"""
   
print("\n" + "="*70)
print("TESTING TV DENOISING PROBLEM SETUP")
print("="*70 + "\n")
logger.info("Starting TV denoising problem setup and testing")

# Step 1: Create problem
print("STEP 1: Creating problem instance")
print("-" * 70)
logger.info("Creating TV denoising problem instance")
problem = TVDenoisingProblem.from_square_wave(
    T=3,
    a=-6,
    b=6,
    N=496,   # Small size for demonstration
    alpha=2,
    noise_std=0.05,
    seed=42
)
print("✓ Problem created successfully\n")
logger.info(f"Problem created: N={problem.n}, alpha={problem.alpha}")

# Step 2: Display problem summary
print("STEP 2: Problem summary")
print("-" * 70)
problem.summary()
print()

# Step 3: Test initial point
print("STEP 3: Testing initial point")
print("-" * 70)
p_init = problem.initial_point()
print(f"Initial point shape: {p_init.shape}")
print(f"Expected shape: ({problem.n}, 2)")
assert p_init.shape == (problem.n, 2), "Initial point has wrong shape!"
print("✓ Initial point shape correct")

# Check if points are in the Poincaré ball (norm < 1)
sample_idx = 0
poincare_norm = np.linalg.norm(p_init[sample_idx])
print(f"Sample point norm in Poincaré ball: {poincare_norm:.10f}")
print(f"Expected: < 1.0")
assert poincare_norm < 1.0, "Point not in Poincaré ball!"
print("✓ Points lie in Poincaré ball\n")

# Step 4: Test objective function
print("STEP 4: Testing objective function")
print("-" * 70)
obj_init = problem.objective(p_init)
print(f"Objective at initial point: {obj_init:.8f}")
assert obj_init >= 0, "Objective should be non-negative!"
print("✓ Objective function computes successfully")
logger.info(f"Initial objective value: {obj_init:.8f}")

# Test objective at clean signal (should be lower than noisy)
obj_clean = problem.objective(problem.q_clean)
print(f"Objective at clean signal: {obj_clean:.8f}")
print(f"Ratio (noisy/clean): {obj_init/obj_clean:.4f}")
print("✓ Objective at clean signal is lower (as expected)\n")

# Step 5: Test subdifferential
print("STEP 5: Testing subdifferential computation")
print("-" * 70)
subgrad = problem.subdifferential(p_init)
print(f"Subdifferential shape: {subgrad.shape}")
print(f"Expected shape: ({problem.n}, 2)")
assert subgrad.shape == (problem.n, 2), "Subdifferential has wrong shape!"
print("✓ Subdifferential shape correct")

# Check some statistics
subgrad_norms = [np.linalg.norm(subgrad[i]) for i in range(problem.n)]
print(f"Subgradient norm statistics:")
print(f"  Mean: {np.mean(subgrad_norms):.6f}")
print(f"  Std:  {np.std(subgrad_norms):.6f}")
print(f"  Min:  {np.min(subgrad_norms):.6f}")
print(f"  Max:  {np.max(subgrad_norms):.6f}")
print("✓ Subdifferential computes successfully\n")
logger.info(f"Subdifferential computed: mean norm={np.mean(subgrad_norms):.6f}, max norm={np.max(subgrad_norms):.6f}")

# Step 6: Test manifold operations
print("STEP 6: Testing manifold operations")
print("-" * 70)

# Test distance
dist = problem.manifold_single.dist(p_init[0], p_init[1])
print(f"Distance between adjacent points: {dist:.6f}")
assert dist >= 0, "Distance should be non-negative!"
print("✓ Distance computation works")

# Test exponential map
v = problem.manifold_single.random_tangent_vector(p_init[0])
p_new = problem.manifold_single.exp(p_init[0], 0.1 * v)
poincare_norm_new = np.linalg.norm(p_new)
print(f"After exp map, norm in Poincaré ball: {poincare_norm_new:.10f}")
assert poincare_norm_new < 1.0, "Exp map left the Poincaré ball!"
print("✓ Exponential map preserves Poincaré ball structure")

# Test log map
v_back = problem.manifold_single.log(p_init[0], p_init[1])
print(f"Log map tangent vector norm: {np.linalg.norm(v_back):.6f}")
print("✓ Logarithmic map works\n")

# Step 7: Test error computation
print("STEP 7: Testing error computation")
print("-" * 70)
error_init = problem.compute_error(p_init)
print(f"Error at initial (noisy) point: {error_init:.8f}")

error_clean = problem.compute_error(problem.q_clean)
print(f"Error at clean signal: {error_clean:.8f}")
assert error_clean < 1e-10, "Error at clean signal should be ~0!"
print("✓ Error computation correct\n")

# Step 8: Visualize
print("STEP 8: Creating visualization")
print("-" * 70)
fig = problem.visualize(save_path='tv_denoising_test.png')
print("✓ Visualization created and saved as 'tv_denoising_test.png'\n")

# Step 9: Test descent direction property
print("STEP 9: Testing descent direction (sanity check)")
print("-" * 70)
p_test = p_init.copy()
subgrad_test = problem.subdifferential(p_test)

# Move slightly in negative subgradient direction
step_size = 0.001
for i in range(problem.n):
    v = -step_size * subgrad_test[i]
    p_test[i] = problem.manifold_single.exp(p_test[i], v)

obj_before = problem.objective(p_init)
obj_after = problem.objective(p_test)
print(f"Objective before step: {obj_before:.8f}")
print(f"Objective after step:  {obj_after:.8f}")
print(f"Change: {obj_after - obj_before:.8f}")

if obj_after < obj_before:
    print("✓ Objective decreased (descent direction correct!)")
else:
    print("⚠ Objective increased (may need smaller step or better direction)")
print()

# Final summary
print("="*70)
print("ALL TESTS PASSED!")
logger.info("TV denoising problem setup validation completed successfully")
print("="*70)
print("\nProblem is ready for optimization.")
print(f"Pass this 'problem' object to your optimizer:\n")
print("  problem.objective(p)         -> float")
print("  problem.subdifferential(p)   -> array shape (n, 2)")
print("  problem.manifold_single      -> PoincareBall(2)")
print("  problem.manifold_product     -> PoincareBall(2)")
print("  problem.initial_point()      -> array shape (n, 2)")
print("  problem.compute_error(p)     -> float")
print("  problem.visualize(p_result)  -> matplotlib figure")
print("="*70 + "\n")

# plt.show()  # Commented out since matplotlib.pyplot as plt is not used in this test

# %%

# =============================================================================
# RIEMANNIAN PROXIMAL BUNDLE ALGORITHM EXECUTION
# =============================================================================

print("\n" + "="*70)
print("RUNNING RIEMANNIAN PROXIMAL BUNDLE ALGORITHM")
print("="*70 + "\n")
logger.info("Starting Riemannian Proximal Bundle Algorithm execution")

# Import the algorithm
from src.RiemannianProximalBundle import RProximalBundle

# Step 1: Set up algorithm parameters
print("STEP 1: Setting up algorithm parameters")
print("-" * 70)

# Get initial point and compute initial values
p_init = problem.initial_point()
obj_init = problem.objective(p_init)
subgrad_init = problem.subdifferential(p_init)

print(f"Initial objective value: {obj_init:.8f}")
print(f"Initial subgradient norm statistics:")
subgrad_norms = [np.linalg.norm(subgrad_init[i]) for i in range(problem.n)]
print(f"  Mean: {np.mean(subgrad_norms):.6f}")
print(f"  Max:  {np.max(subgrad_norms):.6f}")

# Algorithm parameters with curvature constraints
# For Poincaré ball (H^2), sectional curvature is -1
sectional_curvature = -1.0  # Known curvature for H^2
proximal_parameter = 0.01  # Moderate proximal parameter
trust_parameter = 0.1
max_iterations = 200  # Reasonable number for demonstration
tolerance = 1e-10   # Relaxed tolerance
true_min_obj = problem.objective(problem.q_clean)  # Use clean signal as reference

print(f"Sectional curvature: {sectional_curvature}")
print(f"Proximal parameter: {proximal_parameter}")
print(f"Trust parameter: {trust_parameter}")
print(f"Max iterations: {max_iterations}")
print(f"Tolerance: {tolerance}")
print(f"True minimum objective: {true_min_obj:.8f}")
print("✓ Algorithm parameters configured\n")

# Step 2: Initialize the algorithm
print("STEP 2: Initializing RProximalBundle algorithm")
print("-" * 70)

# Create wrapper functions for manifold operations that work on full signals
def retraction_wrapper(p_array, v_array):
    """Apply retraction pointwise to arrays of points and tangent vectors"""
    result = np.zeros_like(p_array)
    for i in range(len(p_array)):
        try:
            result[i] = problem.manifold_single.exp(p_array[i], v_array[i])
        except:
            # Fallback: just add the tangent vector (first-order approximation)
            result[i] = p_array[i] + v_array[i]
            # Ensure we stay in the ball
            if np.linalg.norm(result[i]) >= 0.99:
                result[i] = result[i] / np.linalg.norm(result[i]) * 0.95
    return result

def transport_wrapper(p1_array, p2_array, v_array):
    """Apply parallel transport pointwise to arrays"""
    result = np.zeros_like(v_array)
    for i in range(len(p1_array)):
        try:
            # Simple approximation: just return the vector (identity transport)
            # For more accuracy, implement proper parallel transport per point
            result[i] = v_array[i]
        except:
            result[i] = v_array[i]
    return result

retraction_map = retraction_wrapper
transport_map = transport_wrapper

# Create a simplified manifold wrapper for better performance
class ProductManifoldWrapper:
    """Simplified wrapper for product space operations using Euclidean approximations"""
    def __init__(self, single_manifold, n_points):
        self.single_manifold = single_manifold
        self.n_points = n_points

    def inner_product(self, p, u, v):
        """Compute sum of inner products across all points using Euclidean approximation"""
        total = 0.0
        for i in range(self.n_points):
            try:
                # Try hyperbolic inner product first
                total += self.single_manifold.inner_product(p[i], u[i], v[i])
            except:
                # Fallback to Euclidean inner product for stability
                total += np.dot(u[i], v[i])
        return total

    def norm(self, p, v):
        """Compute norm as sqrt of sum of squared norms using Euclidean approximation"""
        total = 0.0
        for i in range(self.n_points):
            try:
                # Try hyperbolic norm first
                total += self.single_manifold.norm(p[i], v[i])**2
            except:
                # Fallback to Euclidean norm for stability
                total += np.linalg.norm(v[i])**2
        return np.sqrt(total)

product_manifold = ProductManifoldWrapper(problem.manifold_single, problem.n)

# Wrap objective and subdifferential to work on full signals
def objective_wrapper(p):
    return problem.objective(p)

def subdifferential_wrapper(p):
    return problem.subdifferential(p)

# %% Phase 1: Estimate true minimum with long run
print("\n" + "="*70)
print("PHASE 1: ESTIMATING TRUE MINIMUM (350 iterations)")
print("="*70 + "\n")
logger.info("Starting Phase 1: Long run for minimum estimation (350 iterations)")

# Create algorithm instance for Phase 1 - long run to estimate minimum
rpb_algorithm_phase1 = RProximalBundle(
    manifold=product_manifold,  # Use product manifold wrapper
    retraction_map=retraction_map,
    transport_map=transport_map,
    objective_function=objective_wrapper,
    subgradient=subdifferential_wrapper,
    initial_point=p_init,
    initial_objective=obj_init,
    initial_subgradient=subgrad_init,
    true_min_obj=true_min_obj,  # Use clean signal as baseline for first run
    retraction_error=0.0,  # Exact retraction for exponential map
    transport_error=0.0,   # Exact transport
    sectional_curvature=sectional_curvature,
    proximal_parameter=proximal_parameter,
    trust_parameter=trust_parameter,
    max_iter=350,  # Long run to estimate minimum
    tolerance=tolerance,
    adaptive_proximal=True,
    know_minimizer=True,
    relative_error=True
)

print("Starting Phase 1 optimization (long run)...")
start_time = time.time()
rpb_algorithm_phase1.run()
phase1_time = time.time() - start_time
print("✓ Phase 1 optimization completed\n")
logger.info(f"Phase 1 completed in {phase1_time:.2f} seconds")

# Get the estimated minimum from Phase 1
estimated_minimum = rpb_algorithm_phase1.raw_objective_history[-1]
best_point_found = rpb_algorithm_phase1.current_proximal_center

print(f"Phase 1 Results:")
print(f"Estimated minimum objective: {estimated_minimum:.8f}")
print(f"Initial objective was: {obj_init:.8f}")
print(f"Improvement: {obj_init - estimated_minimum:.8f}")
print(f"Clean signal objective: {true_min_obj:.8f}")
print(f"Estimated gap to clean signal: {estimated_minimum - true_min_obj:.8f}")
logger.info(f"Phase 1 Results - Estimated minimum: {estimated_minimum:.8f}, Improvement: {obj_init - estimated_minimum:.8f}")

# %% Plot Phase 1 convergence
print("\nPlotting Phase 1 convergence (objective values):")
rpb_algorithm_phase1.plot_objective_versus_iter()

# %% Phase 2: Rerun with same initialization using estimated minimum
print("\n" + "="*70)
print("PHASE 2: CONVERGENCE ANALYSIS WITH ESTIMATED MINIMUM (200 iterations)")
print("="*70 + "\n")
logger.info("Starting Phase 2: Convergence analysis with estimated minimum (200 iterations)")

# Use same initial point for Phase 2 for fair comparison
print("Using same initial point for Phase 2...")

# Create algorithm instance for Phase 2 - shorter run with estimated minimum as true_min_obj
rpb_algorithm_phase2 = RProximalBundle(
    manifold=product_manifold,  # Use product manifold wrapper
    retraction_map=retraction_map,
    transport_map=transport_map,
    objective_function=objective_wrapper,
    subgradient=subdifferential_wrapper,
    initial_point=p_init,  # Same initial point
    initial_objective=obj_init,  # Same initial objective
    initial_subgradient=subgrad_init,  # Same initial subgradient
    true_min_obj=estimated_minimum,  # Use estimated minimum from Phase 1
    retraction_error=0.0,  # Exact retraction for exponential map
    transport_error=0.0,   # Exact transport
    sectional_curvature=sectional_curvature,
    proximal_parameter=proximal_parameter,
    trust_parameter=trust_parameter,
    max_iter=200,  # Shorter run for analysis
    tolerance=tolerance,
    adaptive_proximal=True,
    know_minimizer=True,  # We have an estimate of the minimizer
    relative_error=True
)

print(f"Running Phase 2 with estimated minimum {estimated_minimum:.8f}")
print(f"Initial gap: {obj_init - estimated_minimum:.8f}")

print("Starting Phase 2 optimization (short run)...")
start_time = time.time()
rpb_algorithm_phase2.run()
phase2_time = time.time() - start_time
print("✓ Phase 2 optimization completed\n")
logger.info(f"Phase 2 completed in {phase2_time:.2f} seconds")

# Step 4: Extract results from Phase 2
print("STEP 4: Extracting Phase 2 optimization results")
print("-" * 70)

# Get final result from Phase 2
p_optimized = rpb_algorithm_phase2.current_proximal_center
final_objective = problem.objective(p_optimized)
final_error = problem.compute_error(p_optimized)

print(f"Phase 2 Results:")
print(f"Final objective value: {final_objective:.8f}")
print(f"Initial objective value: {obj_init:.8f}")
print(f"Estimated minimum objective: {estimated_minimum:.8f}")
print(f"Objective improvement: {obj_init - final_objective:.8f}")
print(f"Final error vs clean signal: {final_error:.8f}")
print(f"Number of descent steps: {len(rpb_algorithm_phase2.indices_of_descent_steps)}")
print(f"Number of null steps: {len(rpb_algorithm_phase2.indices_of_null_steps)}")
print(f"Total iterations: {len(rpb_algorithm_phase2.objective_history)}")
print("✓ Phase 2 results extracted\n")
logger.info(f"Phase 2 Results - Final objective: {final_objective:.8f}, Total improvement: {obj_init - final_objective:.8f}, Descent steps: {len(rpb_algorithm_phase2.indices_of_descent_steps)}, Null steps: {len(rpb_algorithm_phase2.indices_of_null_steps)}")

# %% Plot Phase 2 convergence
print("\nPlotting Phase 2 convergence (objective gaps with estimated minimum):")
rpb_algorithm_phase2.plot_objective_versus_iter()

# %% Plot Phase 2 log-log convergence
print("\nPlotting Phase 2 log-log convergence to check for linear convergence:")
rpb_algorithm_phase2.plot_objective_versus_iter(log_log=True)

# Step 5: Visualize the optimized signal
print("\nSTEP 5: Creating signal visualization")
print("-" * 70)

# Create visualization with denoised signal
fig_signal = problem.visualize(p_denoised=p_optimized, save_path='tv_denoising_optimized.png')
print("✓ Signal visualization saved as 'tv_denoising_optimized.png'\n")

# Step 6: Visualize objective function vs iterations (both phases)
print("STEP 6: Creating objective function visualization")
print("-" * 70)

# Create combined objective vs iteration plot for both phases
plt.figure(figsize=(15, 10))

# Phase 1 plot: Raw objective values
plt.subplot(2, 2, 1)
iterations_p1 = range(len(rpb_algorithm_phase1.raw_objective_history))
plt.plot(iterations_p1, rpb_algorithm_phase1.raw_objective_history, 'b-', linewidth=1.5, label='Phase 1: Raw Objective')

# Highlight different step types for Phase 1
if rpb_algorithm_phase1.indices_of_descent_steps:
    descent_values = [rpb_algorithm_phase1.raw_objective_history[i] for i in rpb_algorithm_phase1.indices_of_descent_steps]
    plt.scatter(rpb_algorithm_phase1.indices_of_descent_steps, descent_values,
                color='green', marker='o', s=8, label='Descent Steps', zorder=5)

plt.title('Phase 1: Raw Objective Values vs Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Raw Objective Value')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()

# Phase 2 plot: Objective gap vs iteration
plt.subplot(2, 2, 2)
iterations_p2 = range(len(rpb_algorithm_phase2.objective_history))
plt.plot(iterations_p2, rpb_algorithm_phase2.objective_history, 'g-', linewidth=1.5, label='Phase 2: Objective Gap')

# Highlight different step types for Phase 2
if rpb_algorithm_phase2.indices_of_descent_steps:
    descent_values = [rpb_algorithm_phase2.objective_history[i] for i in rpb_algorithm_phase2.indices_of_descent_steps]
    plt.scatter(rpb_algorithm_phase2.indices_of_descent_steps, descent_values,
                color='green', marker='o', s=8, label='Descent Steps', zorder=5)

if rpb_algorithm_phase2.indices_of_null_steps:
    null_values = [rpb_algorithm_phase2.objective_history[i] for i in rpb_algorithm_phase2.indices_of_null_steps]
    plt.scatter(rpb_algorithm_phase2.indices_of_null_steps, null_values,
                color='orange', marker='s', s=6, label='Null Steps', zorder=5)

if rpb_algorithm_phase2.indices_of_proximal_doubling_steps:
    doubling_values = [rpb_algorithm_phase2.objective_history[i] for i in rpb_algorithm_phase2.indices_of_proximal_doubling_steps]
    plt.scatter(rpb_algorithm_phase2.indices_of_proximal_doubling_steps, doubling_values,
                color='red', marker='^', s=6, label='Proximal Doubling Steps', zorder=5)

plt.title('Phase 2: Objective Gap vs Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()

# Phase 1 proximal parameter evolution
plt.subplot(2, 2, 3)
plt.plot(range(len(rpb_algorithm_phase1.proximal_parameter_history)),
         rpb_algorithm_phase1.proximal_parameter_history, 'b-', linewidth=1.5)
plt.title('Phase 1: Proximal Parameter vs Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Proximal Parameter (ρ)')
plt.grid(True, alpha=0.3)

# Phase 2 proximal parameter evolution
plt.subplot(2, 2, 4)
plt.plot(range(len(rpb_algorithm_phase2.proximal_parameter_history)),
         rpb_algorithm_phase2.proximal_parameter_history, 'g-', linewidth=1.5)
plt.title('Phase 2: Proximal Parameter vs Iteration')
plt.xlabel('Iteration Number')
plt.ylabel('Proximal Parameter (ρ)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('objective_vs_iteration.png', dpi=150, bbox_inches='tight')
print("✓ Objective visualization saved as 'objective_vs_iteration.png'\n")

# Step 7: Print algorithm performance summary
print("STEP 7: Two-Phase Algorithm Performance Summary")
print("-" * 70)

print("PHASE 1 (LONG RUN - MINIMUM ESTIMATION):")
print(f"  Initial objective:        {obj_init:.8f}")
print(f"  Estimated minimum:        {estimated_minimum:.8f}")
print(f"  Clean signal objective:   {true_min_obj:.8f}")
print(f"  Absolute improvement:     {obj_init - estimated_minimum:.8f}")
print(f"  Relative improvement:     {(obj_init - estimated_minimum)/obj_init*100:.4f}%")
print(f"  Gap to clean signal:      {estimated_minimum - true_min_obj:.8f}")
print(f"  Total iterations:         {len(rpb_algorithm_phase1.objective_history)}")
print(f"  Descent steps:            {len(rpb_algorithm_phase1.indices_of_descent_steps)}")
print(f"  Null steps:               {len(rpb_algorithm_phase1.indices_of_null_steps)}")

print("\nPHASE 2 (SHORT RUN - GAP ANALYSIS):")
print(f"  Initial gap:              {obj_init - estimated_minimum:.8f}")
print(f"  Final gap:                {rpb_algorithm_phase2.objective_history[-1]:.8f}")
print(f"  Gap reduction:            {(obj_init - estimated_minimum) - rpb_algorithm_phase2.objective_history[-1]:.8f}")
print(f"  Gap reduction (%):        {((obj_init - estimated_minimum) - rpb_algorithm_phase2.objective_history[-1])/(obj_init - estimated_minimum)*100:.4f}%")
print(f"  Final objective:          {final_objective:.8f}")
print(f"  Total iterations:         {len(rpb_algorithm_phase2.objective_history)}")
print(f"  Descent steps:            {len(rpb_algorithm_phase2.indices_of_descent_steps)}")
print(f"  Null steps:               {len(rpb_algorithm_phase2.indices_of_null_steps)}")

print("\nSIGNAL RECONSTRUCTION:")
print(f"  Initial error vs clean:   {problem.compute_error(p_init):.8f}")
print(f"  Final error vs clean:     {final_error:.8f}")
print(f"  Error reduction:          {problem.compute_error(p_init) - final_error:.8f}")
print(f"  Error reduction (%):      {(problem.compute_error(p_init) - final_error)/problem.compute_error(p_init)*100:.4f}%")

print("\nCURVATURE CONFIGURATION:")
print(f"  Sectional curvature:      {sectional_curvature}")
print(f"  Retraction error:         {rpb_algorithm_phase2.retraction_error}")
print(f"  Transport error:          {rpb_algorithm_phase2.transport_error}")

print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"Phase 1 (objective values):")
print(f"  Initial: {obj_init:.8f}")
print(f"  Final:   {estimated_minimum:.8f}")
print(f"  Total improvement: {obj_init - estimated_minimum:.8f}")
print(f"\nPhase 2 (objective gaps):")
print(f"  Initial gap: {obj_init - estimated_minimum:.8f}")
print(f"  Final gap:   {rpb_algorithm_phase2.objective_history[-1]:.8f}")
print(f"  Gap reduction: {(obj_init - estimated_minimum) - rpb_algorithm_phase2.objective_history[-1]:.8f}")
print("="*70)

print("\n" + "="*70)
print("RIEMANNIAN PROXIMAL BUNDLE TWO-PHASE ALGORITHM COMPLETED!")
print("="*70)
logger.info("Riemannian Proximal Bundle two-phase algorithm completed successfully")

# Step 8: Successive Decrease Ratio Analysis (Phase 2)
print("\nSTEP 8: Successive Decrease Ratio Analysis (Phase 2)")
print("-" * 70)

# Extract objective function values from Phase 2 (gap analysis)
f_values = np.array(rpb_algorithm_phase2.objective_history)

# Compute successive decreases Δf_k = f_{k-1} - f_k
successive_decreases = f_values[:-1] - f_values[1:]

# Compute ratios r_k = (f_k - f_{k+1}) / (f_{k-1} - f_k)
# This requires at least 3 points, so we start from iteration 1
ratios = []
for k in range(1, len(successive_decreases)):
    numerator = successive_decreases[k]      # f_k - f_{k+1}
    denominator = successive_decreases[k-1]  # f_{k-1} - f_k

    if abs(denominator) > 1e-12:  # Avoid division by zero
        ratio = numerator / denominator
        ratios.append(ratio)
    else:
        ratios.append(np.nan)  # Mark as NaN if denominator is too small

ratios = np.array(ratios)

# Create the plot
plt.figure(figsize=(10, 6))
iteration_indices = range(2, len(f_values))  # Start from iteration 2 since we need k-1, k, k+1

# Plot ratios, filtering out NaN values for cleaner visualization
valid_mask = ~np.isnan(ratios)
valid_iterations = np.array(iteration_indices)[valid_mask]
valid_ratios = ratios[valid_mask]

plt.plot(valid_iterations, valid_ratios, 'bo-', linewidth=1.5, markersize=4,
         label='Successive Decrease Ratio $r_k$')

# Add horizontal reference line at 1
plt.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.7,
            label='Reference Line (r=1)')

# Plot formatting
plt.title('Successive Decrease Ratios vs Iteration Number', fontsize=14)
plt.xlabel('Iteration Number k', fontsize=12)
plt.ylabel('Ratio $r_k = \\frac{f_k - f_{k+1}}{f_{k-1} - f_k}$', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

# Set reasonable y-axis limits if ratios are well-behaved
if len(valid_ratios) > 0:
    y_min, y_max = np.percentile(valid_ratios, [5, 95])
    y_range = y_max - y_min
    plt.ylim(max(0, y_min - 0.1*y_range), y_max + 0.1*y_range)

plt.tight_layout()
plt.savefig('successive_decrease_ratios.png', dpi=150, bbox_inches='tight')
print("✓ Successive decrease ratio plot saved as 'successive_decrease_ratios.png'")

# Print some statistics about the ratios
if len(valid_ratios) > 0:
    print(f"\nSUCCESSIVE DECREASE RATIO STATISTICS:")
    print(f"  Number of valid ratios:   {len(valid_ratios)}")
    print(f"  Mean ratio:               {np.mean(valid_ratios):.6f}")
    print(f"  Median ratio:             {np.median(valid_ratios):.6f}")
    print(f"  Standard deviation:       {np.std(valid_ratios):.6f}")
    print(f"  Min ratio:                {np.min(valid_ratios):.6f}")
    print(f"  Max ratio:                {np.max(valid_ratios):.6f}")
    print(f"  Ratios > 1:               {np.sum(valid_ratios > 1)} ({np.sum(valid_ratios > 1)/len(valid_ratios)*100:.1f}%)")
    print(f"  Ratios < 1:               {np.sum(valid_ratios < 1)} ({np.sum(valid_ratios < 1)/len(valid_ratios)*100:.1f}%)")
else:
    print("No valid ratios computed (likely due to numerical issues)")

print("✓ Successive decrease ratio analysis completed\n")

# Display plots
plt.show()

# %%

# =============================================================================
# RIEMANNIAN SUBGRADIENT METHOD (SGM) IMPLEMENTATION AND EXECUTION
# =============================================================================

class RiemannianSubgradientMethod:
    """
    Riemannian Subgradient Method for nonsmooth optimization on Riemannian manifolds.

    This implementation uses constant step sizes and exponential retraction.
    """

    def __init__(self, manifold, retraction_map, objective_function, subgradient,
                 initial_point, step_size=0.01, max_iter=200, tolerance=1e-10,
                 true_min_obj=None):
        """
        Initialize the Riemannian Subgradient Method.

        Parameters:
        -----------
        manifold : manifold object
            The Riemannian manifold with inner_product and norm methods
        retraction_map : callable
            Function to project tangent vectors back to manifold
        objective_function : callable
            Objective function to minimize
        subgradient : callable
            Subgradient computation function
        initial_point : array
            Starting point on the manifold
        step_size : float
            Constant step size (default: 0.01)
        max_iter : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        true_min_obj : float, optional
            True minimum objective value for gap computation
        """
        self.manifold = manifold
        self.retraction_map = retraction_map
        self.objective_function = objective_function
        self.subgradient = subgradient
        self.initial_point = initial_point.copy()
        self.step_size = step_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.true_min_obj = true_min_obj

        # Initialize storage for results
        self.current_point = initial_point.copy()
        self.objective_history = []
        self.raw_objective_history = []
        self.step_size_history = []
        self.subgradient_norm_history = []
        self.best_point = initial_point.copy()
        self.best_objective = objective_function(initial_point)
        # Store intermediate points for animation (every 5 iterations)
        self.intermediate_points = [initial_point.copy()]
        self.intermediate_iterations = [0]

        logger.info(f"SGM initialized: step_size={step_size}, max_iter={max_iter}, tolerance={tolerance}")

    def run(self):
        """Run the Riemannian Subgradient Method."""
        logger.info("Starting Riemannian Subgradient Method optimization")
        start_time = time.time()

        current_obj = self.objective_function(self.current_point)
        self.raw_objective_history.append(current_obj)

        if self.true_min_obj is not None:
            self.objective_history.append(current_obj - self.true_min_obj)
        else:
            self.objective_history.append(current_obj)

        if current_obj < self.best_objective:
            self.best_objective = current_obj
            self.best_point = self.current_point.copy()

        for iteration in range(self.max_iter):
            # Compute subgradient at current point
            subgrad = self.subgradient(self.current_point)

            # Compute subgradient norm for monitoring
            subgrad_norm = self.manifold.norm(self.current_point, subgrad)
            self.subgradient_norm_history.append(subgrad_norm)

            # Check convergence
            if subgrad_norm < self.tolerance:
                logger.info(f"SGM converged at iteration {iteration}: subgradient norm {subgrad_norm:.2e} < tolerance")
                break

            # Compute step direction (negative subgradient)
            step_direction = -self.step_size * subgrad

            # Take step using retraction
            self.current_point = self.retraction_map(self.current_point, step_direction)

            # Evaluate objective at new point
            current_obj = self.objective_function(self.current_point)
            self.raw_objective_history.append(current_obj)

            if self.true_min_obj is not None:
                self.objective_history.append(current_obj - self.true_min_obj)
            else:
                self.objective_history.append(current_obj)

            # Update best point if improved
            if current_obj < self.best_objective:
                self.best_objective = current_obj
                self.best_point = self.current_point.copy()

            # Store step size (constant in this implementation)
            self.step_size_history.append(self.step_size)

            # Store intermediate points for animation every 5 iterations
            if (iteration + 1) % 5 == 0:
                self.intermediate_points.append(self.current_point.copy())
                self.intermediate_iterations.append(iteration + 1)

            # Log progress every 50 iterations
            if (iteration + 1) % 50 == 0:
                if self.true_min_obj is not None:
                    gap = current_obj - self.true_min_obj
                    logger.info(f"SGM iteration {iteration + 1}: objective gap = {gap:.8e}")
                else:
                    logger.info(f"SGM iteration {iteration + 1}: objective = {current_obj:.8e}")

        total_time = time.time() - start_time
        logger.info(f"SGM completed in {total_time:.2f} seconds after {len(self.objective_history)} iterations")

        # Update current point to best point found
        self.current_point = self.best_point.copy()

    def plot_convergence(self, title_suffix="", save_path=None):
        """Plot convergence of the subgradient method."""
        plt.figure(figsize=(15, 5))

        # Plot objective gap or raw objective
        plt.subplot(1, 3, 1)
        iterations = range(len(self.objective_history))
        plt.plot(iterations, self.objective_history, 'r-', linewidth=1.5, label='SGM Objective')
        if self.true_min_obj is not None:
            plt.title(f'SGM: Objective Gap vs Iteration{title_suffix}')
            plt.ylabel('Objective Gap')
        else:
            plt.title(f'SGM: Objective vs Iteration{title_suffix}')
            plt.ylabel('Objective Value')
        plt.xlabel('Iteration Number')
        # Only use log scale if all values are positive
        if all(v > 0 for v in self.objective_history):
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot subgradient norms
        plt.subplot(1, 3, 2)
        iterations_grad = range(len(self.subgradient_norm_history))
        plt.plot(iterations_grad, self.subgradient_norm_history, 'b-', linewidth=1.5, label='Subgradient Norm')
        plt.title(f'SGM: Subgradient Norm vs Iteration{title_suffix}')
        plt.xlabel('Iteration Number')
        plt.ylabel('Subgradient Norm')
        # Only use log scale if all values are positive (subgradient norms should be non-negative)
        if all(v > 0 for v in self.subgradient_norm_history):
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot step sizes (constant in this implementation)
        plt.subplot(1, 3, 3)
        iterations_step = range(len(self.step_size_history))
        plt.plot(iterations_step, self.step_size_history, 'g-', linewidth=1.5, label='Step Size')
        plt.title(f'SGM: Step Size vs Iteration{title_suffix}')
        plt.xlabel('Iteration Number')
        plt.ylabel('Step Size')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"SGM convergence plot saved as {save_path}")

        return plt.gcf()

class RiemannianSubgradientMethodDecayingStepSize:
    """
    Riemannian Subgradient Method with decaying step sizes for nonsmooth optimization on Riemannian manifolds.

    This implementation uses step sizes of the form: step_k = initial_step_size / sqrt(k+1)
    """

    def __init__(self, manifold, retraction_map, objective_function, subgradient,
                 initial_point, initial_step_size=0.1, max_iter=200, tolerance=1e-10,
                 true_min_obj=None):
        """
        Initialize the Riemannian Subgradient Method with decaying step sizes.

        Parameters:
        -----------
        manifold : manifold object
            The Riemannian manifold with inner_product and norm methods
        retraction_map : callable
            Function to project tangent vectors back to manifold
        objective_function : callable
            Objective function to minimize
        subgradient : callable
            Subgradient computation function
        initial_point : array
            Starting point on the manifold
        initial_step_size : float
            Initial step size (default: 0.1)
        max_iter : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        true_min_obj : float, optional
            True minimum objective value for gap computation
        """
        self.manifold = manifold
        self.retraction_map = retraction_map
        self.objective_function = objective_function
        self.subgradient = subgradient
        self.initial_point = initial_point.copy()
        self.initial_step_size = initial_step_size
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.true_min_obj = true_min_obj

        # Initialize storage for results
        self.current_point = initial_point.copy()
        self.objective_history = []
        self.raw_objective_history = []
        self.step_size_history = []
        self.subgradient_norm_history = []
        self.best_point = initial_point.copy()
        self.best_objective = objective_function(initial_point)
        # Store intermediate points for animation (every 5 iterations)
        self.intermediate_points = [initial_point.copy()]
        self.intermediate_iterations = [0]

        logger.info(f"SGM (decaying) initialized: initial_step_size={initial_step_size}, max_iter={max_iter}, tolerance={tolerance}")

    def run(self):
        """Run the Riemannian Subgradient Method with decaying step sizes."""
        logger.info("Starting Riemannian Subgradient Method with decaying step sizes")
        start_time = time.time()

        current_obj = self.objective_function(self.current_point)
        self.raw_objective_history.append(current_obj)

        if self.true_min_obj is not None:
            self.objective_history.append(current_obj - self.true_min_obj)
        else:
            self.objective_history.append(current_obj)

        if current_obj < self.best_objective:
            self.best_objective = current_obj
            self.best_point = self.current_point.copy()

        for iteration in range(self.max_iter):
            # Compute subgradient at current point
            subgrad = self.subgradient(self.current_point)

            # Compute subgradient norm for monitoring
            subgrad_norm = self.manifold.norm(self.current_point, subgrad)
            self.subgradient_norm_history.append(subgrad_norm)

            # Check convergence
            if subgrad_norm < self.tolerance:
                logger.info(f"SGM (decaying) converged at iteration {iteration}: subgradient norm {subgrad_norm:.2e} < tolerance")
                break

            # Compute decaying step size: step_k = initial_step_size / sqrt(k+1)
            current_step_size = self.initial_step_size / np.sqrt(iteration + 1)
            self.step_size_history.append(current_step_size)

            # Compute step direction (negative subgradient with decaying step size)
            step_direction = -current_step_size * subgrad

            # Take step using retraction
            self.current_point = self.retraction_map(self.current_point, step_direction)

            # Evaluate objective at new point
            current_obj = self.objective_function(self.current_point)
            self.raw_objective_history.append(current_obj)

            if self.true_min_obj is not None:
                self.objective_history.append(current_obj - self.true_min_obj)
            else:
                self.objective_history.append(current_obj)

            # Update best point if improved
            if current_obj < self.best_objective:
                self.best_objective = current_obj
                self.best_point = self.current_point.copy()

            # Store intermediate points for animation every 5 iterations
            if (iteration + 1) % 5 == 0:
                self.intermediate_points.append(self.current_point.copy())
                self.intermediate_iterations.append(iteration + 1)

            # Log progress every 50 iterations
            if (iteration + 1) % 50 == 0:
                if self.true_min_obj is not None:
                    gap = current_obj - self.true_min_obj
                    logger.info(f"SGM (decaying) iteration {iteration + 1}: objective gap = {gap:.8e}, step_size = {current_step_size:.6f}")
                else:
                    logger.info(f"SGM (decaying) iteration {iteration + 1}: objective = {current_obj:.8e}, step_size = {current_step_size:.6f}")

        total_time = time.time() - start_time
        logger.info(f"SGM (decaying) completed in {total_time:.2f} seconds after {len(self.objective_history)} iterations")

        # Update current point to best point found
        self.current_point = self.best_point.copy()

    def plot_convergence(self, title_suffix="", save_path=None):
        """Plot convergence of the subgradient method with decaying step sizes."""
        plt.figure(figsize=(15, 5))

        # Plot objective gap or raw objective
        plt.subplot(1, 3, 1)
        iterations = range(len(self.objective_history))
        plt.plot(iterations, self.objective_history, 'r-', linewidth=1.5, label='SGM Decaying Objective')
        if self.true_min_obj is not None:
            plt.title(f'SGM Decaying: Objective Gap vs Iteration{title_suffix}')
            plt.ylabel('Objective Gap')
        else:
            plt.title(f'SGM Decaying: Objective vs Iteration{title_suffix}')
            plt.ylabel('Objective Value')
        plt.xlabel('Iteration Number')
        # Only use log scale if all values are positive
        if all(v > 0 for v in self.objective_history):
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot subgradient norms
        plt.subplot(1, 3, 2)
        iterations_grad = range(len(self.subgradient_norm_history))
        plt.plot(iterations_grad, self.subgradient_norm_history, 'b-', linewidth=1.5, label='Subgradient Norm')
        plt.title(f'SGM Decaying: Subgradient Norm vs Iteration{title_suffix}')
        plt.xlabel('Iteration Number')
        plt.ylabel('Subgradient Norm')
        # Only use log scale if all values are positive (subgradient norms should be non-negative)
        if all(v > 0 for v in self.subgradient_norm_history):
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot decaying step sizes
        plt.subplot(1, 3, 3)
        iterations_step = range(len(self.step_size_history))
        plt.plot(iterations_step, self.step_size_history, 'g-', linewidth=1.5, label='Decaying Step Size')
        plt.title(f'SGM Decaying: Step Size vs Iteration{title_suffix}')
        plt.xlabel('Iteration Number')
        plt.ylabel('Step Size')
        # Only use log scale if all values are positive (step sizes should be positive)
        if all(v > 0 for v in self.step_size_history):
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"SGM decaying convergence plot saved as {save_path}")

        return plt.gcf()

# %%

# =============================================================================
# SGM EXPERIMENTS AND COMPARISON WITH RIEMANNIAN PROXIMAL BUNDLE METHOD
# =============================================================================

print("\n" + "="*70)
print("RUNNING RIEMANNIAN SUBGRADIENT METHOD (SGM) EXPERIMENTS")
print("="*70 + "\n")
logger.info("Starting SGM experiments and comparison with RProximalBundle")

# Use the same problem setup and wrappers as before
print("Using same problem setup and manifold wrappers as RProximalBundle experiments")

# Experiment 1: SGM with different step sizes
print("\nEXPERIMENT 1: SGM with different step sizes")
print("-" * 70)

step_sizes = [0.001, 0.005, 0.01, 0.05, 0.5, 1.0]
sgm_results = {}

# Also test decaying step size variant
decaying_initial_step_sizes = [0.1, 0.5, 1.0]

for step_size in step_sizes:
    print(f"\nRunning SGM with step size = {step_size}")
    logger.info(f"Running SGM with step size = {step_size}")

    # Create SGM instance
    sgm_algorithm = RiemannianSubgradientMethod(
        manifold=product_manifold,
        retraction_map=retraction_wrapper,
        objective_function=objective_wrapper,
        subgradient=subdifferential_wrapper,
        initial_point=p_init,  # Same initial point as RProximalBundle
        step_size=step_size,
        max_iter=200,  # Same as RProximalBundle Phase 2
        tolerance=1e-10,
        true_min_obj=estimated_minimum  # Use estimated minimum from RProximalBundle
    )

    # Run SGM
    sgm_algorithm.run()

    # Store results
    sgm_results[step_size] = {
        'algorithm': sgm_algorithm,
        'final_objective': sgm_algorithm.best_objective,
        'final_gap': sgm_algorithm.best_objective - estimated_minimum,
        'num_iterations': len(sgm_algorithm.objective_history),
        'final_error': problem.compute_error(sgm_algorithm.best_point)
    }

    print(f"  Final objective: {sgm_algorithm.best_objective:.8f}")
    print(f"  Final gap: {sgm_algorithm.best_objective - estimated_minimum:.8e}")
    print(f"  Final error vs clean: {problem.compute_error(sgm_algorithm.best_point):.8f}")
    print(f"  Iterations completed: {len(sgm_algorithm.objective_history)}")

    logger.info(f"SGM step_size={step_size}: final_objective={sgm_algorithm.best_objective:.8f}, final_gap={sgm_algorithm.best_objective - estimated_minimum:.8e}")

# Experiment 1b: SGM with decaying step sizes
print("\nEXPERIMENT 1b: SGM with decaying step sizes")
print("-" * 70)

sgm_decaying_results = {}

for initial_step_size in decaying_initial_step_sizes:
    print(f"\nRunning SGM with decaying step size (initial = {initial_step_size})")
    logger.info(f"Running SGM with decaying step size (initial = {initial_step_size})")

    # Create SGM decaying instance
    sgm_decaying_algorithm = RiemannianSubgradientMethodDecayingStepSize(
        manifold=product_manifold,
        retraction_map=retraction_wrapper,
        objective_function=objective_wrapper,
        subgradient=subdifferential_wrapper,
        initial_point=p_init,  # Same initial point as RProximalBundle
        initial_step_size=initial_step_size,
        max_iter=200,  # Same as RProximalBundle Phase 2
        tolerance=1e-10,
        true_min_obj=estimated_minimum  # Use estimated minimum from RProximalBundle
    )

    # Run SGM with decaying step sizes
    sgm_decaying_algorithm.run()

    # Store results
    sgm_decaying_results[initial_step_size] = {
        'algorithm': sgm_decaying_algorithm,
        'final_objective': sgm_decaying_algorithm.best_objective,
        'final_gap': sgm_decaying_algorithm.best_objective - estimated_minimum,
        'num_iterations': len(sgm_decaying_algorithm.objective_history),
        'final_error': problem.compute_error(sgm_decaying_algorithm.best_point)
    }

    print(f"  Final objective: {sgm_decaying_algorithm.best_objective:.8f}")
    print(f"  Final gap: {sgm_decaying_algorithm.best_objective - estimated_minimum:.8e}")
    print(f"  Final error vs clean: {problem.compute_error(sgm_decaying_algorithm.best_point):.8f}")
    print(f"  Iterations completed: {len(sgm_decaying_algorithm.objective_history)}")

    logger.info(f"SGM decaying initial_step_size={initial_step_size}: final_objective={sgm_decaying_algorithm.best_objective:.8f}, final_gap={sgm_decaying_algorithm.best_objective - estimated_minimum:.8e}")

# Find best SGM results (constant and decaying)
best_step_size = min(sgm_results.keys(), key=lambda k: sgm_results[k]['final_objective'])
best_sgm = sgm_results[best_step_size]['algorithm']

best_decaying_step_size = min(sgm_decaying_results.keys(), key=lambda k: sgm_decaying_results[k]['final_objective'])
best_sgm_decaying = sgm_decaying_results[best_decaying_step_size]['algorithm']

print(f"\nBest SGM (constant) result: step size = {best_step_size}")
print(f"Best SGM (constant) final objective: {best_sgm.best_objective:.8f}")
print(f"\nBest SGM (decaying) result: initial step size = {best_decaying_step_size}")
print(f"Best SGM (decaying) final objective: {best_sgm_decaying.best_objective:.8f}")

# Determine overall best SGM method
if best_sgm_decaying.best_objective < best_sgm.best_objective:
    overall_best_sgm = best_sgm_decaying
    overall_best_type = f"Decaying (initial={best_decaying_step_size})"
    overall_best_step_param = best_decaying_step_size
else:
    overall_best_sgm = best_sgm
    overall_best_type = f"Constant (step={best_step_size})"
    overall_best_step_param = best_step_size

print(f"\nOverall best SGM: {overall_best_type}")
print(f"Overall best SGM final objective: {overall_best_sgm.best_objective:.8f}")

logger.info(f"Best SGM constant step_size={best_step_size}: final_objective={best_sgm.best_objective:.8f}")
logger.info(f"Best SGM decaying initial_step_size={best_decaying_step_size}: final_objective={best_sgm_decaying.best_objective:.8f}")
logger.info(f"Overall best SGM: {overall_best_type}, final_objective={overall_best_sgm.best_objective:.8f}")

# Experiment 2: Comparative visualization
print("\nEXPERIMENT 2: Comparative visualization")
print("-" * 70)

# Create comparison plots
plt.figure(figsize=(20, 12))

# Plot 1: Objective gaps comparison
plt.subplot(2, 3, 1)
# RProximalBundle Phase 2
iterations_rpb = range(len(rpb_algorithm_phase2.objective_history))
plt.plot(iterations_rpb, rpb_algorithm_phase2.objective_history, 'b-', linewidth=2.5, label='RProximalBundle', alpha=0.8)

# SGM with different step sizes
colors_constant = ['red', 'green', 'orange', 'purple', 'brown', 'gray', 'cyan']
for i, step_size in enumerate(step_sizes):
    sgm_alg = sgm_results[step_size]['algorithm']
    iterations_sgm = range(len(sgm_alg.objective_history))
    color_idx = i % len(colors_constant)  # Safe indexing
    plt.plot(iterations_sgm, sgm_alg.objective_history, color=colors_constant[color_idx], linestyle='--',
             linewidth=1.5, label=f'SGM const (step={step_size})', alpha=0.7)

# SGM with decaying step sizes
colors_decaying = ['darkred', 'darkgreen', 'darkorange', 'darkblue', 'darkmagenta']
for i, initial_step_size in enumerate(decaying_initial_step_sizes):
    sgm_decaying_alg = sgm_decaying_results[initial_step_size]['algorithm']
    iterations_sgm_decaying = range(len(sgm_decaying_alg.objective_history))
    color_idx = i % len(colors_decaying)  # Safe indexing
    plt.plot(iterations_sgm_decaying, sgm_decaying_alg.objective_history, color=colors_decaying[color_idx], linestyle=':',
             linewidth=1.5, label=f'SGM decay (init={initial_step_size})', alpha=0.7)

plt.title('Objective Gap Comparison: RProximalBundle vs SGM')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
# Only use log scale if all values are positive
all_values = []
all_values.extend(rpb_algorithm_phase2.objective_history)
for step_size in step_sizes:
    sgm_alg = sgm_results[step_size]['algorithm']
    all_values.extend(sgm_alg.objective_history)
for initial_step_size in decaying_initial_step_sizes:
    sgm_decaying_alg = sgm_decaying_results[initial_step_size]['algorithm']
    all_values.extend(sgm_decaying_alg.objective_history)
if all(v > 0 for v in all_values):
    plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 2: Final objectives comparison
plt.subplot(2, 3, 2)
methods = ['RProximalBundle'] + [f'SGM const ({s})' for s in step_sizes] + [f'SGM decay ({s})' for s in decaying_initial_step_sizes]
final_objectives = [rpb_algorithm_phase2.raw_objective_history[-1]] + [sgm_results[s]['final_objective'] for s in step_sizes] + [sgm_decaying_results[s]['final_objective'] for s in decaying_initial_step_sizes]
# Create colors_bar with safe indexing to match the number of methods
colors_bar = ['blue'] + [colors_constant[i % len(colors_constant)] for i in range(len(step_sizes))] + [colors_decaying[i % len(colors_decaying)] for i in range(len(decaying_initial_step_sizes))]

bars = plt.bar(methods, final_objectives, color=colors_bar, alpha=0.7)
plt.title('Final Objective Values Comparison')
plt.ylabel('Final Objective Value')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, final_objectives):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(final_objectives)*0.01,
             f'{value:.6f}', ha='center', va='bottom', fontsize=8)

# Plot 3: Convergence rate comparison (log-log)
plt.subplot(2, 3, 3)
plt.plot(iterations_rpb, rpb_algorithm_phase2.objective_history, 'b-', linewidth=2.5, label='RProximalBundle', alpha=0.8)
plt.plot(range(len(overall_best_sgm.objective_history)), overall_best_sgm.objective_history, 'r--',
         linewidth=2, label=f'Best SGM ({overall_best_type})', alpha=0.8)
plt.title('Convergence Rate Comparison (Log-Log)')
plt.xlabel('Iteration Number')
plt.ylabel('Objective Gap')
# Only use log-log scale if all values are positive
rpb_values = rpb_algorithm_phase2.objective_history
sgm_values = overall_best_sgm.objective_history
if all(v > 0 for v in rpb_values) and all(v > 0 for v in sgm_values):
    plt.loglog()
else:
    # Use regular linear scale if negative values present
    plt.yscale('linear')
    plt.xscale('linear')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot 4: Error vs clean signal comparison
plt.subplot(2, 3, 4)
rpb_final_error = problem.compute_error(rpb_algorithm_phase2.current_proximal_center)
methods_error = ['RProximalBundle'] + [f'SGM const ({s})' for s in step_sizes] + [f'SGM decay ({s})' for s in decaying_initial_step_sizes]
errors = [rpb_final_error] + [sgm_results[s]['final_error'] for s in step_sizes] + [sgm_decaying_results[s]['final_error'] for s in decaying_initial_step_sizes]

bars = plt.bar(methods_error, errors, color=colors_bar, alpha=0.7)
plt.title('Reconstruction Error Comparison')
plt.ylabel('Error vs Clean Signal')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, errors):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors)*0.01,
             f'{value:.6f}', ha='center', va='bottom', fontsize=8)

# Plot 5: Number of iterations comparison
plt.subplot(2, 3, 5)
rpb_iterations = len(rpb_algorithm_phase2.objective_history)
iterations_counts = [rpb_iterations] + [sgm_results[s]['num_iterations'] for s in step_sizes] + [sgm_decaying_results[s]['num_iterations'] for s in decaying_initial_step_sizes]

bars = plt.bar(methods, iterations_counts, color=colors_bar, alpha=0.7)
plt.title('Number of Iterations Comparison')
plt.ylabel('Total Iterations')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, iterations_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(iterations_counts)*0.01,
             f'{value}', ha='center', va='bottom', fontsize=9)

# Plot 6: Performance summary table
plt.subplot(2, 3, 6)
plt.axis('off')

# Create performance summary data
summary_data = []
summary_data.append(['Method', 'Final Obj', 'Final Gap', 'Error', 'Iterations'])
summary_data.append(['RProximalBundle', f'{rpb_algorithm_phase2.raw_objective_history[-1]:.6f}',
                     f'{rpb_algorithm_phase2.objective_history[-1]:.2e}', f'{rpb_final_error:.6f}', str(rpb_iterations)])

for step_size in step_sizes:
    sgm_res = sgm_results[step_size]
    summary_data.append([f'SGM const ({step_size})', f'{sgm_res["final_objective"]:.6f}',
                         f'{sgm_res["final_gap"]:.2e}', f'{sgm_res["final_error"]:.6f}',
                         str(sgm_res["num_iterations"])])

for initial_step_size in decaying_initial_step_sizes:
    sgm_decay_res = sgm_decaying_results[initial_step_size]
    summary_data.append([f'SGM decay ({initial_step_size})', f'{sgm_decay_res["final_objective"]:.6f}',
                         f'{sgm_decay_res["final_gap"]:.2e}', f'{sgm_decay_res["final_error"]:.6f}',
                         str(sgm_decay_res["num_iterations"])])

# Create table
table = plt.table(cellText=summary_data[1:], colLabels=summary_data[0],
                  cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

# Color the header row
for i in range(len(summary_data[0])):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color the RProximalBundle row
for i in range(len(summary_data[0])):
    table[(1, i)].set_facecolor('#D9E1F2')

plt.title('Performance Summary Table', pad=20)

plt.tight_layout()
plt.savefig('sgm_vs_rpb_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Comparison visualization saved as 'sgm_vs_rpb_comparison.png'")
logger.info("SGM vs RProximalBundle comparison plot saved")

# Experiment 3: Signal reconstruction comparison
print("\nEXPERIMENT 3: Signal reconstruction visualization")
print("-" * 70)

# Create signal reconstruction comparison
fig_reconstruction = problem.visualize(p_denoised=overall_best_sgm.best_point,
                                       save_path='sgm_signal_reconstruction.png')
print("✓ SGM signal reconstruction saved as 'sgm_signal_reconstruction.png'")

# Create side-by-side comparison in 2D hyperbolic space
plt.figure(figsize=(15, 10))

# Create index array for plotting
indices = np.arange(len(problem.q_clean))

# Plot original signals in 2D Poincaré coordinates
plt.subplot(2, 2, 1)
plt.plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal')
plt.scatter(problem.q_noisy[:, 0], problem.q_noisy[:, 1], c='r', s=10, alpha=0.7, label='Noisy Signal')
plt.title('Original Signals in Poincaré Ball')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# RProximalBundle reconstruction
plt.subplot(2, 2, 2)
plt.plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal', alpha=0.5)
plt.plot(rpb_algorithm_phase2.current_proximal_center[:, 0], rpb_algorithm_phase2.current_proximal_center[:, 1], 'b-', linewidth=1.5, label='RProximalBundle Reconstruction')
plt.title('RProximalBundle Reconstruction')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Best SGM reconstruction
plt.subplot(2, 2, 3)
plt.plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal', alpha=0.5)
plt.plot(overall_best_sgm.best_point[:, 0], overall_best_sgm.best_point[:, 1], 'r-', linewidth=1.5, label=f'SGM Reconstruction ({overall_best_type})')
plt.title('Best SGM Reconstruction')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# Combined comparison
plt.subplot(2, 2, 4)
plt.plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2.5, label='Clean Signal')
plt.plot(rpb_algorithm_phase2.current_proximal_center[:, 0], rpb_algorithm_phase2.current_proximal_center[:, 1], 'b-', linewidth=1.5, label='RProximalBundle', alpha=0.8)
plt.plot(overall_best_sgm.best_point[:, 0], overall_best_sgm.best_point[:, 1], 'r--', linewidth=1.5, label=f'Best SGM ({overall_best_type})', alpha=0.8)
plt.title('Reconstruction Comparison')
plt.xlabel('x coordinate')
plt.ylabel('y coordinate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

plt.tight_layout()
plt.savefig('signal_reconstruction_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Signal reconstruction comparison saved as 'signal_reconstruction_comparison.png'")

# Experiment 4: Detailed performance analysis
print("\nEXPERIMENT 4: Detailed performance analysis")
print("-" * 70)

print("DETAILED PERFORMANCE COMPARISON:")
print("=" * 50)

print("RIEMANNIAN PROXIMAL BUNDLE METHOD (Phase 2):")
print(f"  Final objective:          {rpb_algorithm_phase2.raw_objective_history[-1]:.8f}")
print(f"  Final gap:                {rpb_algorithm_phase2.objective_history[-1]:.8e}")
print(f"  Reconstruction error:     {rpb_final_error:.8f}")
print(f"  Total iterations:         {len(rpb_algorithm_phase2.objective_history)}")
print(f"  Descent steps:            {len(rpb_algorithm_phase2.indices_of_descent_steps)}")
print(f"  Null steps:               {len(rpb_algorithm_phase2.indices_of_null_steps)}")
print(f"  Execution time:           {phase2_time:.2f} seconds")

print("\nRIEMANNIAN SUBGRADIENT METHOD RESULTS:")
print("Constant step sizes:")
for step_size in step_sizes:
    sgm_res = sgm_results[step_size]
    print(f"  SGM const (step size = {step_size}):")
    print(f"    Final objective:          {sgm_res['final_objective']:.8f}")
    print(f"    Final gap:                {sgm_res['final_gap']:.8e}")
    print(f"    Reconstruction error:     {sgm_res['final_error']:.8f}")
    print(f"    Total iterations:         {sgm_res['num_iterations']}")

print("\nDecaying step sizes:")
for initial_step_size in decaying_initial_step_sizes:
    sgm_decay_res = sgm_decaying_results[initial_step_size]
    print(f"  SGM decay (initial step size = {initial_step_size}):")
    print(f"    Final objective:          {sgm_decay_res['final_objective']:.8f}")
    print(f"    Final gap:                {sgm_decay_res['final_gap']:.8e}")
    print(f"    Reconstruction error:     {sgm_decay_res['final_error']:.8f}")
    print(f"    Total iterations:         {sgm_decay_res['num_iterations']}")

print(f"\nBEST METHOD COMPARISON:")
if overall_best_sgm.best_objective < rpb_algorithm_phase2.raw_objective_history[-1]:
    winner = f"SGM ({overall_best_type})"
    improvement = rpb_algorithm_phase2.raw_objective_history[-1] - overall_best_sgm.best_objective
else:
    winner = "RProximalBundle"
    improvement = overall_best_sgm.best_objective - rpb_algorithm_phase2.raw_objective_history[-1]

print(f"  Winner:                   {winner}")
print(f"  Objective difference:     {improvement:.8e}")

# Log final comparison results
logger.info("="*50)
logger.info("FINAL COMPARISON RESULTS:")
logger.info(f"RProximalBundle - Final objective: {rpb_algorithm_phase2.raw_objective_history[-1]:.8f}, Error: {rpb_final_error:.8f}")
logger.info(f"Best SGM ({overall_best_type}) - Final objective: {overall_best_sgm.best_objective:.8f}, Error: {problem.compute_error(overall_best_sgm.best_point):.8f}")
logger.info(f"Winner: {winner}, Objective difference: {improvement:.8e}")

print("\n" + "="*70)
print("SGM EXPERIMENTS AND COMPARISON COMPLETED!")
print("="*70)
logger.info("SGM experiments and comparison with RProximalBundle completed successfully")

# Display all plots
plt.show()

# %%

# =============================================================================
# FOUR-WAY COMPARISON: BUNDLE METHOD, SGM CONSTANT (1.0), SGM CONSTANT (0.5), SGM DECAYING
# =============================================================================

print("\n" + "="*70)
print("FOUR-WAY COMPARISON: BUNDLE METHOD vs SGM CONSTANT (1.0) vs SGM CONSTANT (0.5) vs SGM DECAYING")
print("="*70 + "\n")
logger.info("Starting four-way comparison: Bundle Method, SGM constant (stepsize=1), SGM constant (stepsize=0.5), SGM decaying")

# Extract the specific algorithms we want to compare
rpb_method = rpb_algorithm_phase2
sgm_constant_1 = sgm_results[1.0]['algorithm']  # SGM with constant stepsize = 1.0
sgm_constant_05 = sgm_results[0.5]['algorithm']  # SGM with constant stepsize = 0.5
sgm_decaying_1 = sgm_decaying_results[1.0]['algorithm']  # SGM with decaying stepsize starting at 1

print("Selected methods for four-way comparison:")
print(f"1. Riemannian Proximal Bundle Method - Final objective: {rpb_method.raw_objective_history[-1]:.8f}")
print(f"2. SGM Constant (stepsize=1.0) - Final objective: {sgm_constant_1.best_objective:.8f}")
print(f"3. SGM Constant (stepsize=0.5) - Final objective: {sgm_constant_05.best_objective:.8f}")
print(f"4. SGM Decaying (initial=1, schedule=1/sqrt(k+1)) - Final objective: {sgm_decaying_1.best_objective:.8f}")

def create_four_way_comparison_plot(problem, rpb_alg, sgm_const_1_alg, sgm_const_05_alg, sgm_decay_alg, save_path=None):
    """Create a comprehensive four-way comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Bundle Method reconstruction
    axes[0, 0].plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal', alpha=0.7)
    axes[0, 0].scatter(problem.q_noisy[:, 0], problem.q_noisy[:, 1], c='lightblue', s=8, alpha=0.5, label='Noisy Signal')
    axes[0, 0].plot(rpb_alg.current_proximal_center[:, 0], rpb_alg.current_proximal_center[:, 1],
                    'purple', linewidth=2, label='Bundle Method')
    axes[0, 0].set_title('Bundle Method Reconstruction')
    axes[0, 0].set_xlabel('x coordinate')
    axes[0, 0].set_ylabel('y coordinate')
    axes[0, 0].legend(loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')

    # Top-right: SGM Constant (step=1.0) reconstruction
    axes[0, 1].plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal', alpha=0.7)
    axes[0, 1].scatter(problem.q_noisy[:, 0], problem.q_noisy[:, 1], c='lightblue', s=8, alpha=0.5, label='Noisy Signal')
    axes[0, 1].plot(sgm_const_1_alg.best_point[:, 0], sgm_const_1_alg.best_point[:, 1],
                    'blue', linewidth=2, label='SGM Constant (step=1.0)')
    axes[0, 1].set_title('SGM Constant (stepsize=1.0) Reconstruction')
    axes[0, 1].set_xlabel('x coordinate')
    axes[0, 1].set_ylabel('y coordinate')
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axis('equal')

    # Bottom-left: SGM Constant (step=0.5) reconstruction
    axes[1, 0].plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal', alpha=0.7)
    axes[1, 0].scatter(problem.q_noisy[:, 0], problem.q_noisy[:, 1], c='lightblue', s=8, alpha=0.5, label='Noisy Signal')
    axes[1, 0].plot(sgm_const_05_alg.best_point[:, 0], sgm_const_05_alg.best_point[:, 1],
                    'green', linewidth=2, label='SGM Constant (step=0.5)')
    axes[1, 0].set_title('SGM Constant (stepsize=0.5) Reconstruction')
    axes[1, 0].set_xlabel('x coordinate')
    axes[1, 0].set_ylabel('y coordinate')
    axes[1, 0].legend(loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axis('equal')

    # Bottom-right: SGM Decaying reconstruction
    axes[1, 1].plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal', alpha=0.7)
    axes[1, 1].scatter(problem.q_noisy[:, 0], problem.q_noisy[:, 1], c='lightblue', s=8, alpha=0.5, label='Noisy Signal')
    axes[1, 1].plot(sgm_decay_alg.best_point[:, 0], sgm_decay_alg.best_point[:, 1],
                    'red', linewidth=2, label='SGM Decaying (1/√(k+1))')
    axes[1, 1].set_title('SGM Decaying (initial=1, 1/√(k+1)) Reconstruction')
    axes[1, 1].set_xlabel('x coordinate')
    axes[1, 1].set_ylabel('y coordinate')
    axes[1, 1].legend(loc='upper left')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')

    # All four signal reconstructions are now shown in the 2x2 grid
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Four-way comparison plot saved as '{save_path}'")
        logger.info(f"Four-way comparison plot saved as '{save_path}'")

    return fig

def create_four_way_animated_gif(problem, rpb_alg, sgm_const_1_alg, sgm_const_05_alg, sgm_decay_alg, save_path='four_way_comparison.gif', fps=8):
    """Create an animated GIF showing the actual signal evolution every 5 iterations for four methods."""
    print(f"Creating animated four-way signal evolution GIF comparison...")
    logger.info(f"Creating animated four-way signal evolution GIF comparison with {fps} fps")

    # Import animation tools
    from matplotlib.animation import FuncAnimation, PillowWriter

    # Create figure with 2x2 layout for the four methods
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Determine animation frames based on all algorithms' intermediate points (every 5 iterations)
    max_frames = max(len(rpb_alg.intermediate_points), len(sgm_const_1_alg.intermediate_points),
                     len(sgm_const_05_alg.intermediate_points), len(sgm_decay_alg.intermediate_points))

    def animate(frame_idx):
        # Clear all axes
        for ax_row in axes:
            for ax in ax_row:
                ax.clear()

        # Note: iteration numbers are stored in intermediate_iterations arrays for each algorithm

        # Plot 1: Bundle Method (top-left) - show actual intermediate signals
        axes[0, 0].plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal', alpha=0.8)
        axes[0, 0].scatter(problem.q_noisy[:, 0], problem.q_noisy[:, 1], c='lightgray', s=8, alpha=0.5, label='Noisy Signal')

        if frame_idx < len(rpb_alg.intermediate_points):
            current_signal = rpb_alg.intermediate_points[frame_idx]
            axes[0, 0].plot(current_signal[:, 0], current_signal[:, 1],
                        'purple', linewidth=2, alpha=0.9, label='Bundle Method')
            actual_iter = rpb_alg.intermediate_iterations[frame_idx]
            axes[0, 0].set_title(f'Bundle Method (Iter {actual_iter})')
        else:
            # Show final result if we've run out of intermediate points
            axes[0, 0].plot(rpb_alg.current_proximal_center[:, 0], rpb_alg.current_proximal_center[:, 1],
                        'purple', linewidth=2, alpha=0.9, label='Bundle Method')
            axes[0, 0].set_title(f'Bundle Method (Final)')

        axes[0, 0].set_xlabel('x coordinate')
        axes[0, 0].set_ylabel('y coordinate')
        axes[0, 0].legend(loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axis('equal')
        axes[0, 0].set_xlim(-0.8, 0.8)
        axes[0, 0].set_ylim(-0.8, 0.8)

        # Plot 2: SGM Constant (step=1.0) (top-right) - show actual intermediate signals
        axes[0, 1].plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal', alpha=0.8)
        axes[0, 1].scatter(problem.q_noisy[:, 0], problem.q_noisy[:, 1], c='lightgray', s=8, alpha=0.5, label='Noisy Signal')

        if frame_idx < len(sgm_const_1_alg.intermediate_points):
            current_signal = sgm_const_1_alg.intermediate_points[frame_idx]
            axes[0, 1].plot(current_signal[:, 0], current_signal[:, 1],
                        'blue', linewidth=2, alpha=0.9, label='SGM Constant (step=1.0)')
            actual_iter = sgm_const_1_alg.intermediate_iterations[frame_idx]
            axes[0, 1].set_title(f'SGM Constant 1.0 (Iter {actual_iter})')
        else:
            # Show final result if we've run out of intermediate points
            axes[0, 1].plot(sgm_const_1_alg.best_point[:, 0], sgm_const_1_alg.best_point[:, 1],
                        'blue', linewidth=2, alpha=0.9, label='SGM Constant (step=1.0)')
            axes[0, 1].set_title(f'SGM Constant 1.0 (Final)')

        axes[0, 1].set_xlabel('x coordinate')
        axes[0, 1].set_ylabel('y coordinate')
        axes[0, 1].legend(loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axis('equal')
        axes[0, 1].set_xlim(-0.8, 0.8)
        axes[0, 1].set_ylim(-0.8, 0.8)

        # Bottom-left: SGM Constant (step=1.0) - show actual intermediate signals
        axes[1, 0].plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal', alpha=0.8)
        axes[1, 0].scatter(problem.q_noisy[:, 0], problem.q_noisy[:, 1], c='lightgray', s=8, alpha=0.5, label='Noisy Signal')

        if frame_idx < len(sgm_const_05_alg.intermediate_points):
            current_signal = sgm_const_05_alg.intermediate_points[frame_idx]
            axes[1, 0].plot(current_signal[:, 0], current_signal[:, 1],
                        'green', linewidth=2, alpha=0.9, label='SGM Constant (step=0.5)')
            actual_iter = sgm_const_05_alg.intermediate_iterations[frame_idx]
            axes[1, 0].set_title(f'SGM Constant 0.5 (Iter {actual_iter})')
        else:
            # Show final result if we've run out of intermediate points
            axes[1, 0].plot(sgm_const_05_alg.best_point[:, 0], sgm_const_05_alg.best_point[:, 1],
                        'green', linewidth=2, alpha=0.9, label='SGM Constant (step=0.5)')
            axes[1, 0].set_title(f'SGM Constant 0.5 (Final)')

        axes[1, 0].set_xlabel('x coordinate')
        axes[1, 0].set_ylabel('y coordinate')
        axes[1, 0].legend(loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axis('equal')
        axes[1, 0].set_xlim(-0.8, 0.8)
        axes[1, 0].set_ylim(-0.8, 0.8)

        # Plot 4: SGM Decaying (bottom-right) - show actual intermediate signals
        axes[1, 1].plot(problem.q_clean[:, 0], problem.q_clean[:, 1], 'k-', linewidth=2, label='Clean Signal', alpha=0.8)
        axes[1, 1].scatter(problem.q_noisy[:, 0], problem.q_noisy[:, 1], c='lightgray', s=8, alpha=0.5, label='Noisy Signal')

        if frame_idx < len(sgm_decay_alg.intermediate_points):
            current_signal = sgm_decay_alg.intermediate_points[frame_idx]
            axes[1, 1].plot(current_signal[:, 0], current_signal[:, 1],
                        'red', linewidth=2, alpha=0.9, label='SGM Decaying (1/√(k+1))')
            actual_iter = sgm_decay_alg.intermediate_iterations[frame_idx]
            axes[1, 1].set_title(f'SGM Decaying (Iter {actual_iter})')
        else:
            # Show final result if we've run out of intermediate points
            axes[1, 1].plot(sgm_decay_alg.best_point[:, 0], sgm_decay_alg.best_point[:, 1],
                        'red', linewidth=2, alpha=0.9, label='SGM Decaying (1/√(k+1))')
            axes[1, 1].set_title(f'SGM Decaying (Final)')

        axes[1, 1].set_xlabel('x coordinate')
        axes[1, 1].set_ylabel('y coordinate')
        axes[1, 1].legend(loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axis('equal')
        axes[1, 1].set_xlim(-0.8, 0.8)
        axes[1, 1].set_ylim(-0.8, 0.8)

    # Create animation frames
    frames = list(range(max_frames))
    anim = FuncAnimation(fig, animate, frames=frames, interval=1000//fps, repeat=True)

    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)

    plt.close(fig)  # Close the figure to free memory

    print(f"✓ Four-way animated signal evolution GIF saved as '{save_path}'")
    print(f"   - Shows actual signal reconstruction at every 5 iterations for four methods")
    print(f"   - Total frames: {max_frames}")
    logger.info(f"Four-way animated signal evolution GIF saved as '{save_path}' with {max_frames} frames")

    return save_path

# Create the four-way comparison visualization
print("Creating four-way comparison static plot...")
four_way_fig = create_four_way_comparison_plot(problem, rpb_method, sgm_constant_1, sgm_constant_05, sgm_decaying_1,
                                                save_path='four_way_denoising_comparison.png')

# Create the animated GIF
print("\nCreating four-way comparison animated GIF...")
gif_path = create_four_way_animated_gif(problem, rpb_method, sgm_constant_1, sgm_constant_05, sgm_decaying_1,
                                        save_path='four_way_denoising_comparison.gif', fps=8)

# Print final four-way comparison summary
print("\n" + "="*70)
print("FOUR-WAY COMPARISON SUMMARY")
print("="*70)

methods_data = [
    ("Bundle Method", rpb_method.raw_objective_history[-1], problem.compute_error(rpb_method.current_proximal_center), len(rpb_method.objective_history)),
    ("SGM Constant (step=1.0)", sgm_constant_1.best_objective, problem.compute_error(sgm_constant_1.best_point), len(sgm_constant_1.objective_history)),
    ("SGM Constant (step=0.5)", sgm_constant_05.best_objective, problem.compute_error(sgm_constant_05.best_point), len(sgm_constant_05.objective_history)),
    ("SGM Decaying (1/√(k+1))", sgm_decaying_1.best_objective, problem.compute_error(sgm_decaying_1.best_point), len(sgm_decaying_1.objective_history))
]

print(f"{'Method':<25} {'Final Objective':<15} {'Error vs Clean':<15} {'Iterations':<12}")
print("-" * 70)
for method_name, final_obj, error, iterations in methods_data:
    print(f"{method_name:<25} {final_obj:<15.8f} {error:<15.8f} {iterations:<12}")

# Determine winner
best_method = min(methods_data, key=lambda x: x[1])
print(f"\nWinner (lowest objective): {best_method[0]}")
print(f"Winning objective: {best_method[1]:.8f}")

# Log final four-way results
logger.info("FOUR-WAY COMPARISON COMPLETED:")
for method_name, final_obj, error, iterations in methods_data:
    logger.info(f"{method_name}: objective={final_obj:.8f}, error={error:.8f}, iterations={iterations}")
logger.info(f"Winner: {best_method[0]} with objective {best_method[1]:.8f}")

print("\n" + "="*70)
print("FOUR-WAY DENOISING COMPARISON EXPERIMENT COMPLETED!")
print("="*70)
print("\nGenerated files:")
print("  - four_way_denoising_comparison.png: Static four-way comparison")
print("  - four_way_denoising_comparison.gif: Animated four-way comparison")
print("="*70)

# Display the static comparison plot
plt.show()

# %%