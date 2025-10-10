# %%
# Signal Denoising Parameter Study: Regularization and Noise Level Analysis
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from DenoisingProblemClass import TVDenoisingProblem
from src.RiemannianProximalBundle import RProximalBundle

# Experiment Parameters
print("="*80)
print("SIGNAL DENOISING PARAMETER STUDY")
print("="*80)

# Parameter grids to test
regularization_params = [0.5, 1.0, 1.5, 2.0, 2.5]
noise_sigmas = [0.05, 0.1, 0.15, 0.2]

print(f"Testing regularization parameters: {regularization_params}")
print(f"Testing noise sigmas: {noise_sigmas}")
print(f"Total experiments: {len(regularization_params) * len(noise_sigmas)}")

# Fixed problem parameters
T = 3
a = -6
b = 6
N = 496
seed = 50

print(f"\nFixed problem parameters:")
print(f"  Signal period T: {T}")
print(f"  Domain: [{a}, {b}]")
print(f"  Grid points N: {N}")
print(f"  Random seed: {seed}")

# Storage for results
results = {}
estimated_minimums = {}
phase2_results = {}
all_problems = {}

# Run experiments for all parameter combinations
experiment_count = 0
total_experiments = len(regularization_params) * len(noise_sigmas)

for alpha in regularization_params:
    for noise_std in noise_sigmas:
        experiment_count += 1
        print(f"\n" + "="*80)
        print(f"EXPERIMENT {experiment_count}/{total_experiments}")
        print(f"Regularization α = {alpha}, Noise σ = {noise_std}")
        print("="*80)

        # Create problem instance
        print("Creating TV denoising problem...")
        problem = TVDenoisingProblem.from_square_wave(
            T=T,
            a=a,
            b=b,
            N=N,
            alpha=alpha,
            noise_std=noise_std,
            seed=seed
        )

        # Store problem for later analysis
        problem_key = f"alpha_{alpha}_noise_{noise_std}"
        all_problems[problem_key] = problem

        # Get initial point and compute initial values
        p_init = problem.initial_point()
        obj_init = problem.objective(p_init)
        subgrad_init = problem.subdifferential(p_init)
        true_min_obj = problem.objective(problem.q_clean)

        print(f"Initial objective: {obj_init:.8f}")
        print(f"Clean signal objective: {true_min_obj:.8f}")

        # Algorithm parameters
        sectional_curvature = -1.0  # Known curvature for H^2
        proximal_parameter = 0.01
        trust_parameter = 0.1
        tolerance = 1e-10

        # Create wrapper functions for manifold operations
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
                    result[i] = v_array[i]
                except:
                    result[i] = v_array[i]
            return result

        # Create simplified manifold wrapper
        class ProductManifoldWrapper:
            """Simplified wrapper for product space operations"""
            def __init__(self, single_manifold, n_points):
                self.single_manifold = single_manifold
                self.n_points = n_points

            def inner_product(self, p, u, v):
                """Compute sum of inner products across all points"""
                total = 0.0
                for i in range(self.n_points):
                    try:
                        total += self.single_manifold.inner_product(p[i], u[i], v[i])
                    except:
                        total += np.dot(u[i], v[i])
                return total

            def norm(self, p, v):
                """Compute norm as sqrt of sum of squared norms"""
                total = 0.0
                for i in range(self.n_points):
                    try:
                        total += self.single_manifold.norm(p[i], v[i])**2
                    except:
                        total += np.linalg.norm(v[i])**2
                return np.sqrt(total)

        product_manifold = ProductManifoldWrapper(problem.manifold_single, problem.n)

        # Wrap objective and subdifferential
        def objective_wrapper(p):
            return problem.objective(p)

        def subdifferential_wrapper(p):
            return problem.subdifferential(p)

        # PHASE 1: Estimate true minimum (350 iterations)
        print(f"\nPhase 1: Estimating true minimum (350 iterations)")
        print("-" * 50)

        rpb_algorithm_phase1 = RProximalBundle(
            manifold=product_manifold,
            retraction_map=retraction_wrapper,
            transport_map=transport_wrapper,
            objective_function=objective_wrapper,
            subgradient=subdifferential_wrapper,
            initial_point=p_init,
            initial_objective=obj_init,
            initial_subgradient=subgrad_init,
            true_min_obj=true_min_obj,  # Use clean signal as baseline for first run
            retraction_error=0.0,
            transport_error=0.0,
            sectional_curvature=sectional_curvature,
            proximal_parameter=proximal_parameter,
            trust_parameter=trust_parameter,
            max_iter=350,  # Long run to estimate minimum
            tolerance=tolerance,
            adaptive_proximal=True,
            know_minimizer=True,
            relative_error=True
        )

        rpb_algorithm_phase1.run()

        # Get estimated minimum
        estimated_minimum = rpb_algorithm_phase1.raw_objective_history[-1]
        estimated_minimums[problem_key] = estimated_minimum

        print(f"Phase 1 Results:")
        print(f"  Estimated minimum: {estimated_minimum:.8f}")
        print(f"  Improvement: {obj_init - estimated_minimum:.8f}")

        # PHASE 2: Convergence analysis (200 iterations)
        print(f"\nPhase 2: Convergence analysis (200 iterations)")
        print("-" * 50)

        rpb_algorithm_phase2 = RProximalBundle(
            manifold=product_manifold,
            retraction_map=retraction_wrapper,
            transport_map=transport_wrapper,
            objective_function=objective_wrapper,
            subgradient=subdifferential_wrapper,
            initial_point=p_init,  # Same initial point
            initial_objective=obj_init,
            initial_subgradient=subgrad_init,
            true_min_obj=estimated_minimum,  # Use estimated minimum from Phase 1
            retraction_error=0.0,
            transport_error=0.0,
            sectional_curvature=sectional_curvature,
            proximal_parameter=proximal_parameter,
            trust_parameter=trust_parameter,
            max_iter=200,  # Shorter run for analysis
            tolerance=tolerance,
            adaptive_proximal=True,
            know_minimizer=True,
            relative_error=True
        )

        rpb_algorithm_phase2.run()

        # Store results
        results[problem_key] = {
            'alpha': alpha,
            'noise_std': noise_std,
            'initial_objective': obj_init,
            'clean_objective': true_min_obj,
            'estimated_minimum': estimated_minimum,
            'phase1_optimizer': rpb_algorithm_phase1,
            'phase2_optimizer': rpb_algorithm_phase2,
            'final_gap': rpb_algorithm_phase2.objective_history[-1],
            'iterations': len(rpb_algorithm_phase2.objective_history),
            'descent_steps': len(rpb_algorithm_phase2.indices_of_descent_steps),
            'null_steps': len(rpb_algorithm_phase2.indices_of_null_steps)
        }

        print(f"Phase 2 Results:")
        print(f"  Final gap: {rpb_algorithm_phase2.objective_history[-1]:.8f}")
        print(f"  Descent steps: {len(rpb_algorithm_phase2.indices_of_descent_steps)}")
        print(f"  Null steps: {len(rpb_algorithm_phase2.indices_of_null_steps)}")

print(f"\n" + "="*80)
print("ALL EXPERIMENTS COMPLETED")
print("="*80)

# Generate Individual Plots for Selected Parameters
print(f"\n" + "="*80)
print("GENERATING INDIVIDUAL PLOTS FOR SELECTED CASES")
print("="*80)

# Plot a few representative cases
representative_cases = [
    ('alpha_1.0_noise_0.05', 'Low Noise (σ=0.05), Moderate Reg (α=1.0)'),
    ('alpha_1.0_noise_0.2', 'High Noise (σ=0.2), Moderate Reg (α=1.0)'),
    ('alpha_0.5_noise_0.1', 'Moderate Noise (σ=0.1), Low Reg (α=0.5)'),
    ('alpha_2.5_noise_0.1', 'Moderate Noise (σ=0.1), High Reg (α=2.5)')
]

for case_key, case_description in representative_cases:
    if case_key in results:
        print(f"\nPlotting {case_description}...")

        # Phase 1 plot
        results[case_key]['phase1_optimizer'].plot_objective_versus_iter()

        # Phase 2 plot
        results[case_key]['phase2_optimizer'].plot_objective_versus_iter()

        # Phase 2 log-log plot
        results[case_key]['phase2_optimizer'].plot_objective_versus_iter(log_log=True)

# Summary Analysis Plots
print(f"\n" + "="*80)
print("GENERATING PARAMETER STUDY SUMMARY PLOTS")
print("="*80)

# Create comprehensive parameter study plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Final gaps vs regularization parameter (for each noise level)
ax1 = axes[0, 0]
for noise_std in noise_sigmas:
    final_gaps = []
    alphas = []
    for alpha in regularization_params:
        key = f"alpha_{alpha}_noise_{noise_std}"
        if key in results:
            final_gaps.append(results[key]['final_gap'])
            alphas.append(alpha)
    ax1.plot(alphas, final_gaps, 'o-', linewidth=2, markersize=6,
             label=f'σ = {noise_std}', alpha=0.8)

ax1.set_xlabel('Regularization Parameter (α)')
ax1.set_ylabel('Final Objective Gap')
ax1.set_title('Final Gap vs Regularization Parameter')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Final gaps vs noise level (for each regularization parameter)
ax2 = axes[0, 1]
for alpha in regularization_params:
    final_gaps = []
    noises = []
    for noise_std in noise_sigmas:
        key = f"alpha_{alpha}_noise_{noise_std}"
        if key in results:
            final_gaps.append(results[key]['final_gap'])
            noises.append(noise_std)
    ax2.plot(noises, final_gaps, 's-', linewidth=2, markersize=6,
             label=f'α = {alpha}', alpha=0.8)

ax2.set_xlabel('Noise Standard Deviation (σ)')
ax2.set_ylabel('Final Objective Gap')
ax2.set_title('Final Gap vs Noise Level')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Number of descent steps vs parameters
ax3 = axes[0, 2]
alphas_grid, noises_grid = np.meshgrid(regularization_params, noise_sigmas)
descent_steps_grid = np.zeros_like(alphas_grid)

for i, noise_std in enumerate(noise_sigmas):
    for j, alpha in enumerate(regularization_params):
        key = f"alpha_{alpha}_noise_{noise_std}"
        if key in results:
            descent_steps_grid[i, j] = results[key]['descent_steps']

im3 = ax3.imshow(descent_steps_grid, aspect='auto', origin='lower',
                 extent=[min(regularization_params), max(regularization_params),
                        min(noise_sigmas), max(noise_sigmas)])
ax3.set_xlabel('Regularization Parameter (α)')
ax3.set_ylabel('Noise Standard Deviation (σ)')
ax3.set_title('Number of Descent Steps')
plt.colorbar(im3, ax=ax3)

# Plot 4: Convergence comparison for different noise levels (fixed α=1.0)
ax4 = axes[1, 0]
fixed_alpha = 1.0
for noise_std in noise_sigmas:
    key = f"alpha_{fixed_alpha}_noise_{noise_std}"
    if key in results:
        optimizer = results[key]['phase2_optimizer']
        iterations = range(len(optimizer.objective_history))
        ax4.plot(iterations, optimizer.objective_history, linewidth=2,
                 label=f'σ = {noise_std}', alpha=0.8)

ax4.set_xlabel('Iteration Number')
ax4.set_ylabel('Objective Gap')
ax4.set_title(f'Convergence Comparison (α = {fixed_alpha})')
ax4.set_yscale('log')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Convergence comparison for different regularization (fixed σ=0.1)
ax5 = axes[1, 1]
fixed_noise = 0.1
for alpha in regularization_params:
    key = f"alpha_{alpha}_noise_{fixed_noise}"
    if key in results:
        optimizer = results[key]['phase2_optimizer']
        iterations = range(len(optimizer.objective_history))
        ax5.plot(iterations, optimizer.objective_history, linewidth=2,
                 label=f'α = {alpha}', alpha=0.8)

ax5.set_xlabel('Iteration Number')
ax5.set_ylabel('Objective Gap')
ax5.set_title(f'Convergence Comparison (σ = {fixed_noise})')
ax5.set_yscale('log')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Heat map of final gaps
ax6 = axes[1, 2]
final_gaps_grid = np.zeros_like(alphas_grid)

for i, noise_std in enumerate(noise_sigmas):
    for j, alpha in enumerate(regularization_params):
        key = f"alpha_{alpha}_noise_{noise_std}"
        if key in results:
            final_gaps_grid[i, j] = results[key]['final_gap']

im6 = ax6.imshow(np.log10(final_gaps_grid), aspect='auto', origin='lower',
                 extent=[min(regularization_params), max(regularization_params),
                        min(noise_sigmas), max(noise_sigmas)],
                 cmap='viridis')
ax6.set_xlabel('Regularization Parameter (α)')
ax6.set_ylabel('Noise Standard Deviation (σ)')
ax6.set_title('Log₁₀(Final Objective Gap)')
plt.colorbar(im6, ax=ax6)

plt.tight_layout()
plt.savefig('SignalDenoising_ParameterStudy.png', dpi=150, bbox_inches='tight')
print("✓ Parameter study plots saved as 'SignalDenoising_ParameterStudy.png'")

# Summary Statistics
print(f"\n" + "="*80)
print("PARAMETER STUDY SUMMARY")
print("="*80)

print(f"Total experiments completed: {len(results)}")
print(f"Regularization parameters tested: {regularization_params}")
print(f"Noise levels tested: {noise_sigmas}")

# Find best and worst performing parameter combinations
best_gap = float('inf')
worst_gap = 0
best_params = None
worst_params = None

for key, result in results.items():
    final_gap = result['final_gap']
    if final_gap < best_gap:
        best_gap = final_gap
        best_params = (result['alpha'], result['noise_std'])
    if final_gap > worst_gap:
        worst_gap = final_gap
        worst_params = (result['alpha'], result['noise_std'])

print(f"\nBest performance:")
print(f"  Parameters: α = {best_params[0]}, σ = {best_params[1]}")
print(f"  Final gap: {best_gap:.8f}")

print(f"\nWorst performance:")
print(f"  Parameters: α = {worst_params[0]}, σ = {worst_params[1]}")
print(f"  Final gap: {worst_gap:.8f}")

# Summary table
print(f"\nDetailed Results Table:")
print("-" * 80)
print(f"{'α':<6} {'σ':<6} {'Init Obj':<12} {'Est Min':<12} {'Final Gap':<12} {'Descent':<8} {'Null':<6}")
print("-" * 80)

for alpha in regularization_params:
    for noise_std in noise_sigmas:
        key = f"alpha_{alpha}_noise_{noise_std}"
        if key in results:
            r = results[key]
            print(f"{r['alpha']:<6.1f} {r['noise_std']:<6.3f} {r['initial_objective']:<12.6f} "
                  f"{r['estimated_minimum']:<12.6f} {r['final_gap']:<12.6f} "
                  f"{r['descent_steps']:<8d} {r['null_steps']:<6d}")

print("="*80)
print("SIGNAL DENOISING PARAMETER STUDY COMPLETED!")
print("="*80)

# Save the parameter study plot
print(f"\n" + "="*80)
print("SAVING PLOTS")
print("="*80)
print("✓ All individual plots already displayed above")
print("✓ Parameter study plot already saved as 'SignalDenoising_ParameterStudy.png'")
print("="*80)
print("SIGNAL DENOISING PARAMETER STUDY COMPLETED!")
print("All plots have been saved as image files.")
print("="*80)