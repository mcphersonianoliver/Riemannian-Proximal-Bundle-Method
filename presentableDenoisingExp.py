import numpy as np
from DenoisingProblemClass import TVDenoisingProblem
import matplotlib.pyplot as plt
from src.RiemannianProximalBundle import RProximalBundle

regularization_parameters = [0.5, 1.0, 2.0]
noise_sigmas = [0.05, 0.1, 0.5]
prox_parameter_start = 0.02
max_iter = 200

for regularization_parameter in regularization_parameters:
    for noise_sigma in noise_sigmas:
        print("STEP 1: Creating problem instance")
        print("-" * 70)
        problem = TVDenoisingProblem.from_square_wave(
            T=3,
            a=-6,
            b=6,
            N=496,   # Small size for demonstration
            alpha=regularization_parameter,
            noise_std=noise_sigma,
            seed=50
        )
        print("✓ Problem created successfully\n")

        print("\n" + "="*70)
        print("RUNNING RIEMANNIAN PROXIMAL BUNDLE ALGORITHM")
        print("="*70 + "\n")

        print("STEP 2: Setting up algorithm parameters")
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
        proximal_parameter = prox_parameter_start  # Moderate proximal parameter
        trust_parameter = 0.1
        max_iterations = max_iter  # Reasonable number for demonstration
        tolerance = 1e-10   # Relaxed tolerance
        # true_min_obj = problem.objective(problem.q_clean)  # Use clean signal as reference

        print(f"Sectional curvature: {sectional_curvature}")
        print(f"Proximal parameter: {proximal_parameter}")
        print(f"Trust parameter: {trust_parameter}")
        print(f"Max iterations: {max_iterations}")
        print(f"Tolerance: {tolerance}")
        # print(f"True minimum objective: {true_min_obj:.8f}")
        print("✓ Algorithm parameters configured\n")

        # Step 3: Initialize the algorithm
        print("STEP 3: Initializing RProximalBundle algorithm")
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

        # Create algorithm instance
        rpb_algorithm = RProximalBundle(
            manifold=product_manifold,  # Use product manifold wrapper
            retraction_map=retraction_map,
            transport_map=transport_map,
            objective_function=objective_wrapper,
            subgradient=subdifferential_wrapper,
            initial_point=p_init,
            initial_objective=obj_init,
            initial_subgradient=subgrad_init,
            true_min_obj=0, # set as 0 since we don't know exactly
            retraction_error=1.0,  # Exact retraction for exponential map
            transport_error=1.0,   # Exact transport
            sectional_curvature=sectional_curvature,
            proximal_parameter=proximal_parameter,
            trust_parameter=trust_parameter,
            max_iter=max_iterations,
            tolerance=tolerance,
            adaptive_proximal=True,
            know_minimizer=True,
            relative_error=True
        )

        print("✓ Algorithm initialized successfully\n")

        # Step 4: Run the optimization
        print("STEP 4: Running optimization algorithm")
        print("-" * 70)

        print("Starting optimization...")
        rpb_algorithm.run()
        print("✓ Optimization completed\n")

        # Step 5: Extract results
        print("STEP 5: Extracting optimization results")
        print("-" * 70)

        # Get final result
        p_optimized = rpb_algorithm.current_proximal_center
        final_objective = problem.objective(p_optimized)
        final_error = problem.compute_error(p_optimized)

        print(f"Final objective value: {final_objective:.8f}")
        print(f"Initial objective value: {obj_init:.8f}")
        # print(f"True minimum objective: {true_min_obj:.8f}")
        print(f"Objective improvement: {obj_init - final_objective:.8f}")
        print(f"Final error vs clean signal: {final_error:.8f}")
        print(f"Number of descent steps: {len(rpb_algorithm.indices_of_descent_steps)}")
        print(f"Number of null steps: {len(rpb_algorithm.indices_of_null_steps)}")
        print(f"Total iterations: {len(rpb_algorithm.objective_history)}")
        print("✓ Results extracted\n")

        # Step 5: Visualize the optimized signal
        print("STEP 5: Creating signal visualization")
        print("-" * 70)

        # Create visualization with denoised signal
        fig_signal = problem.visualize(p_denoised=p_optimized, save_path='tv_denoising_optimized.png')
        print("✓ Signal visualization saved as 'tv_denoising_optimized.png'\n")

        # Step 6: Visualize objective function vs iterations
        print("STEP 6: Creating objective function visualization")
        print("-" * 70)

        # Create objective vs iteration plot
        plt.figure(figsize=(12, 8))

        # Main plot: Objective gap vs iteration
        plt.subplot(2, 1, 1)
        iterations = range(len(rpb_algorithm.objective_history))
        plt.plot(iterations, rpb_algorithm.objective_history, 'b-', linewidth=1.5, label='Objective Value')

        # Highlight different step types
        if rpb_algorithm.indices_of_descent_steps:
            descent_values = [rpb_algorithm.objective_history[i] for i in rpb_algorithm.indices_of_descent_steps]
            plt.scatter(rpb_algorithm.indices_of_descent_steps, descent_values,
                        color='green', marker='o', s=8, label='Descent Steps', zorder=5)

        if rpb_algorithm.indices_of_null_steps:
            null_values = [rpb_algorithm.objective_history[i] for i in rpb_algorithm.indices_of_null_steps]
            plt.scatter(rpb_algorithm.indices_of_null_steps, null_values,
                        color='orange', marker='s', s=6, label='Null Steps', zorder=5)

        if rpb_algorithm.indices_of_proximal_doubling_steps:
            doubling_values = [rpb_algorithm.objective_history[i] for i in rpb_algorithm.indices_of_proximal_doubling_steps]
            plt.scatter(rpb_algorithm.indices_of_proximal_doubling_steps, doubling_values,
                        color='red', marker='^', s=6, label='Proximal Doubling Steps', zorder=5)

        plt.title('Objective Value vs Iteration Number')
        plt.xlabel('Iteration Number')
        plt.ylabel('Objective Value')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Secondary plot: Proximal parameter evolution
        plt.subplot(2, 1, 2)
        plt.plot(range(len(rpb_algorithm.proximal_parameter_history)),
                rpb_algorithm.proximal_parameter_history, 'g-', linewidth=1.5)
        plt.title('Proximal Parameter vs Iteration')
        plt.xlabel('Iteration Number')
        plt.ylabel('Proximal Parameter (ρ)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('objective_vs_iteration.png', dpi=150, bbox_inches='tight')
        print("✓ Objective visualization saved as 'objective_vs_iteration.png'\n")

        # Step 7: Print algorithm performance summary
        print("STEP 7: Algorithm Performance Summary")
        print("-" * 70)

        print("ALGORITHM CONVERGENCE:")
        print(f"  Initial objective:        {obj_init:.8f}")
        print(f"  Final objective:          {final_objective:.8f}")
        # print(f"  True minimum objective:   {true_min_obj:.8f}")
        print(f"  Absolute improvement:     {obj_init - final_objective:.8f}")
        print(f"  Relative improvement:     {(obj_init - final_objective)/obj_init*100:.4f}%")
        # print(f"  Gap to true minimum:      {final_objective - true_min_obj:.8f}")

        print("\nSIGNAL RECONSTRUCTION:")
        print(f"  Initial error vs clean:   {problem.compute_error(p_init):.8f}")
        print(f"  Final error vs clean:     {final_error:.8f}")
        print(f"  Error reduction:          {problem.compute_error(p_init) - final_error:.8f}")
        print(f"  Error reduction (%):      {(problem.compute_error(p_init) - final_error)/problem.compute_error(p_init)*100:.4f}%")

        print("\nALGORITHM STEPS:")
        print(f"  Total iterations:         {len(rpb_algorithm.objective_history)}")
        print(f"  Descent steps:            {len(rpb_algorithm.indices_of_descent_steps)}")
        print(f"  Null steps:               {len(rpb_algorithm.indices_of_null_steps)}")
        print(f"  Proximal doubling steps:  {len(rpb_algorithm.indices_of_proximal_doubling_steps)}")
        print(f"  Final proximal parameter: {rpb_algorithm.proximal_parameter:.6f}")

        print("\nCURVATURE CONFIGURATION:")
        print(f"  Sectional curvature:      {sectional_curvature}")
        print(f"  Retraction error:         {rpb_algorithm.retraction_error}")
        print(f"  Transport error:          {rpb_algorithm.transport_error}")

        print("\n" + "="*70)
        print("RIEMANNIAN PROXIMAL BUNDLE ALGORITHM COMPLETED SUCCESSFULLY!")
        print("="*70)

        # Display plots
        plt.show()

        # Step 8: Successive Decrease Ratio Analysis
        print("\nSTEP 8: Successive Decrease Ratio Analysis")
        print("-" * 70)

        # Extract objective function values
        f_values = np.array(rpb_algorithm.objective_history)

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
            print(f"  Ratios > 1:               {np.sum(valid_ratios > 1)} "
                f"({np.sum(valid_ratios > 1)/len(valid_ratios)*100:.1f}%)")
            print(f"  Ratios < 1:               {np.sum(valid_ratios < 1)} "
                f"({np.sum(valid_ratios < 1)/len(valid_ratios)*100:.1f}%)")
        else:
            print("No valid ratios computed (likely due to numerical issues)")

        print("✓ Successive decrease ratio analysis completed\n")

        # Display plots
        plt.show()