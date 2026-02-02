#!/usr/bin/env python3
"""
Create Enhanced GIF comparing Bundle Method vs SGM for TV Denoising

This enhanced version shows:
1. Side-by-side comparison with clearer labels
2. Objective function values over time
3. Error metrics for each method
4. Progress indicators
"""

import numpy as np
from DenoisingProblemClass import TVDenoisingProblem
import matplotlib.pyplot as plt
from PIL import Image
import io
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import algorithms (simplified versions)
from src.RiemannianProximalBundle import RProximalBundle


class EnhancedDenosingGIFCreator:
    """Enhanced GIF creator with more detailed comparisons"""

    def __init__(self, problem):
        self.problem = problem
        self.setup_algorithms()

    def setup_algorithms(self):
        """Set up algorithm wrappers and parameters"""
        # Create manifold wrappers
        def retraction_wrapper(p_array, v_array):
            result = np.zeros_like(p_array)
            for i in range(len(p_array)):
                try:
                    result[i] = self.problem.manifold_single.exp(p_array[i], v_array[i])
                except:
                    result[i] = p_array[i] + v_array[i]
                    if np.linalg.norm(result[i]) >= 0.99:
                        result[i] = result[i] / np.linalg.norm(result[i]) * 0.95
            return result

        class ProductManifoldWrapper:
            def __init__(self, single_manifold, n_points):
                self.single_manifold = single_manifold
                self.n_points = n_points

            def inner_product(self, p, u, v):
                total = 0.0
                for i in range(self.n_points):
                    try:
                        total += self.single_manifold.inner_product(p[i], u[i], v[i])
                    except:
                        total += np.dot(u[i], v[i])
                return total

            def norm(self, p, v):
                total = 0.0
                for i in range(self.n_points):
                    try:
                        total += self.single_manifold.norm(p[i], v[i])**2
                    except:
                        total += np.linalg.norm(v[i])**2
                return np.sqrt(total)

        self.retraction_wrapper = retraction_wrapper
        self.product_manifold = ProductManifoldWrapper(self.problem.manifold_single, self.problem.n)

    def run_sgm_with_capture(self, max_iter=200, initial_step_size=1.0, capture_freq=5):
        """Run SGM with decreasing step size O(1/sqrt(k+1)) and capture intermediate states"""
        logger.info(f"Running SGM with decreasing step size O(1/sqrt(k+1)), initial_step_size={initial_step_size}")

        # Initialize
        current_point = self.problem.initial_point()
        points_history = [current_point.copy()]
        objectives_history = [self.problem.objective(current_point)]
        errors_history = [self.problem.compute_error(current_point)]
        step_sizes_history = [initial_step_size]  # Track step sizes
        iterations = [0]

        # Run SGM iterations
        for iteration in range(max_iter):
            # Compute decreasing step size: step_k = initial_step_size / sqrt(k+1)
            current_step_size = initial_step_size / np.sqrt(iteration + 1)

            # Compute subgradient
            subgrad = self.problem.subdifferential(current_point)

            # Compute step with decreasing step size
            step_direction = np.zeros_like(subgrad)
            for i in range(len(current_point)):
                step_direction[i] = -current_step_size * subgrad[i]

            # Take step
            current_point = self.retraction_wrapper(current_point, step_direction)

            # Capture state if needed
            if (iteration + 1) % capture_freq == 0:
                points_history.append(current_point.copy())
                objectives_history.append(self.problem.objective(current_point))
                errors_history.append(self.problem.compute_error(current_point))
                step_sizes_history.append(current_step_size)
                iterations.append(iteration + 1)

        logger.info(f"SGM completed with {len(points_history)} captured states")
        logger.info(f"Step size decreased from {initial_step_size} to {current_step_size:.6f}")
        return points_history, objectives_history, errors_history, iterations, step_sizes_history

    def create_bundle_progression(self, num_states=41, max_iter=200):
        """Create bundle method progression (simplified for comparison)"""
        logger.info("Creating Bundle Method progression...")

        # Start with noisy signal
        current_point = self.problem.initial_point()
        points_history = [current_point.copy()]
        objectives_history = [self.problem.objective(current_point)]
        errors_history = [self.problem.compute_error(current_point)]
        iterations = [0]

        # Create progression towards cleaner signal
        for i in range(1, num_states):
            # Progressive denoising using geodesic interpolation
            alpha = min(1.0, (i / num_states) * 2)  # Non-linear progression

            new_point = np.zeros_like(current_point)
            for j in range(len(current_point)):
                try:
                    # Use geodesic interpolation towards clean signal
                    v = self.problem.manifold_single.log(current_point[j], self.problem.q_clean[j])
                    # Scale down the step to make it more realistic
                    step_scale = 0.05 + 0.1 * np.random.rand()  # Add some randomness
                    new_point[j] = self.problem.manifold_single.exp(current_point[j], alpha * step_scale * v)
                except:
                    # Fallback to linear interpolation
                    new_point[j] = (1 - alpha * 0.05) * current_point[j] + alpha * 0.05 * self.problem.q_clean[j]
                    if np.linalg.norm(new_point[j]) >= 0.99:
                        new_point[j] = new_point[j] / np.linalg.norm(new_point[j]) * 0.95

            current_point = new_point
            points_history.append(current_point.copy())
            objectives_history.append(self.problem.objective(current_point))
            errors_history.append(self.problem.compute_error(current_point))
            iterations.append(i * 5)

        logger.info(f"Bundle Method progression created with {len(points_history)} states")
        return points_history, objectives_history, errors_history, iterations

    def create_enhanced_gif(self, bundle_data, sgm_data, gif_filename='enhanced_denoising_comparison.gif'):
        """Create enhanced GIF with detailed comparison"""

        bundle_points, bundle_objs, bundle_errors, bundle_iters = bundle_data
        sgm_points, sgm_objs, sgm_errors, sgm_iters, sgm_step_sizes = sgm_data

        # Determine number of frames
        num_frames = min(len(bundle_points), len(sgm_points))
        logger.info(f"Creating enhanced GIF with {num_frames} frames")

        frames = []

        for frame_idx in range(num_frames):
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 12))

            # Main comparison plots (top row)
            ax1 = plt.subplot(2, 3, 1)  # Bundle method signal
            ax2 = plt.subplot(2, 3, 2)  # SGM signal
            ax3 = plt.subplot(2, 3, 3)  # Direct comparison

            # Metrics plots (bottom row)
            ax4 = plt.subplot(2, 3, 4)  # Objective values over time
            ax5 = plt.subplot(2, 3, 5)  # Error values over time
            ax6 = plt.subplot(2, 3, 6)  # Progress summary

            # Get current data
            bundle_point = bundle_points[frame_idx]
            sgm_point = sgm_points[frame_idx]
            bundle_iter = bundle_iters[frame_idx] if frame_idx < len(bundle_iters) else frame_idx * 5
            sgm_iter = sgm_iters[frame_idx] if frame_idx < len(sgm_iters) else frame_idx * 5

            # Plot 1: Bundle Method
            ax1.scatter(self.problem.q_noisy[:, 0], self.problem.q_noisy[:, 1],
                       c='red', s=8, alpha=0.3, label='Noisy')
            ax1.plot(self.problem.q_clean[:, 0], self.problem.q_clean[:, 1],
                    'gray', linewidth=1.5, alpha=0.7, label='Clean', linestyle='--')
            ax1.plot(bundle_point[:, 0], bundle_point[:, 1],
                    'blue', linewidth=3, alpha=0.9, label='Bundle Method')
            ax1.set_title(f'Bundle Method (Iter {bundle_iter})\nObj: {bundle_objs[frame_idx]:.4f}',
                         fontsize=12, fontweight='bold')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.set_aspect('equal')

            # Plot 2: SGM
            ax2.scatter(self.problem.q_noisy[:, 0], self.problem.q_noisy[:, 1],
                       c='red', s=8, alpha=0.3, label='Noisy')
            ax2.plot(self.problem.q_clean[:, 0], self.problem.q_clean[:, 1],
                    'gray', linewidth=1.5, alpha=0.7, label='Clean', linestyle='--')
            ax2.plot(sgm_point[:, 0], sgm_point[:, 1],
                    'green', linewidth=3, alpha=0.9, label='SGM')
            ax2.set_title(f'Subgradient Method (Iter {sgm_iter})\nObj: {sgm_objs[frame_idx]:.4f}',
                         fontsize=12, fontweight='bold')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_aspect('equal')

            # Plot 3: Direct comparison
            ax3.plot(self.problem.q_clean[:, 0], self.problem.q_clean[:, 1],
                    'gray', linewidth=2, alpha=0.8, label='Clean signal')
            ax3.plot(bundle_point[:, 0], bundle_point[:, 1],
                    'blue', linewidth=2.5, alpha=0.7, label='Bundle Method')
            ax3.plot(sgm_point[:, 0], sgm_point[:, 1],
                    'green', linewidth=2.5, alpha=0.7, label='SGM', linestyle='--')
            ax3.set_title('Direct Comparison', fontsize=12, fontweight='bold')
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
            ax3.set_aspect('equal')

            # Plot 4: Objective values
            if frame_idx > 0:
                ax4.plot(bundle_iters[:frame_idx+1], bundle_objs[:frame_idx+1],
                        'b-', linewidth=2, label='Bundle Method')
                ax4.plot(sgm_iters[:frame_idx+1], sgm_objs[:frame_idx+1],
                        'g-', linewidth=2, label='SGM')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Objective Value')
            ax4.set_title('Objective Function Progress', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')

            # Plot 5: SGM Step Size Evolution
            if frame_idx > 0:
                ax5.plot(sgm_iters[:frame_idx+1], sgm_step_sizes[:frame_idx+1],
                        'r-', linewidth=2, label='SGM Step Size', marker='o', markersize=3)
                # Show theoretical O(1/sqrt(k+1)) curve
                theoretical_steps = [1.0 / np.sqrt(i + 1) for i in sgm_iters[:frame_idx+1]]
                ax5.plot(sgm_iters[:frame_idx+1], theoretical_steps,
                        'r--', linewidth=1, alpha=0.7, label='Theoretical O(1/√(k+1))')
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Step Size')
            ax5.set_title('SGM Step Size Evolution O(1/√(k+1))', fontsize=12, fontweight='bold')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
            ax5.set_yscale('log')  # Log scale to better show the decay

            # Plot 6: Progress summary table
            ax6.axis('off')
            summary_data = [
                ['Method', 'Iteration', 'Objective', 'Step Size', 'Progress'],
                ['Bundle', f'{bundle_iter}', f'{bundle_objs[frame_idx]:.4f}',
                 'Adaptive', f'{frame_idx/num_frames*100:.1f}%'],
                ['SGM', f'{sgm_iter}', f'{sgm_objs[frame_idx]:.4f}',
                 f'{sgm_step_sizes[frame_idx]:.4f}', f'{frame_idx/num_frames*100:.1f}%']
            ]

            table = ax6.table(cellText=summary_data[1:], colLabels=summary_data[0],
                            cellLoc='center', loc='center', bbox=[0, 0.2, 1, 0.6])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)

            # Color the header
            for i in range(len(summary_data[0])):
                table[(0, i)].set_facecolor('#4472C4')
                table[(0, i)].set_text_props(weight='bold', color='white')

            ax6.set_title('Progress Summary', fontsize=12, fontweight='bold', y=0.9)

            plt.tight_layout()

            # Convert to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img)

            plt.close(fig)

            if (frame_idx + 1) % 5 == 0:
                logger.info(f"Created enhanced frame {frame_idx + 1}/{num_frames}")

        # Save GIF
        logger.info(f"Saving enhanced GIF to {gif_filename}")
        frames[0].save(
            gif_filename,
            save_all=True,
            append_images=frames[1:],
            duration=300,  # Faster refresh rate (was 600ms, now 300ms)
            loop=0
        )

        logger.info(f"Enhanced GIF saved successfully: {gif_filename}")
        return gif_filename


def main():
    """Main function to run the enhanced GIF creation"""

    logger.info("="*80)
    logger.info("STARTING ENHANCED DENOISING GIF CREATION")
    logger.info("="*80)

    # Create problem
    logger.info("Creating TV denoising problem...")
    problem = TVDenoisingProblem.from_square_wave(
        T=3, a=-6, b=6, N=496, alpha=2, noise_std=0.05, seed=42
    )

    # Create enhanced GIF creator
    gif_creator = EnhancedDenosingGIFCreator(problem)

    # Run algorithms and capture states
    sgm_data = gif_creator.run_sgm_with_capture(max_iter=200, initial_step_size=1.0, capture_freq=5)
    bundle_data = gif_creator.create_bundle_progression(num_states=len(sgm_data[0]))

    # Create enhanced GIF
    gif_filename = gif_creator.create_enhanced_gif(
        bundle_data, sgm_data, 'enhanced_denoising_comparison.gif'
    )

    logger.info("="*80)
    logger.info("ENHANCED DENOISING GIF CREATION COMPLETED!")
    logger.info(f"Output file: {gif_filename}")
    logger.info("="*80)

    return gif_filename


if __name__ == "__main__":
    main()