import numpy as np
import sys
sys.path.append('src')
from scipy.optimize import minimize
from scipy.linalg import expm, logm, sqrtm, inv
from test_python_implementation import SPDManifold, RiemannianMedianProblem

def robust_generate_test_data(manifold, num_points, seed=42):
    """
    Generate test data more robustly to avoid numerical issues.
    """
    np.random.seed(seed)

    # Start with a well-conditioned base point
    base_point = np.diag([2.0, 3.0])  # Simple diagonal matrix

    data_points = []

    for i in range(num_points):
        # Generate a small perturbation in the tangent space
        V = np.random.randn(manifold.n, manifold.n) * 0.3  # Smaller perturbations
        V = (V + V.T) / 2  # Make symmetric

        # Use a safer exponential map approximation for small perturbations
        # For small V: exp_X(V) ≈ X + V + X*V/2 + V*X/2 (but ensure SPD)
        try:
            point = manifold.exp_map(base_point, V)

            # Check if the result is actually SPD
            eigenvals = np.linalg.eigvals(point)
            if np.all(eigenvals > 1e-8):  # All eigenvalues positive
                data_points.append(point)
            else:
                # Fallback: create a valid SPD matrix directly
                A = np.random.randn(manifold.n, manifold.n) * 0.5
                point = base_point + A @ A.T + 0.1 * np.eye(manifold.n)
                data_points.append(point)
        except:
            # Fallback if exponential map fails
            A = np.random.randn(manifold.n, manifold.n) * 0.5
            point = base_point + A @ A.T + 0.1 * np.eye(manifold.n)
            data_points.append(point)

    return data_points, base_point

def phase1_find_median(data_points, manifold, max_iter=1000, tol=1e-10):
    """
    Phase 1: Robustly find the Riemannian median using multiple methods.
    """
    print("=== PHASE 1: Finding True Riemannian Median ===")

    median_problem = RiemannianMedianProblem(manifold, data_points)

    # Try multiple starting points and methods
    candidates = []

    # Method 1: Start from each data point
    print("Method 1: Starting from each data point...")
    for i, start_point in enumerate(data_points):
        try:
            result = gradient_descent_robust(median_problem, start_point, max_iter, tol)
            obj_val = median_problem.objective_function(result)
            candidates.append((obj_val, result, f"data_point_{i}"))
            print(f"  From data point {i}: objective = {obj_val:.10f}")
        except Exception as e:
            print(f"  From data point {i}: failed ({e})")

    # Method 2: Start from geometric mean (Karcher mean approximation)
    print("Method 2: Starting from geometric mean...")
    try:
        # Compute geometric mean in log space
        log_sum = np.zeros((manifold.n, manifold.n))
        identity = np.eye(manifold.n)

        for point in data_points:
            log_point = manifold.log_map(identity, point)
            log_sum += log_point

        geom_mean_log = log_sum / len(data_points)
        geom_mean = manifold.exp_map(identity, geom_mean_log)

        result = gradient_descent_robust(median_problem, geom_mean, max_iter, tol)
        obj_val = median_problem.objective_function(result)
        candidates.append((obj_val, result, "geometric_mean"))
        print(f"  From geometric mean: objective = {obj_val:.10f}")
    except Exception as e:
        print(f"  From geometric mean: failed ({e})")

    # Method 3: Start from arithmetic mean (projected to SPD)
    print("Method 3: Starting from arithmetic mean...")
    try:
        arith_mean = np.mean(data_points, axis=0)
        # Ensure it's SPD by adding identity if needed
        eigenvals = np.linalg.eigvals(arith_mean)
        if np.min(eigenvals) <= 0:
            arith_mean += (1e-6 - np.min(eigenvals)) * np.eye(manifold.n)

        result = gradient_descent_robust(median_problem, arith_mean, max_iter, tol)
        obj_val = median_problem.objective_function(result)
        candidates.append((obj_val, result, "arithmetic_mean"))
        print(f"  From arithmetic mean: objective = {obj_val:.10f}")
    except Exception as e:
        print(f"  From arithmetic mean: failed ({e})")

    # Find the best candidate
    if not candidates:
        raise RuntimeError("All Phase 1 methods failed!")

    candidates.sort(key=lambda x: x[0])  # Sort by objective value
    best_obj, best_point, best_method = candidates[0]

    print(f"\nBest result: objective = {best_obj:.10f} (from {best_method})")

    # Verify this is actually a critical point
    grad = median_problem.subgradient_function(best_point)
    grad_norm = manifold.norm(best_point, grad)
    print(f"Gradient norm at best point: {grad_norm:.10f}")

    if grad_norm > 1e-6:
        print(f"WARNING: Gradient norm is large ({grad_norm}), may not be converged!")

    return best_point, best_obj

def gradient_descent_robust(problem, start_point, max_iter=1000, tol=1e-10):
    """
    Robust gradient descent with adaptive step size and backtracking.
    """
    X = start_point.copy()
    manifold = problem.manifold

    step_size = 0.1

    for iter_num in range(max_iter):
        grad = problem.subgradient_function(X)
        grad_norm = manifold.norm(X, grad)

        if grad_norm < tol:
            break

        # Backtracking line search
        current_obj = problem.objective_function(X)
        best_step = step_size
        best_X = None
        best_obj = current_obj

        for alpha in [step_size, step_size/2, step_size/4, step_size/8, step_size/16]:
            try:
                X_new = manifold.exp_map(X, -alpha * grad)
                new_obj = problem.objective_function(X_new)

                if new_obj < best_obj:
                    best_obj = new_obj
                    best_X = X_new
                    best_step = alpha
            except:
                continue

        if best_X is not None:
            X = best_X
            step_size = min(best_step * 1.1, 0.5)  # Adaptive step size
        else:
            step_size *= 0.5  # Reduce step size if no improvement
            if step_size < 1e-12:
                break

    return X

# Test the Phase 1 approach
if __name__ == "__main__":
    print("Testing Phase 1 Median Finding")

    manifold = SPDManifold(2)
    data_points, base_point = robust_generate_test_data(manifold, 5)

    print(f"Base point used for generation:")
    print(base_point)
    print(f"Objective at base point: {RiemannianMedianProblem(manifold, data_points).objective_function(base_point)}")

    print(f"\nGenerated data points:")
    for i, point in enumerate(data_points):
        print(f"Point {i}: eigenvals = {np.linalg.eigvals(point)}")

    # Find the true median
    true_median, true_min_obj = phase1_find_median(data_points, manifold)

    print(f"\n=== PHASE 1 RESULTS ===")
    print(f"True median:")
    print(true_median)
    print(f"True minimum objective: {true_min_obj}")
    print(f"Base point objective: {RiemannianMedianProblem(manifold, data_points).objective_function(base_point)}")
    print(f"Difference: {RiemannianMedianProblem(manifold, data_points).objective_function(base_point) - true_min_obj}")