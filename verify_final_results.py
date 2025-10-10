"""
Quick verification that the final implementations produce positive objective gaps
"""

import numpy as np
from pymanopt.manifolds import SymmetricPositiveDefinite
import sys
sys.path.append('src')
from RiemannianProximalBundle import RProximalBundle

def quick_test():
    print("=== QUICK VERIFICATION ===")

    # Use the corrected approach
    dim = 2
    manifold = SymmetricPositiveDefinite(dim)

    # Generate data
    np.random.seed(42)
    base_point = manifold.random_point()
    data = []
    for i in range(3):  # Smaller test for speed
        tangent = manifold.random_tangent_vector(base_point)
        tangent /= manifold.norm(base_point, tangent)
        tangent *= 0.5
        data.append(manifold.exp(base_point, tangent))

    # Define functions
    def cost(point):
        return sum([manifold.dist(point, x) for x in data]) / len(data)

    def subgradient(point):
        import autograd.numpy as anp
        grad = anp.zeros_like(point)
        for x in data:
            log_point = manifold.log(point, x)
            norm_log = manifold.norm(point, log_point)
            if norm_log > 1e-12:
                grad += -(log_point) / norm_log
        return grad / len(data)

    # Find true minimum (simple approach)
    candidates = []
    for start in data:
        X = start.copy()
        for _ in range(100):
            grad = subgradient(X)
            if manifold.norm(X, grad) < 1e-8:
                break
            X = manifold.exp(X, -0.01 * grad)
        candidates.append((cost(X), X))

    true_min_obj, true_median = min(candidates, key=lambda x: x[0])

    # Test proximal bundle with small number of iterations
    initial_point = manifold.random_point()
    initial_objective = cost(initial_point)
    initial_gap = initial_objective - true_min_obj

    print(f"Initial gap: {initial_gap}")
    print(f"Gap is positive: {initial_gap > 0}")

    if initial_gap <= 0:
        print("❌ PROBLEM: Initial gap is not positive!")
        return False

    # Quick RPB test
    rpb = RProximalBundle(
        manifold=manifold,
        retraction_map=manifold.exp,
        transport_map=manifold.transport,
        objective_function=cost,
        subgradient=subgradient,
        initial_point=initial_point,
        initial_objective=initial_objective,
        initial_subgradient=subgradient(initial_point),
        true_min_obj=true_min_obj,
        max_iter=5,  # Just a few iterations
        know_minimizer=True
    )

    rpb.run()

    print(f"Objective history: {rpb.objective_history}")
    all_positive = all(gap > 0 for gap in rpb.objective_history)
    print(f"All gaps positive: {all_positive}")

    if all_positive:
        print("✅ SUCCESS: All objective gaps are positive!")
        return True
    else:
        print("❌ PROBLEM: Some gaps are negative!")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 VERIFICATION PASSED: Implementations produce correct positive gaps!")
    else:
        print("\n💥 VERIFICATION FAILED: Issues with gap calculation!")