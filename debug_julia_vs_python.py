"""
Debug script to trace why Julia implementation gives increasing gaps while Python gives decreasing gaps
"""

import numpy as np
from pymanopt.manifolds import SymmetricPositiveDefinite
import sys
sys.path.append('src')
from RiemannianProximalBundle import RProximalBundle

def debug_run_python():
    """Run Python version with detailed debugging"""
    print("=" * 50)
    print("DEBUGGING PYTHON IMPLEMENTATION")
    print("=" * 50)

    # Use same setup as corrected version
    dim = 2
    manifold = SymmetricPositiveDefinite(dim)

    # Generate data
    np.random.seed(42)
    base_point = manifold.random_point()
    data = []
    for i in range(3):  # Small test
        tangent = manifold.random_tangent_vector(base_point)
        tangent /= manifold.norm(base_point, tangent)
        tangent *= 0.8
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

    # Find true minimum quickly
    true_median = base_point  # Approximate
    true_min_obj = cost(true_median)
    print(f"True minimum objective: {true_min_obj}")

    # Set up initial point
    np.random.seed(123)
    initial_point = manifold.random_point()
    initial_objective = cost(initial_point)
    initial_gap = initial_objective - true_min_obj
    print(f"Initial objective: {initial_objective}")
    print(f"Initial gap: {initial_gap}")

    # Run with debug
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

    print(f"\\nStarting RPB with initial gap: {rpb.objective_history[0]}")

    # Manual iteration with debugging
    for i in range(3):
        print(f"\\n--- ITERATION {i+1} ---")

        # Store state before iteration
        before_center = rpb.current_proximal_center.copy()
        before_obj = rpb.compute_objective(before_center)
        before_gap = before_obj - true_min_obj

        print(f"Before iteration:")
        print(f"  Center objective: {before_obj}")
        print(f"  Gap: {before_gap}")

        # Run one iteration manually by calling internal methods
        # This mimics what happens in rpb.run() but with debugging

        # Compute candidate direction
        candidate_direction = rpb.cand_prox_direction()
        rpb.candidate_directions.append(candidate_direction)

        # Retract to manifold
        candidate_point = rpb.retraction_map(rpb.current_proximal_center, candidate_direction)

        # Compute objectives
        model_objective = rpb.model_evaluation(candidate_direction)
        rpb.candidate_model_obj_history.append(model_objective)

        candidate_objective = rpb.compute_objective(candidate_point)
        rpb.candidate_obj_history.append(candidate_objective)

        current_objective = rpb.compute_objective(rpb.current_proximal_center)

        # Compute ratio
        ratio = rpb.model_versus_true(candidate_objective, model_objective, current_objective)

        print(f"  Candidate objective: {candidate_objective}")
        print(f"  Model objective: {model_objective}")
        print(f"  Ratio: {ratio}")
        print(f"  Trust parameter: {rpb.trust_parameter}")

        # Step decision
        if ratio > rpb.trust_parameter:
            print("  -> DESCENT STEP")
            rpb.current_proximal_center = candidate_point
            rpb.proximal_center_history.append(candidate_point)

            new_subgradient = rpb.compute_subgradient(candidate_point)
            rpb.subgradient_at_center = new_subgradient

            rpb.untransported_subgradients.append(new_subgradient)
            rpb.transported_subgradients.append(new_subgradient)
            rpb.error_shifts.append(0)

            rpb.single_cut = True
            rpb.indices_of_descent_steps.append(i)
            rpb.proximal_parameter_history.append(rpb.proximal_parameter)
        else:
            print("  -> NULL or DOUBLING STEP")
            # For simplicity, just do null step
            new_subgradient = rpb.compute_subgradient(candidate_point)
            transported_subg = rpb.transport_map(candidate_point, rpb.current_proximal_center, new_subgradient)

            rpb.untransported_subgradients.append(new_subgradient)
            rpb.transported_subgradients.append(transported_subg)
            rpb.error_shifts.append(0)  # Simplified

            rpb.single_cut = False
            rpb.indices_of_null_steps.append(i)
            rpb.proximal_parameter_history.append(rpb.proximal_parameter)

        # Update objective history
        current_proximal_objective = rpb.compute_objective(rpb.current_proximal_center)
        gap = current_proximal_objective - rpb.true_min_obj
        rpb.objective_history.append(gap)
        rpb.raw_objective_history.append(current_proximal_objective)

        print(f"After iteration:")
        print(f"  New center objective: {current_proximal_objective}")
        print(f"  New gap: {gap}")
        print(f"  Gap change: {gap - before_gap}")

        if gap > before_gap:
            print("  ⚠️  GAP INCREASED!")
        else:
            print("  ✅ Gap decreased or stayed same")

    return rpb

if __name__ == "__main__":
    rpb = debug_run_python()
    print(f"\\nFinal objective history: {rpb.objective_history}")
    print(f"All gaps decreasing: {all(rpb.objective_history[i] >= rpb.objective_history[i+1] for i in range(len(rpb.objective_history)-1))}")