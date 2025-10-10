"""
Simplified diagnostic to understand why Julia gaps are increasing
"""

using Random
using LinearAlgebra
using Printf

include("src/RiemannianProximalBundle.jl")
include("SPDManifold.jl")
include("RiemannianMedian.jl")

function diagnose_julia_issue()
    println("=" ^ 50)
    println("DIAGNOSING JULIA IMPLEMENTATION")
    println("=" ^ 50)

    # Simple setup
    n = 2
    manifold = SPDManifold(n)

    # Create very simple data - just 2 points around identity
    Random.seed!(42)
    data_points = [
        Matrix{Float64}(I, n, n) * 1.5,
        Matrix{Float64}(I, n, n) * 2.5
    ]

    median_problem = RiemannianMedianProblem(manifold, data_points)

    # Simple true minimum - just use the midpoint
    true_median = Matrix{Float64}(I, n, n) * 2.0
    true_min_obj = objective_function(median_problem, true_median)

    println("Data points:")
    for (i, point) in enumerate(data_points)
        obj = objective_function(median_problem, point)
        println("  Point $i: objective = $obj")
    end
    println("True minimum objective: $true_min_obj")

    # Initial point slightly away
    initial_point = Matrix{Float64}(I, n, n) * 3.0
    initial_objective = objective_function(median_problem, initial_point)
    initial_gap = initial_objective - true_min_obj

    println("\\nInitial setup:")
    println("  Initial objective: $initial_objective")
    println("  True minimum: $true_min_obj")
    println("  Initial gap: $initial_gap")

    if initial_gap <= 0
        println("❌ PROBLEM: Initial gap is not positive!")
        return
    end

    # Create RPB with minimal settings
    function objective_wrapper(X)
        return objective_function(median_problem, X)
    end

    function subgradient_wrapper(X)
        return subgradient_function(median_problem, X)
    end

    function retraction_wrapper(X, V)
        return exp_map(manifold, X, V)
    end

    function transport_wrapper(X, Y, V)
        return V  # Simple transport
    end

    initial_subgradient = subgradient_wrapper(initial_point)

    rpb = RProximalBundle(
        manifold,
        retraction_wrapper,
        transport_wrapper,
        objective_wrapper,
        subgradient_wrapper,
        initial_point,
        initial_objective,
        initial_subgradient;
        true_min_obj=true_min_obj,
        proximal_parameter=0.1,
        trust_parameter=0.2,
        max_iter=3,  # Just a few iterations
        tolerance=1e-8,
        adaptive_proximal=false,  # Turn off adaptive features
        know_minimizer=true
    )

    println("\\nInitial RPB state:")
    println("  Objective history: $(rpb.objective_history)")
    println("  Current center objective: $(objective_wrapper(rpb.current_proximal_center))")

    # Run and monitor each iteration
    for i in 1:3
        println("\\n" * "=" ^ 30)
        println("ITERATION $i")
        println("=" ^ 30)

        # Store state before
        before_center = copy(rpb.current_proximal_center)
        before_obj = objective_wrapper(before_center)
        before_gap = before_obj - true_min_obj

        println("Before iteration $i:")
        println("  Center: $(before_center)")
        println("  Center objective: $before_obj")
        println("  Gap: $before_gap")

        # Manual step through one iteration

        # 1. Fix two-cut model information
        rpb.prev_candidate_direction = length(rpb.candidate_directions) > 0 ? rpb.candidate_directions[end] : nothing
        rpb.prev_true_obj = length(rpb.candidate_obj_history) > 0 ? rpb.candidate_obj_history[end] : nothing
        rpb.prev_model_obj = length(rpb.candidate_model_obj_history) > 0 ? rpb.candidate_model_obj_history[end] : nothing
        rpb.prev_transport_subg = length(rpb.transported_subgradients) > 0 ? rpb.transported_subgradients[end] : nothing
        rpb.prev_model_subg = rpb.prev_candidate_direction !== nothing ? -rpb.prev_prox_parameter * rpb.prev_candidate_direction : nothing
        push!(rpb.model_subg, rpb.prev_model_subg)
        rpb.prev_error_shift = length(rpb.error_shifts) > 0 ? rpb.error_shifts[end] : nothing

        # 2. Compute candidate direction
        candidate_direction = cand_prox_direction(rpb)
        push!(rpb.candidate_directions, candidate_direction)
        println("  Candidate direction norm: $(norm(manifold, rpb.current_proximal_center, candidate_direction))")

        # 3. Retract to manifold
        candidate_point = rpb.retraction_map(rpb.current_proximal_center, candidate_direction)

        # 4. Compute objectives
        model_objective = model_evaluation(rpb, candidate_direction)
        push!(rpb.candidate_model_obj_history, model_objective)

        candidate_objective = rpb.compute_objective(candidate_point)
        push!(rpb.candidate_obj_history, candidate_objective)

        current_objective = rpb.compute_objective(rpb.current_proximal_center)

        println("  Current objective: $current_objective")
        println("  Candidate objective: $candidate_objective")
        println("  Model objective: $model_objective")

        # 5. Compute ratio
        ratio = model_versus_true(rpb, candidate_objective, model_objective, current_objective)
        println("  Ratio: $ratio")
        println("  Trust parameter: $(rpb.trust_parameter)")

        # 6. Step decision
        new_subgradient = rpb.compute_subgradient(candidate_point)

        if ratio > rpb.trust_parameter
            println("  -> DESCENT STEP")
            rpb.current_proximal_center = candidate_point
            push!(rpb.proximal_center_history, candidate_point)
            rpb.subgradient_at_center = new_subgradient

            push!(rpb.untransported_subgradients, new_subgradient)
            push!(rpb.transported_subgradients, new_subgradient)
            push!(rpb.error_shifts, 0.0)

            rpb.single_cut = true
            push!(rpb.indices_of_descent_steps, i)
            push!(rpb.proximal_parameter_history, rpb.proximal_parameter)
        else
            println("  -> NULL STEP")
            transported_subg = rpb.transport_map(candidate_point, rpb.current_proximal_center, new_subgradient)

            push!(rpb.untransported_subgradients, new_subgradient)
            push!(rpb.transported_subgradients, transported_subg)
            push!(rpb.error_shifts, 0.0)  # Simplified

            rpb.single_cut = false
            push!(rpb.indices_of_null_steps, i)
            push!(rpb.proximal_parameter_history, rpb.proximal_parameter)
        end

        # 7. Update objective history
        current_proximal_objective = rpb.compute_objective(rpb.current_proximal_center)
        gap = current_proximal_objective - rpb.true_min_obj
        push!(rpb.objective_history, gap)
        push!(rpb.raw_objective_history, current_proximal_objective)

        println("After iteration $i:")
        println("  New center: $(rpb.current_proximal_center)")
        println("  New center objective: $current_proximal_objective")
        println("  New gap: $gap")
        println("  Gap change: $(gap - before_gap)")

        if gap > before_gap
            println("  ❌ GAP INCREASED!")
        else
            println("  ✅ Gap decreased")
        end

        # Break if we find an increasing gap to analyze
        if gap > before_gap
            println("\\n🔍 ANALYSIS OF GAP INCREASE:")
            println("  Before gap: $before_gap")
            println("  After gap: $gap")
            println("  Center moved from $(before_center) to $(rpb.current_proximal_center)")
            println("  Step type was: $(ratio > rpb.trust_parameter ? "DESCENT" : "NULL")")
            break
        end
    end

    println("\\nFinal objective history: $(rpb.objective_history)")
    return rpb
end

if abspath(PROGRAM_FILE) == @__FILE__
    rpb = diagnose_julia_issue()
end