using LinearAlgebra
using Plots

# Export main functions
export RProximalBundle, run!, plot_objective_versus_iter, plot_transport_constant_versus_iter, plot_model_proximal_gap, print_last_step_type

"""
    RProximalBundle{T <: AbstractFloat}

Riemannian proximal bundle algorithm for non-convex optimization on Riemannian manifolds.
This struct implements the proximal bundle algorithm for non-convex optimization on Riemannian manifolds.
The algorithm is based on the proximal bundle method for convex optimization, which is a generalization
of the bundle method for non-convex optimization.
This implementation centers on a two-cut surrogate model, however, one may easily replace with other
convex surrogates, satisfying certain properties.

# Type Parameters
- `T`: Floating point precision type (e.g., Float32, Float64, BigFloat)
"""
mutable struct RProximalBundle{T <: AbstractFloat}
    # Riemannian Optimization Tools
    manifold::Any
    retraction_map::Function
    transport_map::Function

    # Parameters for first-order retraction/transport
    retraction_error::T
    transport_error::T
    sectional_curvature::T

    # Maximum proximal parameter
    max_rho::T

    # Explicit Optimization Oracles
    compute_objective::Function
    compute_subgradient::Function

    # State Variables
    current_proximal_center::Any  # where model lives during constructions
    subgradient_at_center::Any    # g_{k}, subgradient at proximal center
    proximal_parameter::T   # current ρ_{k}
    single_cut::Bool             # flag for if the model is single-cut 
    two_cut::Bool           # flag for if the model is two-cut
    back_tracking_factor::T # factor for backtracking proximal parameter

    initial_objective::T   # f(x_{0})
    max_iter::Int               # maximum number of iterations
    tolerance::T          # tolerance for convergence
    numerical_tolerance::T      # tolerance for numerical precision (shift adjustment)
    trust_parameter::T    # pre-specified beta

    # Current Model Information Storage
    anchor_cut_subg::Any        # g_{k}, subgradient at proximal center for anchor cut
    anchor_cut_shift::T       # f(x_{k}), objective at proximal center for anchor cut
    model_cut_subg::Any         # s_{k}, subgradient for model cut
    model_cut_shift::T        # f_{t-1}(d_{t}), model objective at candidate direction for model cut
    new_cut_subg::Any           # g_{t}, subgradient for new cut
    new_cut_shift::T          # f(R_x(d_{t+1})), true objective at candidate point for new cut

    # Toggles for variants
    adaptive_proximal::Bool     # flag for if the proximal parameter is adaptively updated - unused, remnant from earlier versions
    know_minimizer::Bool        # flag for if the true minimizer is known
    relative_error::Bool        # flag for if the error to store should be taken with the initial objective as a denominator

    # Storage for two-cut model - setting aside memory
    model_subg::Vector{Any}                    # s_{k}, initialize together to line-up indexes, dummy entry
    untransported_subgradients::Vector{Any}    # g_{k}
    transported_subgradients::Vector{Any}      # ĝ_{k}
    candidate_directions::Vector{Any}          # d_{k}
    candidate_obj_history::Vector{T}     # f(R_x(x_{k}))
    candidate_model_obj_history::Vector{T}  # f_k(d_{k})
    error_shifts::Vector{T}              # e_{f_k}(ρ_{k})
    proximal_center_history::Vector{Any}       # x_{k}

    # Model information for two-cut model evaluation
    prev_candidate_direction::Union{Nothing, Any}
    prev_true_obj::Union{Nothing, T}
    prev_model_obj::Union{Nothing, T}
    prev_transport_subg::Union{Nothing, Any}
    prev_model_subg::Union{Nothing, Any}
    prev_error_shift::Union{Nothing, T}
    prev_prox_parameter::T

    # Storage for algorithm run
    proximal_parameter_history::Vector{T}
    relative_objective_history::Vector{T}
    objective_history::Vector{T}          # Store gaps for visualization
    raw_objective_history::Vector{T}      # Store raw objectives for algorithm logic
    true_min_obj::T
    indices_of_descent_steps::Vector{Int}
    indices_of_null_steps::Vector{Int}
    indices_of_proximal_doubling_steps::Vector{Int}
    iteration::Vector{Int}                     # Store iteration numbers
    potential_error_shift::T

    debugging::Bool          # Debugging flag
    transport_constant_approx::Vector{T}      # Placeholder for transport constant approximation
    objective_increase_flags::Vector{Bool}    # Flag when f(x_k) < f(x_{k+1}) at proximal centers
    descent_step_ratios::Vector{T}           # Store ratio values for descent steps
    descent_step_numerators::Vector{T}       # Store numerator values for descent steps
    descent_step_denominators::Vector{T}     # Store denominator values for descent steps
    descent_step_iterations::Vector{Int}     # Store which iterations had descent steps
    model_proximal_gap_history::Vector{T}    # Store model proximal gap values for each iteration
    last_step_type::Symbol                   # Track the type of the last step (:descent, :null, :doubling, or :initial)

    function RProximalBundle{T}(manifold, retraction_map, transport_map, objective_function,
                            subgradient, initial_point, initial_objective::T, initial_subgradient;
                            true_min_obj::T=T(0.0), retraction_error::T=T(0.0), transport_error::T=T(0.0),
                            sectional_curvature::T=T(-1.0), proximal_parameter::T=T(0.02),
                            trust_parameter::T=T(0.1), max_iter::Int=5000, tolerance::T=T(1e-12),
                            numerical_tolerance::T=T(1e-14), max_rho::T=T(1e8), back_tracking_factor::T=T(2.0), adaptive_proximal::Bool=true, know_minimizer::Bool=true, relative_error::Bool=true,
                            debugging::Bool=false) where T <: AbstractFloat

        # Initialize storage arrays
        model_subg = [initial_subgradient]
        untransported_subgradients = [initial_subgradient]
        transported_subgradients = [initial_subgradient]
        candidate_directions = Any[]
        candidate_obj_history = T[initial_objective]
        candidate_model_obj_history = T[]
        error_shifts = T[T(0.0)]
        proximal_center_history = [initial_point]

        # Initialize history arrays
        proximal_parameter_history = T[proximal_parameter]
        relative_objective_history = T[(initial_objective - true_min_obj) / (initial_objective - true_min_obj)]
        objective_history = T[initial_objective - true_min_obj]
        raw_objective_history = T[initial_objective]

        # Initialize index tracking arrays
        indices_of_descent_steps = Int[]
        indices_of_null_steps = Int[]
        indices_of_proximal_doubling_steps = Int[]

        # Initialize iteration array with starting value
        iteration = [0]

        # Initialize debugging arrays
        transport_constant_approx = T[]
        objective_increase_flags = Bool[]
        descent_step_ratios = T[]
        descent_step_numerators = T[]
        descent_step_denominators = T[]
        descent_step_iterations = Int[]
        model_proximal_gap_history = T[]

        new{T}(manifold, retraction_map, transport_map,
            retraction_error, transport_error, sectional_curvature, max_rho,
            objective_function, subgradient,
            initial_point, initial_subgradient, proximal_parameter, true, false, back_tracking_factor,
            initial_objective, max_iter, tolerance, numerical_tolerance, trust_parameter,
            initial_subgradient, initial_objective, nothing, T(0.0), nothing, T(0.0),
            adaptive_proximal, know_minimizer, relative_error,
            model_subg, untransported_subgradients, transported_subgradients,
            candidate_directions, candidate_obj_history, candidate_model_obj_history,
            error_shifts, proximal_center_history,
            nothing, nothing, nothing, nothing, nothing, nothing, proximal_parameter,
            proximal_parameter_history, relative_objective_history, objective_history,
            raw_objective_history, true_min_obj,
            indices_of_descent_steps, indices_of_null_steps, indices_of_proximal_doubling_steps,
            iteration, T(0.0),
            debugging, transport_constant_approx, objective_increase_flags,
            descent_step_ratios, descent_step_numerators, descent_step_denominators, descent_step_iterations,
            model_proximal_gap_history, :initial)
    end
end

# Convenience constructor that defaults to Float64 when no type is specified
function RProximalBundle(manifold, retraction_map, transport_map, objective_function,
                        subgradient, initial_point, initial_objective, initial_subgradient;
                        kwargs...)
    T = typeof(initial_objective)
    return RProximalBundle{T}(manifold, retraction_map, transport_map, objective_function,
                            subgradient, initial_point, initial_objective, initial_subgradient;
                            kwargs...)
end

"""
    run!(rpb::RProximalBundle{T}) where T

Run the proximal bundle algorithm.
"""
function run!(rpb::RProximalBundle{T}) where T
    for i in 1:rpb.max_iter
        # Increment iteration counter
        push!(rpb.iteration, i)

        # --- Remnant code from earlier versions - IGNORE ---
        # # Fix two-non-anchor cuts before any updates
        # rpb.prev_candidate_direction = length(rpb.candidate_directions) > 0 ? rpb.candidate_directions[end] : nothing
        # rpb.prev_true_obj = length(rpb.candidate_obj_history) > 0 ? rpb.candidate_obj_history[end] : nothing
        # rpb.prev_model_obj = length(rpb.candidate_model_obj_history) > 0 ? rpb.candidate_model_obj_history[end] : nothing
        # rpb.prev_transport_subg = length(rpb.transported_subgradients) > 0 ? rpb.transported_subgradients[end] : nothing
        # rpb.prev_model_subg = length(rpb.model_subg) > 0 ? rpb.model_subg[end] : nothing
        # rpb.prev_error_shift = length(rpb.error_shifts) > 0 ? rpb.error_shifts[end] : nothing

        # Backtracking to ensure proximal parameter satisfies a descent or null condition - recover candidate iterate
        candidate_point, candidate_direction, descent_flag, new_subgradient, transported_new_subgradient, potential_error_shift, ratio = backtracking_procedure(rpb)

        # Cache current objective for consistent recording
        current_objective = rpb.compute_objective(rpb.current_proximal_center)

        candidate_objective = rpb.compute_objective(candidate_point)
        candidate_model_objective = model_evaluation(rpb, candidate_direction)

        # Accept rule
        if descent_flag  # DESCENT STEP
            # Store ratio information for debugging (before updating state)
            if rpb.debugging
                candidate_objective = rpb.candidate_obj_history[end]
                model_objective = rpb.candidate_model_obj_history[end]
                numerator = current_objective - candidate_objective
                denominator = current_objective - model_objective
                push!(rpb.descent_step_ratios, ratio)
                push!(rpb.descent_step_numerators, numerator)
                push!(rpb.descent_step_denominators, denominator)
                push!(rpb.descent_step_iterations, i)
            end

            # Update current proximal center
            rpb.current_proximal_center = candidate_point  
            push!(rpb.proximal_center_history, candidate_point) 

            # Update and change to single cut model 
            rpb.anchor_cut_subg = new_subgradient  # g_{k+1}
            rpb.anchor_cut_shift = candidate_objective  # f(x_{k+1})
            rpb.single_cut = true
            rpb.two_cut = false

            # Update model information - one-cut model now! --- Remove this when cleaning up code ---
            push!(rpb.untransported_subgradients, new_subgradient)  # g_{k+1}
            push!(rpb.transported_subgradients, transported_new_subgradient)    # no transport is done
            push!(rpb.error_shifts, T(0.0))  # e_{f_k}(ρ_{k+1}) = 0, no transport is done
 
            push!(rpb.indices_of_descent_steps, i)
            push!(rpb.proximal_parameter_history, rpb.proximal_parameter)

            # Update last step type
            rpb.last_step_type = :descent

        else  # NULL STEP
            # Change sinle-cut to two-cut, two-cut to three-cut
            if rpb.single_cut
                rpb.two_cut = true  # switch to two-cut model if previously single-cut
                rpb.single_cut = false
            end
            if rpb.two_cut 
                rpb.two_cut = false  # switch to three-cut model if previously two-cut
            end

            # Update model information for null step
            if rpb.two_cut # if two-cut model, no aggregation needed
                # adding new cut
                rpb.new_cut_subg = transported_new_subgradient
                rpb.new_cut_shift = candidate_objective - inner_product(rpb.manifold, rpb.current_proximal_center, transported_new_subgradient, candidate_direction) - potential_error_shift
            
            else # if three-cut, aggregate previous model cut
                # aggregation to model cut
                new_model_cut = - (rpb.proximal_parameter * candidate_direction)
                rpb.model_cut_subg =  new_model_cut
                rpb.model_cut_shift = candidate_model_objective - inner_product(rpb.manifold, rpb.current_proximal_center, new_model_cut, candidate_direction)
                if rpb.model_cut_shift > rpb.anchor_cut_shift
                    if abs(rpb.model_cut_shift - rpb.anchor_cut_shift) < T(1e-12)
                        rpb.model_cut_shift = rpb.anchor_cut_shift
                    else
                        print("Warning: Model cut shift greater than proximal center objective - ERROR\n")
                        print("Model cut shift: $(rpb.model_cut_shift), Proximal center objective: $(rpb.anchor_cut_shift)\n")
                    end
                end

                # adding new cut
                rpb.new_cut_subg = transported_new_subgradient
                rpb.new_cut_shift = candidate_objective - inner_product(rpb.manifold, rpb.current_proximal_center, transported_new_subgradient, candidate_direction) - potential_error_shift

                if rpb.new_cut_shift > rpb.anchor_cut_shift
                    print("Warning: New cut shift greater than proximal center objective - ERROR\n")
                    print("New cut shift: $(rpb.new_cut_shift), Proximal center objective: $(rpb.anchor_cut_shift)\n")
                    print("Error shift applied: $(potential_error_shift)\n")
                    print("Proximal parameter: $(rpb.proximal_parameter)\n")
                end
            end

            # updates model for two non-anchor cuts --- remove when cleaning up code ---
            push!(rpb.untransported_subgradients, new_subgradient)  # g_{k+1}
            push!(rpb.transported_subgradients, transported_new_subgradient)   # ĝ_{k+1}
            push!(rpb.model_subg, -(rpb.proximal_parameter * candidate_direction))                     # s_{k+1}
            push!(rpb.error_shifts, potential_error_shift)  # conservative shift adjustment

            push!(rpb.proximal_parameter_history, rpb.proximal_parameter)
            push!(rpb.indices_of_null_steps, i)

            # Update last step type
            rpb.last_step_type = :null
        end

        # Store objective at current proximal center after each iteration (regardless of step type)
        current_proximal_objective = rpb.compute_objective(rpb.current_proximal_center)
        push!(rpb.relative_objective_history, (current_proximal_objective - rpb.true_min_obj) / (rpb.initial_objective - rpb.true_min_obj))
        push!(rpb.objective_history, current_proximal_objective - rpb.true_min_obj)
        push!(rpb.raw_objective_history, current_proximal_objective)

        # Store model proximal gap for each iteration
        current_model_gap = compute_model_proximal_gap(rpb, candidate_direction, candidate_model_objective)
        push!(rpb.model_proximal_gap_history, current_model_gap)

        # Store current proximal center at each iteration for complete record
        if length(rpb.proximal_center_history) <= i
            push!(rpb.proximal_center_history, rpb.current_proximal_center)
        end

        # Check for objective increase at proximal center: f(x_k) < f(x_{k+1})
        if length(rpb.raw_objective_history) > 1
            prev_proximal_objective = rpb.raw_objective_history[end-1]
            current_proximal_objective = rpb.raw_objective_history[end]
            objective_increased = prev_proximal_objective < current_proximal_objective
            push!(rpb.objective_increase_flags, objective_increased)

            if objective_increased && rpb.debugging
                println("Warning: Objective increased at proximal center from $(prev_proximal_objective) to $(current_proximal_objective) at iteration $i")
                # Find the most recent descent step that led to this objective increase
                if length(rpb.descent_step_iterations) > 0
                    # The most recent descent step should be the last one recorded
                    last_descent_idx = length(rpb.descent_step_iterations)
                    last_descent_iter = rpb.descent_step_iterations[last_descent_idx]
                    last_ratio = rpb.descent_step_ratios[last_descent_idx]
                    last_numerator = rpb.descent_step_numerators[last_descent_idx]
                    last_denominator = rpb.descent_step_denominators[last_descent_idx]

                    println("  Increase at descent step at iteration $(last_descent_iter):")
                    println("  Ratio that triggered descent: $(last_ratio)")
                    println("  Numerator (true gap): $(last_numerator)")
                    println("  Denominator (model gap): $(last_denominator)")
                    println(" Proximal parameter at that step: $(rpb.proximal_parameter_history[last_descent_iter + 1])")  # +1 due to initial entry
                end
            end
        else
            # First iteration, no comparison possible
            push!(rpb.objective_increase_flags, false)
        end

        # Check for convergence
        if rpb.know_minimizer
            if abs(rpb.compute_objective(rpb.current_proximal_center) - rpb.true_min_obj) < rpb.tolerance
                println("Converged to true minimum.")
                break
            end
        else
            # Look at objective at previous descent step and descent step before that
            if length(rpb.indices_of_descent_steps) > 1
                prev_descent_step = rpb.indices_of_descent_steps[end]
                prev_prev_descent_step = rpb.indices_of_descent_steps[end-1]

                if abs(rpb.objective_history[prev_descent_step] - rpb.objective_history[prev_prev_descent_step]) < rpb.tolerance
                    println("Converged to local minimum.")
                    break
                end
            end
        end
    end
end

"""
    print_last_step_type(rpb::RProximalBundle)

Print the type of the last step taken in the algorithm.
At iteration k, this shows whether iteration k-1 was a descent, null, or doubling step.
"""
function print_last_step_type(rpb::RProximalBundle)
    current_iter = length(rpb.iteration) > 0 ? rpb.iteration[end] : 0

    if current_iter == 0
        println("No steps have been taken yet. Current step type: $(rpb.last_step_type)")
    else
        step_description = if rpb.last_step_type == :descent
            "descent step (ratio > trust_parameter, model moved to candidate point)"
        elseif rpb.last_step_type == :null
            "null step (ratio ≤ trust_parameter, model updated but not moved)"
        elseif rpb.last_step_type == :doubling
            "proximal parameter doubling step (proximal gap check failed)"
        elseif rpb.last_step_type == :initial
            "initial state (no step taken)"
        else
            "unknown step type"
        end

        println("At iteration $current_iter:")
        println("  Last step (iteration $(current_iter-1)) was a $(step_description)")
        println("  Step type: $(rpb.last_step_type)")
    end
end

## Helper functions for proximal bundle algorithm run

"""
    model_evaluation(rpb::RProximalBundle{T}, direction) where T

Compute the cut surrogate model.
"""
function model_evaluation(rpb::RProximalBundle{T}, direction) where T
    if rpb.single_cut
        # Computes the model objective using the single-cut model: f_k(d_{k}) = f(x_{k}) + ⟨g_{k}, d_{k}⟩
        return rpb.anchor_cut_shift + inner_product(rpb.manifold, rpb.current_proximal_center, rpb.anchor_cut_subg, direction)
    end

    if rpb.two_cut
        # Computes on "new" cut
        new_cut_obj = rpb.new_cut_shift + inner_product(rpb.manifold, rpb.current_proximal_center, rpb.new_cut_subg, direction)

        # Computes on "anchor" cut
        anchor_cut_obj = rpb.anchor_cut_shift + inner_product(rpb.manifold, rpb.current_proximal_center, rpb.anchor_cut_subg, direction)

        # Returns the maximum of the two cuts
        return max(new_cut_obj, anchor_cut_obj)
    end

    # Computes on "new" cut
    new_cut_obj = rpb.new_cut_shift + inner_product(rpb.manifold, rpb.current_proximal_center, rpb.new_cut_subg, direction)

    # Computes on "model" cut
    model_cut_obj = rpb.model_cut_shift + inner_product(rpb.manifold, rpb.current_proximal_center, rpb.model_cut_subg, direction)

    # Computes on "anchor" cut
    anchor_cut_obj = rpb.anchor_cut_shift + inner_product(rpb.manifold, rpb.current_proximal_center, rpb.anchor_cut_subg, direction)

    if rpb.new_cut_shift > rpb.anchor_cut_shift
        print("New cut shift greater than proximal center - ERROR\n")
    end

    if rpb.model_cut_shift > rpb.anchor_cut_shift
        print("Model cut shift greater than proximal center - ERROR\n")
    end

    # Returns the maximum of the three cuts
    return max(new_cut_obj, model_cut_obj, anchor_cut_obj)
end

"""
    cand_prox_direction(rpb::RProximalBundle)

Compute proximal direction due to model.
"""
function cand_prox_direction(rpb::RProximalBundle{T}) where T
    if rpb.single_cut
        # Computes the proximal direction using the single-cut model: -(g_{k}/ρ_{k}) explicit
        return -(T(1) / rpb.proximal_parameter) * rpb.anchor_cut_subg
    end
    
    if rpb.two_cut
        # Use stored two-cut model information to compute proximal direction
        # Check for Nothing values and handle them

        # if rpb.prev_transport_subg === nothing || rpb.prev_model_subg === nothing ||
        #    rpb.prev_true_obj === nothing || rpb.prev_error_shift === nothing || rpb.prev_model_obj === nothing
        #     # Fallback to single-cut if two-cut data is incomplete
        #     return -(rpb.untransported_subgradients[end] / rpb.proximal_parameter)
        # end

        # numerator = rpb.proximal_parameter * (rpb.prev_true_obj - rpb.prev_error_shift - rpb.prev_model_obj)
        # denominator = (norm(rpb.manifold, rpb.current_proximal_center, rpb.prev_transport_subg - rpb.prev_model_subg))^2

        # # Avoid division by zero
        # if denominator < T(1e-12)
        #     println("Warning: Denominator in convex combination calculation is too small; reverting to single-cut proximal direction.")
        #     return -(rpb.untransported_subgradients[end] / rpb.proximal_parameter)
        # end

        # convex_comb_arg = numerator / denominator
        # convex_comb = min(T(1), max(T(0), convex_comb_arg))  # Ensure convex_comb is in [0,1]

        # # Computes the proximal direction - convex combination of two subg
        # return -(T(1) / rpb.proximal_parameter) * (convex_comb * rpb.prev_transport_subg + (T(1) - convex_comb) * rpb.prev_model_subg)

        # Gather necessary data for two-cut proximal computation
        a_1 = rpb.new_cut_subg  
        b_1 = rpb.new_cut_shift

        a_2 = rpb.anchor_cut_subg
        b_2 = rpb.anchor_cut_shift

        # Compute numerator and denominator for convex combination
        numerator = rpb.proximal_parameter * (b_1 - b_2) + inner_product(rpb.manifold, rpb.current_proximal_center, a_2, a_2) - inner_product(rpb.manifold, rpb.current_proximal_center, a_1, a_2)

        denominator = inner_product(rpb.manifold, rpb.current_proximal_center, a_1, a_1) + inner_product(rpb.manifold, rpb.current_proximal_center, a_2, a_2) - 2 * inner_product(rpb.manifold, rpb.current_proximal_center, a_1, a_2)

        # Avoid division by zero
        if abs(denominator) < T(1e-12)
            println("Warning: Denominator in convex combination calculation is too small; reverting to single-cut proximal direction.")
            return -(rpb.anchor_cut_subg / rpb.proximal_parameter)
        end
        convex_comb_arg = numerator / denominator
        convex_comb = min(T(1), max(T(0), convex_comb_arg))  # Ensure convex_comb is in [0,1]

        # Computes the proximal direction - convex combination of two subg
        return -(T(1) / rpb.proximal_parameter) * (convex_comb * a_1 + (T(1) - convex_comb) * a_2)
    end

    # Compute proximal direction for three-cut model
    return compute_three_cut_proximal(rpb)
    
end

"""
    compute_three_cut_proximal(rpb::RProximalBundle)

Compute proximal direction for three-cut model.
"""
function compute_three_cut_proximal(rpb::RProximalBundle{T}) where T
    # To compute this, we appeal to the dual formulation of the proximal problem, and solve the constrained
    # quadratic program in three variables (the dual multipliers). This can then be done explicitly by checking 
    # all possible active sets (since there are only three variables) - avoiding the need for a QP solver.
    
    # step 0: pre-compute gram matrix and b vector, and define the dual objective function
    # defining the cuts from known information
    a_1 = rpb.anchor_cut_subg
    b_1 = rpb.anchor_cut_shift
    a_2 = rpb.model_cut_subg
    b_2 = rpb.model_cut_shift
    a_3 = rpb.new_cut_subg
    b_3 = rpb.new_cut_shift
    
    # Compute Gram matrix directly using inner products on the manifold
    local G
    try
        G = zeros(T, 3, 3)
        G[1, 1] = inner_product(rpb.manifold, rpb.current_proximal_center, a_1, a_1)
        G[1, 2] = inner_product(rpb.manifold, rpb.current_proximal_center, a_1, a_2)
        G[1, 3] = inner_product(rpb.manifold, rpb.current_proximal_center, a_1, a_3)
        G[2, 1] = G[1, 2]  # Symmetric
        G[2, 2] = inner_product(rpb.manifold, rpb.current_proximal_center, a_2, a_2)
        G[2, 3] = inner_product(rpb.manifold, rpb.current_proximal_center, a_2, a_3)
        G[3, 1] = G[1, 3]  # Symmetric
        G[3, 2] = G[2, 3]  # Symmetric
        G[3, 3] = inner_product(rpb.manifold, rpb.current_proximal_center, a_3, a_3)
    catch e
        # Fallback: use single-cut proximal direction if Gram matrix construction fails
        return -(rpb.anchor_cut_subg / rpb.proximal_parameter)
    end

    # form vector b from b_1, b_2, b_3
    b = [b_1; b_2; b_3]
    
    # dual objective function
    function dual_objective(lambda)
        return dot(b, lambda) - (1/(2*rpb.proximal_parameter)) * dot(lambda, G * lambda)
    end

    # collect valid candidates for lambda
    candidates = Vector{Vector{T}}()

    # step 1:check candidates by looking at possible active sets on simplex constraints: 
    #   vertices, edges, and interior

    # --- step 1a: vertices (1, 0, 0), (0, 1, 0), (0, 0, 1) ---
    for i in 1:3
        lambda = zeros(T, 3)
        lambda[i] = T(1)
        push!(candidates, lambda)
    end

    # --- step 1b: Edges (Exactly two variables > 0) ---
    # We solve for λ_i + λ_j = 1, λ_k = 0
    edge_pairs = [(1, 2), (2, 3), (1, 3)]
    for (i, j) in edge_pairs
        # Formula: λ_i = [ρ(b_i - b_j) + G_jj - G_ij] / [G_ii + G_jj - 2G_ij]
        denom = G[i, i] + G[j, j] - 2 * G[i, j]
        if abs(denom) > 1e-12
            num = rpb.proximal_parameter * (b[i] - b[j]) + G[j, j] - G[i, j]
            lam_i = num / denom
            
            if 0 < lam_i < 1
                lambda = zeros(T, 3)
                lambda[i] = lam_i
                lambda[j] = 1 - lam_i
                push!(candidates, lambda)
            end
        end
    end

    # --- step 1c: Interior (All three variables > 0) ---
    # Solve KKT system: [G  1][\lambda] = [\rho b]
    #                   [1^T 0][\nu]   [1 ]
    # Build KKT matrix explicitly
    KKT_mat = zeros(T, 4, 4)
    KKT_mat[1:3, 1:3] .= G
    KKT_mat[1:3, 4] .= one(T)
    KKT_mat[4, 1:3] .= one(T)
    KKT_mat[4, 4] = zero(T)
    
    rhs = [rpb.proximal_parameter .* b; one(T)]
    
    try
        # Use \ for a direct solver; for 4x4 it is extremely efficient
        sol = KKT_mat \ rhs
        lam_interior = sol[1:3]
        
        # Check if the stationary point lies inside the simplex
        if all(x -> x > 1e-9, lam_interior)
            push!(candidates, lam_interior)
        end
    catch e
        # Handle singular matrix if a_i are linearly dependent
    end

    # step 2: evaluate dual objective at all candidates and select the best one
    best_value = -Inf
    optimal_lambda = candidates[1]
    
    for lambda in candidates
        value = dual_objective(lambda)
        if value > best_value
            best_value = value
            optimal_lambda = lambda
        end
    end


    # step 3: compute and return primal solution from optimal dual multipliers
    # Compute linear combination in the tangent space: optimal_lambda[1] * a_1 + optimal_lambda[2] * a_2 + optimal_lambda[3] * a_3
    # Since we're in the tangent space at rpb.current_proximal_center, we can use standard linear combinations
    result = optimal_lambda[1] .* a_1 .+ optimal_lambda[2] .* a_2 .+ optimal_lambda[3] .* a_3

    if dual_objective(optimal_lambda) > rpb.anchor_cut_shift
        # print("Warning: Dual objective at optimal lambda exceeds anchor cut shift; possible numerical issue.\n")
        # print("Optimal lambda: $(optimal_lambda), Dual value: $(dual_objective(optimal_lambda)), Anchor cut shift: $(rpb.anchor_cut_shift)\n")
    end

    # print("dual_objective(optimal_lambda): $(dual_objective(optimal_lambda))\n")
    
    # model_objective = model_evaluation(rpb, -(one(T)/rpb.proximal_parameter) .* result)
    # print("primal_solution_objective: $(model_objective + (rpb.proximal_parameter / 2) * inner_product(rpb.manifold, rpb.current_proximal_center, -(one(T)/rpb.proximal_parameter) .* result, -(one(T)/rpb.proximal_parameter) .* result))\n")

    return -(one(T)/rpb.proximal_parameter) .* result

end


"""
    model_versus_true(rpb::RProximalBundle, cand_obj, cand_model, current_obj=nothing)

Compute the model's predicted objective gap versus the true objective gap.
"""
function model_versus_true(rpb::RProximalBundle{T}, cand_obj, cand_model, current_obj=nothing) where T
    # Failsafe check
    if current_obj === nothing
        current_obj = rpb.compute_objective(rpb.current_proximal_center)
    end

    # true and model gaps
    true_gap = current_obj - cand_obj
    model_gap = current_obj - cand_model

    # numerical_tolerance = 10 * eps(typeof(current_obj))

    # check if gaps are above floating point representation
    # if abs(true_gap) < numerical_tolerance
        # print("Warning: True gap below numerical tolerance; setting to zero.\n")
        # true_gap = T(0.0)
    # end
    
    # if abs(model_gap) < numerical_tolerance
    #     # print("Warning: Model gap below numerical tolerance; setting to zero.\n")
    #     # model_gap = T(0.0)
    # end

    if model_gap == T(0)
        ratio = Inf
    else
        ratio = true_gap / model_gap
    end 

    if true_gap >= (rpb.trust_parameter) * model_gap
        return true, ratio
    end

    # print("Model versus true objective gap check failed.\n")
    # print("Current Obj: $(current_obj), Candidate Obj: $(cand_obj), Model Obj: $(cand_model)\n")
    # print("True Gap: $(current_obj - cand_obj), Model Gap: $(current_obj - cand_model)\n")
    return false, ratio
end

"""
    proximal_parameter_check(rpb::RProximalBundle, candidate_direction, new_subgradient, candidate_point, model_objective)

Compare proximal gap with shift.
"""
function proximal_parameter_check(rpb::RProximalBundle{T}, current_model_proximal_gap, error_shift) where T
    if current_model_proximal_gap >= (2 * error_shift) / (1 - rpb.trust_parameter)
        return true
    end
    return false
end

"""
    compute_shift_adjustment(rpb::RProximalBundle, new_subgradient, candidate_point)

Compute shift adjustment for the model.
"""
function compute_shift_adjustment(rpb::RProximalBundle{T}, new_subgradient, candidate_point) where T
    # Compute relevant subgradient norms
    inner_prod_center = inner_product(rpb.manifold, rpb.current_proximal_center, rpb.anchor_cut_subg, rpb.anchor_cut_subg)
    inner_prod_new = inner_product(rpb.manifold, candidate_point, new_subgradient, new_subgradient)

    # Ensure non-negative values for square root
    norm_subgradient_center = sqrt(max(inner_prod_center, T(0)))
    norm_new_subgradient = sqrt(max(inner_prod_new, T(0)))

    curvature_and_approx_term = sqrt(-rpb.sectional_curvature) + rpb.retraction_error + 2 * rpb.transport_error
    center_subg_prox_frac = (2 * norm_subgradient_center) / rpb.proximal_parameter
    center_subg_prox_frac_sq = (center_subg_prox_frac)^2

    radius_of_cand = center_subg_prox_frac + rpb.retraction_error * center_subg_prox_frac_sq
    shift_adjustment = curvature_and_approx_term * norm_new_subgradient * (radius_of_cand)^2

    # robustness check to floating point issues --- shifts have to be non-negative
    if shift_adjustment < T(0)
        print("Warning: Computed negative shift adjustment due to floating point errors; setting to zero.\n")
        shift_adjustment = T(0)
    end
    # eps(T) is the smallest difference between 1.0 and the next float
    if shift_adjustment < 10 * eps(typeof(shift_adjustment))
        # print("Warning: Computed shift adjustment is extremely small due to floating point errors; setting to zero.\n")
        shift_adjustment = T(0)
    end

    return shift_adjustment
end

"""
    compute_model_proximal_gap(rpb::RProximalBundle, candidate_direction, model_objective)

Compute the proximal gap on the model.
"""
function compute_model_proximal_gap(rpb::RProximalBundle{T}, candidate_direction, model_objective) where T
    current_location_objective = rpb.compute_objective(rpb.current_proximal_center)

    # Computes the proximal objective on the model
    prox_obj_on_model = model_objective + (rpb.proximal_parameter / 2) * inner_product(rpb.manifold, rpb.current_proximal_center, candidate_direction, candidate_direction)

    prox_gap = current_location_objective - prox_obj_on_model
    # if prox_gap < T(0)
    #     print("Warning: Computed negative proximal gap due to floating point errors; setting to zero.\n")
    #     print("prox_gap: $(prox_gap), current_obj: $(current_location_objective), prox_obj_on_model: $(prox_obj_on_model)\n")
    # end
    
    return prox_gap
end

"""
    backtracking_procedure(rpb::RProximalBundle)

Perform a backtracking procedure to adjust proximal parameter such that conditions for descent and null step are met.
"""

function backtracking_procedure(rpb::RProximalBundle{T}) where T
    # Initialize variables outside the loop scope
    local candidate_point, candidate_direction, subgradient_at_candidate_pre_project, subgradient_at_candidate, transported_subgradient_from_candidate_pre_project, transported_subgradient_from_candidate, potential_error_shift, ratio

    descent_flag = false

    # while loop
    while true
        # Compute candidate direction and convert to candidate point
        candidate_direction = cand_prox_direction(rpb)
        candidate_point = rpb.retraction_map(rpb.current_proximal_center, candidate_direction)

        # Compute true objective and predicted objective
        model_objective = model_evaluation(rpb, candidate_direction)
        candidate_objective = rpb.compute_objective(candidate_point)

        # Cache current objective for descent check computation
        current_objective = rpb.compute_objective(rpb.current_proximal_center)
        subgradient_at_candidate_pre_project = rpb.compute_subgradient(candidate_point)
        subgradient_at_candidate = project(rpb.manifold, candidate_point, subgradient_at_candidate_pre_project)

        transported_subgradient_from_candidate_pre_project = rpb.transport_map(candidate_point, rpb.current_proximal_center, subgradient_at_candidate)
        transported_subgradient_from_candidate = project(rpb.manifold, rpb.current_proximal_center, transported_subgradient_from_candidate_pre_project)
        potential_error_shift = compute_shift_adjustment(rpb, subgradient_at_candidate, candidate_point)

        # compute ratio and proximal gap
        descent_flag, ratio = model_versus_true(rpb, candidate_objective, model_objective, current_objective)
        current_model_proximal_gap = compute_model_proximal_gap(rpb, candidate_direction, model_objective)

        if descent_flag
            # Store candidate data before breaking for descent step
            push!(rpb.candidate_directions, candidate_direction)
            push!(rpb.candidate_obj_history, candidate_objective)
            push!(rpb.candidate_model_obj_history, model_objective)
            descent_flag = true
            # print("Descent \n")
            break  # descent condition met
        end

        if proximal_parameter_check(rpb, current_model_proximal_gap, potential_error_shift)
            # Store candidate data before breaking for null step
            push!(rpb.candidate_directions, candidate_direction)
            push!(rpb.candidate_obj_history, candidate_objective)
            push!(rpb.candidate_model_obj_history, model_objective)
            # print("Null \n")
            break  # null step condition met
        end

        # Store objective at current proximal center after each iteration (regardless of step type)
        current_proximal_objective = rpb.anchor_cut_shift
        push!(rpb.relative_objective_history, (current_proximal_objective - rpb.true_min_obj) / (rpb.initial_objective - rpb.true_min_obj))
        push!(rpb.objective_history, current_proximal_objective - rpb.true_min_obj)
        push!(rpb.raw_objective_history, current_proximal_objective)

        # print("Checks before backtracking failed. Adjusting proximal parameter...\n")

        # # for debugging purposes
        # print("\nBacktracking: Proximal parameter adjusted from $(rpb.proximal_parameter) to $(rpb.proximal_parameter * rpb.back_tracking_factor)\n")
        # print("Proximal Gap: $(current_model_proximal_gap),\n") 
        # print("Shift: $(potential_error_shift)\n")
        # print("Inner Product Subgradient at Candidate: $(inner_product(rpb.manifold, candidate_point, subgradient_at_candidate, subgradient_at_candidate))\n")
        # project_subg_cand = project(rpb.manifold, candidate_point, subgradient_at_candidate)
        # print("Inner Product of Projected Subgradient at Candidate: $((inner_product(rpb.manifold, candidate_point, project_subg_cand, project_subg_cand)))\n")
        
        # print("Inner Product of Anchor Cut Subgradient at Proximal Center: $((inner_product(rpb.manifold, rpb.current_proximal_center, rpb.anchor_cut_subg, rpb.anchor_cut_subg)))\n")

        # project_subg = project(rpb.manifold, rpb.current_proximal_center, rpb.anchor_cut_subg)
        # print("Inner Product of Projected Anchor Cut Subgradient at Proximal Center: $((inner_product(rpb.manifold, rpb.current_proximal_center, project_subg, project_subg)))\n")
        # print("Norm of Subgradient at Candidate: $(sqrt(inner_product(rpb.manifold, candidate_point, subgradient_at_candidate, subgradient_at_candidate)))\n")
        # print("Norm of Anchor Cut Subgradient: $(sqrt(inner_product(rpb.manifold, rpb.current_proximal_center, rpb.anchor_cut_subg, rpb.anchor_cut_subg)))\n")

        # If neither condition is met, multiply the proximal parameter by the backtracking factor
        rpb.proximal_parameter *= rpb.back_tracking_factor
    end

    return candidate_point, candidate_direction, descent_flag, subgradient_at_candidate, transported_subgradient_from_candidate, potential_error_shift, ratio
end


## Helper functions for visualizations

"""
    plot_objective_versus_iter(rpb::RProximalBundle; save_path=nothing, use_loglog=false)

Plot objective versus iteration number with step type indicators.
Uses semi-log scale by default, with option for log-log scale.

# Arguments
- `rpb::RProximalBundle`: The proximal bundle solver instance
- `save_path=nothing`: Optional path to save the plot. If provided, saves to this path.
- `use_loglog=false`: If true, uses log-log scale. If false (default), uses semi-log scale.
"""
function plot_objective_versus_iter(rpb::RProximalBundle{T}; save_path=nothing, use_loglog=false) where T
    # Choose what to plot based on whether we know the minimizer
    if rpb.know_minimizer
        y_data = rpb.objective_history
        y_label = "Objective Gap"
        title = "Objective Gap vs Iteration Number"
        print_label = "Final Objective Gap"
    else
        y_data = rpb.raw_objective_history
        y_label = "Objective Value"
        title = "Objective Value vs Iteration Number"
        print_label = "Final Objective Value"
    end

    # Create iteration numbers (add 1 to avoid log(0) issues)
    x_data = 1:length(y_data)

    # Set scale based on use_loglog parameter
    if use_loglog
        scale_x = :log10
        scale_y = :log10
        title = title * " (Log-Log Scale)"
    else
        scale_x = :identity
        scale_y = :log10
        title = title * " (Semi-Log Scale)"
    end

    # Create the main plot with the specified color and scale
    p = plot(x_data, y_data,
             label=y_label,
             color="#785ef0",
             linewidth=2,
             title=title,
             xlabel="Iteration Number",
             ylabel=y_label,
             xscale=scale_x,
             yscale=scale_y,
             dpi=300)

    # Add markers for different step types (adjust indices for 1-based indexing)
    if !isempty(rpb.indices_of_descent_steps)
        valid_indices = filter(i -> i + 1 <= length(y_data), rpb.indices_of_descent_steps)
        if !isempty(valid_indices)
            scatter!(p, valid_indices .+ 1, y_data[valid_indices .+ 1],
                    color=:green, marker=:circle, markersize=4,
                    label="Descent Steps")
        end
    end

    if !isempty(rpb.indices_of_null_steps)
        valid_indices = filter(i -> i + 1 <= length(y_data), rpb.indices_of_null_steps)
        if !isempty(valid_indices)
            scatter!(p, valid_indices .+ 1, y_data[valid_indices .+ 1],
                    color=:orange, marker=:square, markersize=3,
                    label="Null Steps")
        end
    end

    if !isempty(rpb.indices_of_proximal_doubling_steps)
        valid_indices = filter(i -> i + 1 <= length(y_data), rpb.indices_of_proximal_doubling_steps)
        if !isempty(valid_indices)
            scatter!(p, valid_indices .+ 1, y_data[valid_indices .+ 1],
                    color=:red, marker=:uptriangle, markersize=3,
                    label="Proximal Doubling Steps")
        end
    end

    # Add markers for objective increases at proximal centers
    if !isempty(rpb.objective_increase_flags)
        increase_indices = findall(rpb.objective_increase_flags)
        valid_increase_indices = filter(i -> i <= length(y_data), increase_indices)
        if !isempty(valid_increase_indices)
            scatter!(p, valid_increase_indices, y_data[valid_increase_indices],
                    color=:purple, marker=:star5, markersize=5,
                    label="Objective Increase at Proximal Center")
        end
    end

    # Add grid
    plot!(p, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)

    # Save plot if path is provided
    if save_path !== nothing
        savefig(p, save_path)
        println("Plot saved to: $save_path")
    end

    display(p)

    # Print summary statistics
    println(title)
    println("----------------------------------")
    println("$print_label: $(y_data[end])")
    println("----------------------------------")
    println("Descent Steps: $(length(rpb.indices_of_descent_steps))")
    println("Null Steps: $(length(rpb.indices_of_null_steps))")
    println("Proximal Doubling Steps: $(length(rpb.indices_of_proximal_doubling_steps))")
    if !isempty(rpb.objective_increase_flags)
        num_increases = count(rpb.objective_increase_flags)
        println("Objective Increases at Proximal Center: $num_increases")
    end
    println("----------------------------------")

    return p
end

"""
    plot_transport_constant_versus_iter(rpb::RProximalBundle; save_path=nothing, use_loglog=false)

Plot estimated transport constant C_T versus iteration number for debugging.
Uses semi-log scale by default, with option for log-log scale.

# Arguments
- `rpb::RProximalBundle`: The proximal bundle solver instance
- `save_path=nothing`: Optional path to save the plot. If provided, saves to this path.
- `use_loglog=false`: If true, uses log-log scale. If false (default), uses semi-log scale.
"""
function plot_transport_constant_versus_iter(rpb::RProximalBundle{T}; save_path=nothing, use_loglog=false) where T
    # Check if debugging was enabled and transport constants were collected
    if !rpb.debugging || isempty(rpb.transport_constant_approx)
        println("Warning: No transport constant data available. Make sure debugging=true when creating the solver.")
        return nothing
    end

    # Create iteration numbers (add 1 to avoid log(0) issues)
    x_data = 1:length(rpb.transport_constant_approx)
    y_data = rpb.transport_constant_approx

    # Set scale based on use_loglog parameter
    if use_loglog
        scale_x = :log10
        scale_y = :log10
        title = "Transport Constant C_T vs Iteration Number (Log-Log Scale)"
    else
        scale_x = :identity
        scale_y = :log10
        title = "Transport Constant C_T vs Iteration Number (Semi-Log Scale)"
    end

    # Create the main plot
    p = plot(x_data, y_data,
             label="C_T Estimate",
             color="#dc267f",
             linewidth=2,
             title=title,
             xlabel="Iteration Number",
             ylabel="Transport Constant C_T",
             xscale=scale_x,
             yscale=scale_y,
             dpi=300)

    # Add markers for different step types if available
    if !isempty(rpb.indices_of_descent_steps)
        valid_indices = filter(i -> i <= length(y_data), rpb.indices_of_descent_steps)
        if !isempty(valid_indices)
            scatter!(p, valid_indices, y_data[valid_indices],
                    color=:green, marker=:circle, markersize=4,
                    label="Descent Steps")
        end
    end

    if !isempty(rpb.indices_of_null_steps)
        valid_indices = filter(i -> i <= length(y_data), rpb.indices_of_null_steps)
        if !isempty(valid_indices)
            scatter!(p, valid_indices, y_data[valid_indices],
                    color=:orange, marker=:square, markersize=3,
                    label="Null Steps")
        end
    end

    if !isempty(rpb.indices_of_proximal_doubling_steps)
        valid_indices = filter(i -> i <= length(y_data), rpb.indices_of_proximal_doubling_steps)
        if !isempty(valid_indices)
            scatter!(p, valid_indices, y_data[valid_indices],
                    color=:red, marker=:uptriangle, markersize=3,
                    label="Proximal Doubling Steps")
        end
    end

    # Add grid
    plot!(p, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)

    # Add horizontal line for the current transport_error parameter
    if rpb.transport_error > 0
        hline!(p, [rpb.transport_error],
               color=:black, linestyle=:dash, linewidth=2,
               label="Current Transport Error Parameter")
    end

    # Save plot if path is provided
    if save_path !== nothing
        savefig(p, save_path)
        println("Transport constant plot saved to: $save_path")
    end

    display(p)

    # Print summary statistics
    println(title)
    println("----------------------------------")
    if !isempty(y_data)
        println("Final C_T Estimate: $(y_data[end])")
        println("Maximum C_T Estimate: $(maximum(y_data))")
        println("Minimum C_T Estimate: $(minimum(y_data))")
        println("Mean C_T Estimate: $(sum(y_data)/length(y_data))")
        println("Current Transport Error Parameter: $(rpb.transport_error)")
        println("----------------------------------")
        println("Total Estimates: $(length(y_data))")
    end

    return p
end

"""
    compute_transport_error_constant_approximation(rpb:RProximalBundle, new_subgradient, candidate_point)

Compute transport error constant approximation for the model. C_{T} = \frac{||vec_transport(new_subgradient) - par_transport(new_subgradient||}{||new_subgradient|| d(x_{k}, z_{k+1})}
"""
function compute_transport_error_constant_approximation(rpb::RProximalBundle{T}, new_subgradient, candidate_point) where T
    # # Parallel transport of the new subgradient from candidate_point to proximal center
    # parallel_transported = vector_transport_to(rpb.manifold, candidate_point, new_subgradient, rpb.current_proximal_center, ParallelTransport())

    # # Projection transport of the new subgradient from candidate_point to proximal center
    # projection_transported = vector_transport_to(rpb.manifold, candidate_point, new_subgradient, rpb.current_proximal_center, ProjectionTransport())

    # # Compute the norm of the difference between the two transport methods
    # transport_diff_norm = norm(rpb.manifold, rpb.current_proximal_center, parallel_transported - projection_transported)

    # # Compute the norm of the original subgradient
    # original_subg_norm = norm(rpb.manifold, candidate_point, new_subgradient)

    # # Compute the distance between the points
    # transport_distance = distance(rpb.manifold, candidate_point, rpb.current_proximal_center)

    # # Compute the transport error constant approximation
    # # C_T = ||vec_transport(subgradient) - par_transport(subgradient)|| / (||subgradient|| * distance)
    # if original_subg_norm * transport_distance > T(1e-12)
    #     transport_error_constant = transport_diff_norm / (original_subg_norm * transport_distance)
    # else
    #     transport_error_constant = T(0.0)  # Avoid division by zero
    # end

    # # return transport_error_constant

    # Temporary replacement - return the configured transport error parameter
    return rpb.transport_error
end

"""
    plot_model_proximal_gap(rpb::RProximalBundle; save_path=nothing, use_loglog=false)

Plot model proximal gap versus iteration number with symbols only on descent steps.
Uses semi-log scale by default, with option for log-log scale.

# Arguments
- `rpb::RProximalBundle`: The proximal bundle solver instance
- `save_path=nothing`: Optional path to save the plot. If provided, saves to this path.
- `use_loglog=false`: If true, uses log-log scale. If false (default), uses semi-log scale.
"""
function plot_model_proximal_gap(rpb::RProximalBundle{T}; save_path=nothing, use_loglog=false) where T
    # Check if model proximal gap data is available
    if isempty(rpb.model_proximal_gap_history)
        println("Warning: No model proximal gap data available. Make sure run! has been called.")
        return nothing
    end

    # Prepare data
    x_data = 1:length(rpb.model_proximal_gap_history)
    y_data = rpb.model_proximal_gap_history

    # Set scale based on use_loglog parameter
    if use_loglog
        scale_x = :log10
        scale_y = :log10
        title = "Model Proximal Gap vs Iteration Number (Log-Log Scale)"
    else
        scale_x = :identity
        scale_y = :identity
        title = "Model Proximal Gap vs Iteration Number (Semi-Log Scale)"
    end

    # Create the main plot
    p = plot(x_data, y_data,
             label="Model Proximal Gap",
             color="#fe6100",
             linewidth=2,
             title=title,
             xlabel="Iteration Number",
             ylabel="Model Proximal Gap",
             xscale=scale_x,
             yscale=scale_y,
             dpi=300)

    # Add markers ONLY for descent steps
    if !isempty(rpb.indices_of_descent_steps)
        valid_indices = filter(i -> i <= length(y_data), rpb.indices_of_descent_steps)
        if !isempty(valid_indices)
            scatter!(p, valid_indices, y_data[valid_indices],
                    color=:green, marker=:circle, markersize=6,
                    label="Descent Steps")
        end
    end

    # Add grid
    plot!(p, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)

    # Save plot if path is provided
    if save_path !== nothing
        savefig(p, save_path)
        println("Model proximal gap plot saved to: $save_path")
    end

    display(p)

    # Print summary statistics
    println(title)
    println("----------------------------------")
    if !isempty(y_data)
        println("Final Model Proximal Gap: $(y_data[end])")
        println("Maximum Model Proximal Gap: $(maximum(y_data))")
        println("Minimum Model Proximal Gap: $(minimum(y_data))")
        println("Mean Model Proximal Gap: $(sum(y_data)/length(y_data))")
        println("----------------------------------")
        println("Total Iterations: $(length(y_data))")
        println("Descent Steps: $(length(rpb.indices_of_descent_steps))")
    end

    return p
end