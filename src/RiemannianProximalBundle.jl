using LinearAlgebra
using Plots

"""
    RProximalBundle

Riemannian proximal bundle algorithm for non-convex optimization on Riemannian manifolds.
This struct implements the proximal bundle algorithm for non-convex optimization on Riemannian manifolds.
The algorithm is based on the proximal bundle method for convex optimization, which is a generalization
of the bundle method for non-convex optimization.
This implementation centers on a two-cut surrogate model, however, one may easily replace with other
convex surrogates, satisfying certain properties.
"""
mutable struct RProximalBundle
    # Riemannian Optimization Tools
    manifold::Any
    retraction_map::Function
    transport_map::Function

    # Parameters for first-order retraction/transport
    retraction_error::Float64
    transport_error::Float64
    sectional_curvature::Float64

    # Explicit Optimization Oracles
    compute_objective::Function
    compute_subgradient::Function

    # State Variables
    current_proximal_center::Any  # where model lives during constructions
    subgradient_at_center::Any    # g_{k}, subgradient at proximal center
    proximal_parameter::Float64   # current ρ_{k}
    single_cut::Bool             # flag for if the model is single-cut or two-cut

    initial_objective::Float64   # f(x_{0})
    max_iter::Int               # maximum number of iterations
    tolerance::Float64          # tolerance for convergence
    trust_parameter::Float64    # pre-specified β

    # Toggles for variants
    adaptive_proximal::Bool     # flag for if the proximal parameter is adaptively updated
    know_minimizer::Bool        # flag for if the true minimizer is known
    relative_error::Bool        # flag for if the error to store should be taken with the initial objective as a denominator

    # Storage for two-cut model - setting aside memory
    model_subg::Vector{Any}                    # s_{k}, initialize together to line-up indexes, dummy entry
    untransported_subgradients::Vector{Any}    # g_{k}
    transported_subgradients::Vector{Any}      # ĝ_{k}
    candidate_directions::Vector{Any}          # d_{k}
    candidate_obj_history::Vector{Float64}     # f(R_x(x_{k}))
    candidate_model_obj_history::Vector{Float64}  # f_k(d_{k})
    error_shifts::Vector{Float64}              # e_{f_k}(ρ_{k})
    proximal_center_history::Vector{Any}       # x_{k}

    # Model information for two-cut model evaluation
    prev_candidate_direction::Union{Nothing, Any}
    prev_true_obj::Union{Nothing, Float64}
    prev_model_obj::Union{Nothing, Float64}
    prev_transport_subg::Union{Nothing, Any}
    prev_model_subg::Union{Nothing, Any}
    prev_error_shift::Union{Nothing, Float64}
    prev_prox_parameter::Float64

    # Storage for algorithm run
    proximal_parameter_history::Vector{Float64}
    relative_objective_history::Vector{Float64}
    objective_history::Vector{Float64}          # Store gaps for visualization
    raw_objective_history::Vector{Float64}      # Store raw objectives for algorithm logic
    true_min_obj::Float64
    indices_of_descent_steps::Vector{Int}
    indices_of_null_steps::Vector{Int}
    indices_of_proximal_doubling_steps::Vector{Int}

    function RProximalBundle(manifold, retraction_map, transport_map, objective_function,
                            subgradient, initial_point, initial_objective, initial_subgradient;
                            true_min_obj=0.0, retraction_error=0.0, transport_error=0.0,
                            sectional_curvature=-0.5, proximal_parameter=0.02,
                            trust_parameter=0.1, max_iter=200, tolerance=1e-12,
                            adaptive_proximal=false, know_minimizer=true, relative_error=true)

        # Initialize storage arrays
        model_subg = [initial_subgradient]
        untransported_subgradients = [initial_subgradient]
        transported_subgradients = [initial_subgradient]
        candidate_directions = Any[]
        candidate_obj_history = [initial_objective]
        candidate_model_obj_history = Float64[]
        error_shifts = [0.0]
        proximal_center_history = [initial_point]

        # Initialize history arrays
        proximal_parameter_history = [proximal_parameter]
        relative_objective_history = [(initial_objective - true_min_obj) / (initial_objective - true_min_obj)]
        objective_history = [initial_objective - true_min_obj]
        raw_objective_history = [initial_objective]

        # Initialize index tracking arrays
        indices_of_descent_steps = Int[]
        indices_of_null_steps = Int[]
        indices_of_proximal_doubling_steps = Int[]

        new(manifold, retraction_map, transport_map,
            retraction_error, transport_error, sectional_curvature,
            objective_function, subgradient,
            initial_point, initial_subgradient, proximal_parameter, true,
            initial_objective, max_iter, tolerance, trust_parameter,
            adaptive_proximal, know_minimizer, relative_error,
            model_subg, untransported_subgradients, transported_subgradients,
            candidate_directions, candidate_obj_history, candidate_model_obj_history,
            error_shifts, proximal_center_history,
            nothing, nothing, nothing, nothing, nothing, nothing, proximal_parameter,
            proximal_parameter_history, relative_objective_history, objective_history,
            raw_objective_history, true_min_obj,
            indices_of_descent_steps, indices_of_null_steps, indices_of_proximal_doubling_steps)
    end
end

"""
    run!(rpb::RProximalBundle)

Run the proximal bundle algorithm.
"""
function run!(rpb::RProximalBundle)
    for i in 1:rpb.max_iter
        # Fix two-cut model information before any updates
        rpb.prev_candidate_direction = length(rpb.candidate_directions) > 0 ? rpb.candidate_directions[end] : nothing
        rpb.prev_true_obj = length(rpb.candidate_obj_history) > 0 ? rpb.candidate_obj_history[end] : nothing
        rpb.prev_model_obj = length(rpb.candidate_model_obj_history) > 0 ? rpb.candidate_model_obj_history[end] : nothing
        rpb.prev_transport_subg = length(rpb.transported_subgradients) > 0 ? rpb.transported_subgradients[end] : nothing
        rpb.prev_model_subg = rpb.prev_candidate_direction !== nothing ? -rpb.prev_prox_parameter * rpb.prev_candidate_direction : nothing
        push!(rpb.model_subg, rpb.prev_model_subg)  # s_{k+1} = -ρ_{k+1} d_{k+1}
        rpb.prev_error_shift = length(rpb.error_shifts) > 0 ? rpb.error_shifts[end] : nothing

        # Compute the candidate direction and convert to a point on the manifold using retraction map
        candidate_direction = cand_prox_direction(rpb)
        push!(rpb.candidate_directions, candidate_direction)

        candidate_point = rpb.retraction_map(rpb.current_proximal_center, candidate_direction)

        # Compute true objective and predicted objective
        model_objective = model_evaluation(rpb, candidate_direction)
        push!(rpb.candidate_model_obj_history, model_objective)

        candidate_objective = rpb.compute_objective(candidate_point)
        push!(rpb.candidate_obj_history, candidate_objective)

        # Cache current objective for consistent recording
        current_objective = rpb.compute_objective(rpb.current_proximal_center)

        # Query new subgradient
        new_subgradient = rpb.compute_subgradient(candidate_point)

        # Compute the model's predicted objective gap versus the true objective gap
        ratio = model_versus_true(rpb, candidate_objective, model_objective, current_objective)

        # Be more permissive: accept if ratio is good AND we're not going significantly uphill
        if ratio > rpb.trust_parameter  # DESCENT STEP
            rpb.current_proximal_center = candidate_point  # moves model
            push!(rpb.proximal_center_history, candidate_point)  # stores the new proximal center

            # Update proximal center and set initial subgradient
            rpb.current_proximal_center = candidate_point
            rpb.subgradient_at_center = new_subgradient  # updates subgradient at center

            # Update model information - one-cut model now!
            push!(rpb.untransported_subgradients, new_subgradient)  # g_{k+1}
            push!(rpb.transported_subgradients, new_subgradient)    # no transport is done
            push!(rpb.error_shifts, 0.0)  # e_{f_k}(ρ_{k+1}) = 0, no transport is done

            rpb.single_cut = true

            push!(rpb.indices_of_descent_steps, i)
            push!(rpb.proximal_parameter_history, rpb.proximal_parameter)
        else
            if proximal_parameter_check(rpb, candidate_direction, new_subgradient, candidate_point, model_objective) || !rpb.adaptive_proximal  # NULL STEP
                # Don't move - just update model
                # Transports subg to proximal center tangent space
                transported_subg = rpb.transport_map(candidate_point, rpb.current_proximal_center, new_subgradient)

                push!(rpb.untransported_subgradients, new_subgradient)  # g_{k+1}
                push!(rpb.transported_subgradients, transported_subg)   # ĝ_{k+1}

                error_shift = compute_shift_adjustment(rpb, new_subgradient, candidate_point)
                push!(rpb.error_shifts, error_shift)  # conservative shift adjustment

                rpb.single_cut = false  # no longer single-cut model after taking steps unless we take a descent step

                push!(rpb.proximal_parameter_history, rpb.proximal_parameter)
                push!(rpb.indices_of_null_steps, i)
            else  # PROXIMAL PARAMETER DOUBLING STEP
                rpb.prev_prox_parameter = rpb.proximal_parameter
                rpb.proximal_parameter *= 2  # double proximal parameter

                # Don't update model and don't move - repeat previous model
                prev_untransport_subg = rpb.untransported_subgradients[end]  # g_{k+1}
                prev_transport_subg = rpb.transported_subgradients[end]      # ĝ_{k+1}
                prev_error_shift = rpb.error_shifts[end]                     # e_{f_k}(ρ_{k+1})

                push!(rpb.untransported_subgradients, prev_untransport_subg)  # g_{k+1}
                push!(rpb.transported_subgradients, prev_transport_subg)      # ĝ_{k+1}
                push!(rpb.error_shifts, prev_error_shift)                     # e_{f_k}(ρ_{k+1})

                push!(rpb.proximal_parameter_history, rpb.proximal_parameter)
                push!(rpb.indices_of_proximal_doubling_steps, i)
            end
        end

        # Store objective at current proximal center after each iteration (regardless of step type)
        current_proximal_objective = rpb.compute_objective(rpb.current_proximal_center)
        push!(rpb.relative_objective_history, (current_proximal_objective - rpb.true_min_obj) / (rpb.initial_objective - rpb.true_min_obj))
        push!(rpb.objective_history, current_proximal_objective - rpb.true_min_obj)
        push!(rpb.raw_objective_history, current_proximal_objective)

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

## Helper functions for proximal bundle algorithm run

"""
    model_evaluation(rpb::RProximalBundle, direction)

Compute the cut surrogate model.
"""
function model_evaluation(rpb::RProximalBundle, direction)
    if rpb.single_cut
        # Computes the model objective using the single-cut model: f_k(d_{k}) = f(x_{k}) + ⟨g_{k}, d_{k}⟩
        return rpb.raw_objective_history[end] + inner_product(rpb.manifold, rpb.current_proximal_center, rpb.untransported_subgradients[end], direction)
    end

    # Check if we have valid two-cut model data
    if rpb.prev_transport_subg === nothing || rpb.prev_model_subg === nothing ||
       rpb.prev_true_obj === nothing || rpb.prev_error_shift === nothing ||
       rpb.prev_model_obj === nothing || rpb.prev_candidate_direction === nothing
        # Fallback to single-cut model
        return rpb.raw_objective_history[end] + inner_product(rpb.manifold, rpb.current_proximal_center, rpb.untransported_subgradients[end], direction)
    end

    # Computes on "new" cut
    affine_new_shift = rpb.prev_true_obj - inner_product(rpb.manifold, rpb.current_proximal_center, rpb.prev_transport_subg, rpb.prev_candidate_direction) - rpb.prev_error_shift
    new_inner = inner_product(rpb.manifold, rpb.current_proximal_center, rpb.prev_transport_subg, direction)
    new_cut_obj = affine_new_shift + new_inner

    # Computes on "old" cut
    affine_old_shift = rpb.prev_model_obj - inner_product(rpb.manifold, rpb.current_proximal_center, rpb.prev_model_subg, rpb.prev_candidate_direction)
    old_inner = inner_product(rpb.manifold, rpb.current_proximal_center, rpb.prev_model_subg, direction)
    old_cut_obj = affine_old_shift + old_inner

    # Returns the maximum of the two cuts
    return max(new_cut_obj, old_cut_obj)
end

"""
    cand_prox_direction(rpb::RProximalBundle)

Compute proximal direction due to model.
"""
function cand_prox_direction(rpb::RProximalBundle)
    if rpb.single_cut
        # Computes the proximal direction using the single-cut model: -(g_{k}/ρ_{k})
        return -(rpb.untransported_subgradients[end] / rpb.proximal_parameter)
    else
        # Use stored two-cut model information to compute proximal direction
        # Check for Nothing values and handle them
        if rpb.prev_transport_subg === nothing || rpb.prev_model_subg === nothing ||
           rpb.prev_true_obj === nothing || rpb.prev_error_shift === nothing || rpb.prev_model_obj === nothing
            # Fallback to single-cut if two-cut data is incomplete
            return -(rpb.untransported_subgradients[end] / rpb.proximal_parameter)
        end

        numerator = rpb.proximal_parameter * (rpb.prev_true_obj - rpb.prev_error_shift - rpb.prev_model_obj)
        denominator = (norm(rpb.manifold, rpb.current_proximal_center, rpb.prev_transport_subg - rpb.prev_model_subg))^2

        # Avoid division by zero
        if denominator < 1e-12
            return -(rpb.untransported_subgradients[end] / rpb.proximal_parameter)
        end

        convex_comb_arg = numerator / denominator
        convex_comb = min(1, max(0, convex_comb_arg))  # Ensure convex_comb is in [0,1]

        # Computes the proximal direction - convex combination of two subg
        return -(1 / rpb.proximal_parameter) * (convex_comb * rpb.prev_transport_subg + (1 - convex_comb) * rpb.prev_model_subg)
    end
end

"""
    model_versus_true(rpb::RProximalBundle, cand_obj, cand_model, current_obj=nothing)

Compute the model's predicted objective gap versus the true objective gap.
"""
function model_versus_true(rpb::RProximalBundle, cand_obj, cand_model, current_obj=nothing)
    # Failsafe check
    if current_obj === nothing
        current_obj = rpb.compute_objective(rpb.current_proximal_center)
    end

    numerator = current_obj - cand_obj      # computes true gap on the manifold
    denominator = current_obj - cand_model  # computes model gap on the tangent space
    ratio = numerator / denominator         # computes the ratio of the two gaps

    return ratio
end

"""
    proximal_parameter_check(rpb::RProximalBundle, candidate_direction, new_subgradient, candidate_point, model_objective)

Compare proximal gap with shift.
"""
function proximal_parameter_check(rpb::RProximalBundle, candidate_direction, new_subgradient, candidate_point, model_objective)
    # Compute the proximal gap and shift adjustment
    shift_adjustment = compute_shift_adjustment(rpb, new_subgradient, candidate_point)
    model_proximal_gap = compute_model_proximal_gap(rpb, candidate_direction, model_objective)

    check_value = 0.5 * model_proximal_gap - (shift_adjustment / (1 - rpb.trust_parameter))

    return check_value >= 0
end

"""
    compute_shift_adjustment(rpb::RProximalBundle, new_subgradient, candidate_point)

Compute shift adjustment for the model.
"""
function compute_shift_adjustment(rpb::RProximalBundle, new_subgradient, candidate_point)
    # Compute relevant subgradient norms
    norm_subgradient_center = norm(rpb.manifold, rpb.current_proximal_center, rpb.subgradient_at_center)
    norm_new_subgradient = norm(rpb.manifold, candidate_point, new_subgradient)

    radius_of_cand = ((2 * norm_subgradient_center) / rpb.proximal_parameter) + rpb.retraction_error * (((2 * norm_subgradient_center) / rpb.proximal_parameter)^2)
    shift_adjustment = (sqrt(-rpb.sectional_curvature) + rpb.retraction_error + 2 * rpb.transport_error) * norm_new_subgradient * radius_of_cand^2
    return shift_adjustment
end

"""
    compute_model_proximal_gap(rpb::RProximalBundle, candidate_direction, model_objective)

Compute the proximal gap on the model.
"""
function compute_model_proximal_gap(rpb::RProximalBundle, candidate_direction, model_objective)
    current_location_objective = rpb.compute_objective(rpb.current_proximal_center)

    # Computes the proximal objective on the model
    prox_obj_on_model = model_objective + (rpb.proximal_parameter / 2) * (norm(rpb.manifold, rpb.current_proximal_center, candidate_direction))^2

    prox_gap = current_location_objective - prox_obj_on_model

    return prox_gap
end

## Helper functions for visualizations

"""
    plot_objective_versus_iter(rpb::RProximalBundle; log_log=false)

Plot objective versus iteration number.
"""
function plot_objective_versus_iter(rpb::RProximalBundle; log_log=false)
    # Choose what to plot based on whether we know the minimizer
    if rpb.know_minimizer
        y_data = rpb.objective_history  # Plot gaps when we know the minimizer
        y_label = "Objective Gap"
        title = "Objective Gap vs Iteration Number"
        print_label = "Final Objective Gap"
    else
        y_data = rpb.raw_objective_history  # Plot raw values when we don't know the minimizer
        y_label = "Objective Value"
        title = "Objective Value vs Iteration Number"
        print_label = "Final Objective Value"
    end

    # Create the main plot
    p = plot(0:length(y_data)-1, y_data,
             label=y_label,
             color=:blue,
             linewidth=1,
             title=title * (log_log ? " (Log-Log Scale)" : ""),
             xlabel="Iteration Number",
             ylabel=y_label)

    # Plot different types of steps with different colors and markers
    max_index = length(y_data) - 1

    if !isempty(rpb.indices_of_descent_steps)
        valid_indices = filter(i -> i <= max_index, rpb.indices_of_descent_steps)
        if !isempty(valid_indices)
            scatter!(p, valid_indices, y_data[valid_indices .+ 1],
                    color=:green, marker=:circle, markersize=3,
                    label="Descent Steps")
        end
    end

    if !isempty(rpb.indices_of_null_steps)
        valid_indices = filter(i -> i <= max_index, rpb.indices_of_null_steps)
        if !isempty(valid_indices)
            scatter!(p, valid_indices, y_data[valid_indices .+ 1],
                    color=:orange, marker=:square, markersize=2,
                    label="Null Steps")
        end
    end

    if !isempty(rpb.indices_of_proximal_doubling_steps)
        valid_indices = filter(i -> i <= max_index, rpb.indices_of_proximal_doubling_steps)
        if !isempty(valid_indices)
            scatter!(p, valid_indices, y_data[valid_indices .+ 1],
                    color=:red, marker=:uptriangle, markersize=2,
                    label="Proximal Doubling Steps")
        end
    end

    # Set scaling based on log_log parameter
    if log_log
        plot!(p, xscale=:log10, yscale=:log10)
    else
        plot!(p, yscale=:log10)
    end

    plot!(p, grid=true, gridwidth=1, gridcolor=:gray, gridalpha=0.3)

    display(p)

    println(title)
    println("----------------------------------")
    println("$print_label: $(y_data[end])")
    println("----------------------------------")
    println("Descent Steps: $(length(rpb.indices_of_descent_steps))")
    println("Null Steps: $(length(rpb.indices_of_null_steps))")
    println("Proximal Doubling Steps: $(length(rpb.indices_of_proximal_doubling_steps))")
    println("----------------------------------")
end

"""
    plot_proximal_parameter_versus_iter(rpb::RProximalBundle)

Plot the proximal parameter versus iteration number.
"""
function plot_proximal_parameter_versus_iter(rpb::RProximalBundle)
    error("The method 'plot_proximal_parameter_versus_iter' is not yet implemented.")
end