"""
Comprehensive Riemannian Median Comparison Experiment
Includes Riemannian Proximal Bundle Method alongside standard algorithms
Based on: https://juliamanifolds.github.io/ManoptExamples.jl/stable/examples/RCBM-Median/
"""

using Random
using LinearAlgebra
using Printf
using Plots
using Manifolds
using Manopt

# Include our bundle method
include("src/RiemannianProximalBundle.jl")

# Create wrapper manifold for bundle method
struct ManifoldWrapper{TM}
    manifold::TM
end

function inner_product(wrapper::ManifoldWrapper, X, U, V)
    return inner(wrapper.manifold, X, U, V)
end

function norm(wrapper::ManifoldWrapper, X, V)
    return Manifolds.norm(wrapper.manifold, X, V)
end

"""
Generate test data for Riemannian median problem
"""
function generate_median_data(manifold, N=100, seed=57)
    Random.seed!(seed)

    # Generate N random points on the manifold
    data_points = [rand(manifold) for _ in 1:N]

    # Uniform weights
    weights = fill(1.0/N, N)

    return data_points, weights
end

"""
Define cost function for Riemannian median
"""
function median_cost_function(data_points, weights, manifold)
    return function(point)
        return sum(weights[i] * distance(manifold, point, data_points[i]) for i in 1:length(data_points))
    end
end

"""
Define subgradient function for Riemannian median
"""
function median_subgradient_function(data_points, weights, manifold)
    return function(point)
        grad = zeros(size(point))
        for i in 1:length(data_points)
            log_point = log(manifold, point, data_points[i])
            norm_log = Manifolds.norm(manifold, point, log_point)
            if norm_log > 1e-12
                grad += -weights[i] * (log_point) / norm_log
            end
        end
        return grad
    end
end

"""
Find true minimum using multiple starting points (Phase 1)
"""
function find_true_median(data_points, weights, manifold, max_candidates=10)
    println("  Finding true median with multiple starting points...")

    cost_func = median_cost_function(data_points, weights, manifold)
    subgrad_func = median_subgradient_function(data_points, weights, manifold)

    candidates = []

    # Try starting from data points
    for (i, start_point) in enumerate(data_points[1:min(5, length(data_points))])
        result = robust_gradient_descent_manifolds(cost_func, subgrad_func, start_point, manifold)
        obj_val = cost_func(result)
        push!(candidates, (obj_val, result, "data_point_$i"))
    end

    # Try random starting points
    for i in 1:5
        start_point = rand(manifold)
        result = robust_gradient_descent_manifolds(cost_func, subgrad_func, start_point, manifold)
        obj_val = cost_func(result)
        push!(candidates, (obj_val, result, "random_$i"))
    end

    # Find the best
    sort!(candidates, by=x->x[1])
    best_obj, best_point, best_method = candidates[1]

    println("    True minimum: $best_obj (from $best_method)")
    return best_point, best_obj
end

"""
Robust gradient descent for finding true minimum
"""
function robust_gradient_descent_manifolds(cost, subgradient, start_point, manifold, max_iter=1000)
    X = copy(start_point)

    for iter_num in 1:max_iter
        grad = subgradient(X)
        grad_norm = Manifolds.norm(manifold, X, grad)

        if grad_norm < 1e-12
            break
        end

        # Adaptive step size with backtracking
        step_size = 0.1 / (1 + iter_num * 0.01)
        current_obj = cost(X)

        for alpha in [step_size, step_size/2, step_size/4, step_size/8]
            try
                X_new = exp(manifold, X, -alpha * grad)
                new_obj = cost(X_new)
                if new_obj < current_obj
                    X = X_new
                    break
                end
            catch
                continue
            end
        end
    end

    return X
end

"""
Run Riemannian Proximal Bundle Method
"""
function run_proximal_bundle(data_points, weights, manifold, true_median, true_min_obj, max_iter=5000)
    println("  Running Riemannian Proximal Bundle Method...")

    # Create initial point
    Random.seed!(123)
    initial_point = rand(manifold)

    # Create wrapper manifold
    manifold_wrapper = ManifoldWrapper(manifold)

    # Define wrapper functions
    cost_func = median_cost_function(data_points, weights, manifold)
    subgrad_func = median_subgradient_function(data_points, weights, manifold)

    initial_objective = cost_func(initial_point)
    initial_subgradient = subgrad_func(initial_point)

    function objective_wrapper(X)
        return cost_func(X)
    end

    function subgradient_wrapper(X)
        return subgrad_func(X)
    end

    function retraction_wrapper(X, V)
        return exp(manifold, X, V)
    end

    function transport_wrapper(X, Y, V)
        return parallel_transport_to(manifold, X, V, Y)
    end

    # Create and run bundle method
    rpb = RProximalBundle(
        manifold_wrapper,
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
        max_iter=max_iter,
        tolerance=1e-8,
        adaptive_proximal=true,
        know_minimizer=true
    )

    run!(rpb)

    return rpb
end

"""
Run standard algorithms using Manopt.jl
"""
function run_standard_algorithms(data_points, weights, manifold, max_iter=5000)
    println("  Running standard algorithms...")

    # Create Manopt problem
    f = (M, p) -> sum(weights[i] * distance(M, p, data_points[i]) for i in 1:length(data_points))

    # Create subgradient function for Manopt
    function subgrad_f(M, p)
        grad = zero_vector(M, p)
        for i in 1:length(data_points)
            d = distance(M, p, data_points[i])
            if d > 1e-12
                v = log(M, p, data_points[i])
                grad += -weights[i] * v / Manifolds.norm(M, p, v)
            end
        end
        return grad
    end

    # Initial point
    Random.seed!(123)
    p0 = rand(manifold)

    results = Dict()

    # Subgradient Method
    try
        println("    Running Subgradient Method...")
        result_sgm = subgradient_method(
            manifold, f, subgrad_f, p0;
            evaluation=AllocatingEvaluation(),
            stepsize=ArmijoLinesearch(1.0),
            stopping_criterion=StopAfterIteration(max_iter) | StopWhenGradientNormLess(1e-8),
            record=[:Iteration, :Cost, :GradientNorm],
            return_state=true
        )
        results[:SGM] = result_sgm
    catch e
        println("    SGM failed: $e")
        results[:SGM] = nothing
    end

    # Try Proximal Bundle Algorithm if available
    try
        println("    Running Proximal Bundle Algorithm...")
        result_pba = proximal_bundle_method(
            manifold, f, subgrad_f, p0;
            evaluation=AllocatingEvaluation(),
            stopping_criterion=StopAfterIteration(max_iter) | StopWhenGradientNormLess(1e-8),
            record=[:Iteration, :Cost, :GradientNorm],
            return_state=true
        )
        results[:PBA] = result_pba
    catch e
        println("    PBA not available or failed: $e")
        results[:PBA] = nothing
    end

    return results
end

"""
Create comprehensive comparison plots
"""
function create_comparison_plots(rpb_result, standard_results, manifold_name)
    println("  Creating comparison plots...")

    plots = []

    # Plot 1: RPB convergence with step types
    rpb_plot = plot_rpb_convergence(rpb_result, "Riemannian Proximal Bundle ($manifold_name)")
    push!(plots, rpb_plot)

    # Plot 2: Compare all algorithms' objective gaps
    comparison_plot = plot_algorithm_comparison(rpb_result, standard_results, manifold_name)
    push!(plots, comparison_plot)

    return plots
end

"""
Plot RPB convergence with step type markers
"""
function plot_rpb_convergence(rpb, title_suffix="")
    objective_gaps = rpb.objective_history
    iterations = 0:(length(objective_gaps)-1)

    use_log_scale = all(objective_gaps .> 0)

    p = plot(iterations, objective_gaps,
             label="Objective Gap",
             linewidth=2,
             color=:blue,
             xlabel="Iteration Number",
             ylabel=use_log_scale ? "Objective Gap (log scale)" : "Objective Gap",
             title="RPB Convergence $title_suffix",
             grid=true,
             gridwidth=1,
             gridalpha=0.3)

    if use_log_scale
        plot!(p, yscale=:log10)
    end

    # Add step type markers
    if !isempty(rpb.indices_of_descent_steps)
        valid_descent = [i for i in rpb.indices_of_descent_steps if i <= length(objective_gaps)]
        if !isempty(valid_descent)
            scatter!(p, valid_descent .- 1, objective_gaps[valid_descent],
                    color=:green, marker=:circle, markersize=4,
                    label="Descent Steps")
        end
    end

    if !isempty(rpb.indices_of_null_steps)
        valid_null = [i for i in rpb.indices_of_null_steps if i <= length(objective_gaps)]
        if !isempty(valid_null)
            scatter!(p, valid_null .- 1, objective_gaps[valid_null],
                    color=:orange, marker=:square, markersize=3,
                    label="Null Steps")
        end
    end

    if !isempty(rpb.indices_of_proximal_doubling_steps)
        valid_doubling = [i for i in rpb.indices_of_proximal_doubling_steps if i <= length(objective_gaps)]
        if !isempty(valid_doubling)
            scatter!(p, valid_doubling .- 1, objective_gaps[valid_doubling],
                    color=:red, marker=:uptriangle, markersize=3,
                    label="Proximal Doubling Steps")
        end
    end

    return p
end

"""
Plot comparison of all algorithms
"""
function plot_algorithm_comparison(rpb_result, standard_results, manifold_name)
    p = plot(xlabel="Iteration Number",
             ylabel="Objective Gap (log scale)",
             title="Algorithm Comparison ($manifold_name)",
             yscale=:log10,
             grid=true,
             gridwidth=1,
             gridalpha=0.3)

    # Plot RPB
    rpb_gaps = rpb_result.objective_history
    rpb_iters = 0:(length(rpb_gaps)-1)
    plot!(p, rpb_iters, rpb_gaps,
          label="Riemannian Proximal Bundle",
          linewidth=3,
          color=:red)

    # Plot standard algorithms
    colors = [:blue, :green, :purple, :orange]
    color_idx = 1

    for (alg_name, result) in standard_results
        if result !== nothing
            try
                # Extract recorded data from Manopt result
                records = get_record(result)
                if !isempty(records)
                    iterations = [r[1] for r in records]  # Iteration numbers
                    costs = [r[2] for r in records]       # Cost values

                    # Convert to gaps (assuming last cost is near minimum)
                    min_cost = minimum(costs)
                    gaps = costs .- min_cost
                    gaps = max.(gaps, 1e-15)  # Avoid log of zero

                    plot!(p, iterations, gaps,
                          label=string(alg_name),
                          linewidth=2,
                          color=colors[color_idx])

                    color_idx = min(color_idx + 1, length(colors))
                end
            catch e
                println("    Warning: Could not plot $alg_name: $e")
            end
        end
    end

    return p
end

"""
Run complete experiment on SPD manifold
"""
function run_spd_experiment(dimension=3, N=100)
    println("=" ^ 60)
    println("SPD Manifold Experiment (dimension $dimension)")
    println("=" ^ 60)

    # Create manifold
    manifold = SymmetricPositiveDefinite(dimension)

    # Generate data
    println("Generating $N data points...")
    data_points, weights = generate_median_data(manifold, N)

    # Find true median
    true_median, true_min_obj = find_true_median(data_points, weights, manifold)

    # Run RPB
    rpb_result = run_proximal_bundle(data_points, weights, manifold, true_median, true_min_obj)

    # Run standard algorithms
    standard_results = run_standard_algorithms(data_points, weights, manifold)

    # Create plots
    plots = create_comparison_plots(rpb_result, standard_results, "SPD($dimension)")

    # Print summary
    println()
    println("Results Summary:")
    println("=" ^ 40)
    @printf "RPB - Final gap: %.2e, Iterations: %d\\n" rpb_result.objective_history[end] length(rpb_result.objective_history)-1
    @printf "RPB - Descent steps: %d, Null steps: %d, Doubling steps: %d\\n" length(rpb_result.indices_of_descent_steps) length(rpb_result.indices_of_null_steps) length(rpb_result.indices_of_proximal_doubling_steps)

    for (alg_name, result) in standard_results
        if result !== nothing
            try
                records = get_record(result)
                if !isempty(records)
                    final_cost = records[end][2]
                    final_gap = final_cost - true_min_obj
                    println("$alg_name - Final gap: $(final_gap), Iterations: $(length(records))")
                end
            catch e
                println("$alg_name - Could not extract results: $e")
            end
        else
            println("$alg_name - Failed to run")
        end
    end

    return rpb_result, standard_results, plots
end

"""
Main function to run experiments
"""
function main()
    println("Comprehensive Riemannian Median Comparison")
    println("Including Riemannian Proximal Bundle Method")
    println()

    # Run experiments on different SPD dimensions
    dimensions = [2, 3, 5]
    all_results = []

    for dim in dimensions
        rpb_result, standard_results, plots = run_spd_experiment(dim, 50)
        push!(all_results, (dim, rpb_result, standard_results, plots))

        # Save plots
        savefig(plots[1], "rpb_convergence_spd_$dim.png")
        savefig(plots[2], "algorithm_comparison_spd_$dim.png")
        println("Plots saved for SPD($dim)")
        println()
    end

    println("Comprehensive experiment completed!")
    return all_results
end

# Run the experiment when file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end