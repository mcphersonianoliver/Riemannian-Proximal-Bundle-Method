"""
Simple Algorithm Comparison for Riemannian Median
Compares Riemannian Proximal Bundle Method with basic gradient descent
"""

using Random
using LinearAlgebra
using Printf
using Plots
using Manifolds

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
    data_points = [rand(manifold) for _ in 1:N]
    weights = fill(1.0/N, N)
    return data_points, weights
end

"""
Define cost and subgradient functions
"""
function create_median_functions(data_points, weights, manifold)
    function cost_func(point)
        return sum(weights[i] * distance(manifold, point, data_points[i]) for i in 1:length(data_points))
    end

    function subgrad_func(point)
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

    return cost_func, subgrad_func
end

"""
Find true minimum using robust gradient descent
"""
function find_true_median(data_points, weights, manifold)
    println("  Finding true median...")
    cost_func, subgrad_func = create_median_functions(data_points, weights, manifold)

    candidates = []
    # Try multiple starting points
    for i in 1:min(5, length(data_points))
        result = robust_gradient_descent(cost_func, subgrad_func, data_points[i], manifold, 2000)
        obj_val = cost_func(result)
        push!(candidates, (obj_val, result))
    end

    for i in 1:3
        start_point = rand(manifold)
        result = robust_gradient_descent(cost_func, subgrad_func, start_point, manifold, 2000)
        obj_val = cost_func(result)
        push!(candidates, (obj_val, result))
    end

    sort!(candidates, by=x->x[1])
    return candidates[1][2], candidates[1][1]
end

"""
Robust gradient descent
"""
function robust_gradient_descent(cost, subgradient, start_point, manifold, max_iter=1000)
    X = copy(start_point)
    for iter_num in 1:max_iter
        grad = subgradient(X)
        grad_norm = Manifolds.norm(manifold, X, grad)
        if grad_norm < 1e-12
            break
        end
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
Run gradient descent with tracking
"""
function run_gradient_descent_tracked(data_points, weights, manifold, true_min_obj, max_iter=5000)
    println("  Running Gradient Descent...")

    cost_func, subgrad_func = create_median_functions(data_points, weights, manifold)

    Random.seed!(123)
    X = rand(manifold)

    objective_history = Float64[]
    gap_history = Float64[]

    for iter in 1:max_iter
        current_obj = cost_func(X)
        gap = current_obj - true_min_obj

        push!(objective_history, current_obj)
        push!(gap_history, gap)

        if gap < 1e-8
            println("    Converged at iteration $iter")
            break
        end

        grad = subgrad_func(X)
        grad_norm = Manifolds.norm(manifold, X, grad)

        if grad_norm < 1e-12
            break
        end

        # Adaptive step size
        step_size = 0.1 / (1 + iter * 0.01)

        # Line search
        best_X = X
        best_obj = current_obj

        for alpha in [step_size, step_size/2, step_size/4, step_size/8]
            try
                X_new = exp(manifold, X, -alpha * grad)
                new_obj = cost_func(X_new)
                if new_obj < best_obj
                    best_X = X_new
                    best_obj = new_obj
                end
            catch
                continue
            end
        end

        X = best_X
    end

    return X, objective_history, gap_history
end

"""
Run RPB method
"""
function run_rpb_method(data_points, weights, manifold, true_min_obj, max_iter=5000)
    println("  Running Riemannian Proximal Bundle Method...")

    cost_func, subgrad_func = create_median_functions(data_points, weights, manifold)

    Random.seed!(123)
    initial_point = rand(manifold)
    manifold_wrapper = ManifoldWrapper(manifold)

    initial_objective = cost_func(initial_point)
    initial_subgradient = subgrad_func(initial_point)

    rpb = RProximalBundle(
        manifold_wrapper,
        (X, V) -> exp(manifold, X, V),
        (X, Y, V) -> parallel_transport_to(manifold, X, V, Y),
        cost_func,
        subgrad_func,
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
Create comparison plots
"""
function create_comparison_plots(rpb_result, gd_result, manifold_name, dimension)
    gd_final, gd_objectives, gd_gaps = gd_result

    # Plot 1: RPB detailed convergence
    p1 = plot_rpb_detailed(rpb_result, manifold_name, dimension)

    # Plot 2: Algorithm comparison
    p2 = plot(xlabel="Iteration Number",
             ylabel="Objective Gap (log scale)",
             title="Algorithm Comparison: $manifold_name($dimension)",
             yscale=:log10,
             grid=true,
             legend=:topright)

    # Plot RPB
    rpb_gaps = rpb_result.objective_history
    rpb_iters = 0:(length(rpb_gaps)-1)
    plot!(p2, rpb_iters, max.(rpb_gaps, 1e-15),
          label="Riemannian Proximal Bundle",
          linewidth=3,
          color=:red)

    # Plot Gradient Descent
    gd_iters = 0:(length(gd_gaps)-1)
    plot!(p2, gd_iters, max.(gd_gaps, 1e-15),
          label="Gradient Descent",
          linewidth=2,
          color=:blue)

    return p1, p2
end

"""
Plot detailed RPB convergence
"""
function plot_rpb_detailed(rpb, manifold_name, dimension)
    objective_gaps = rpb.objective_history
    iterations = 0:(length(objective_gaps)-1)

    use_log_scale = all(objective_gaps .> 0)

    p = plot(iterations, max.(objective_gaps, 1e-15),
             label="Objective Gap",
             linewidth=2,
             color=:blue,
             xlabel="Iteration Number",
             ylabel="Objective Gap (log scale)",
             title="RPB Convergence: $manifold_name($dimension)",
             yscale=:log10,
             grid=true,
             legend=:topright)

    # Add step type markers
    if !isempty(rpb.indices_of_descent_steps)
        valid_descent = [i for i in rpb.indices_of_descent_steps if i <= length(objective_gaps)]
        if !isempty(valid_descent)
            scatter!(p, valid_descent .- 1, max.(objective_gaps[valid_descent], 1e-15),
                    color=:green, marker=:circle, markersize=4,
                    label="Descent Steps")
        end
    end

    if !isempty(rpb.indices_of_null_steps)
        valid_null = [i for i in rpb.indices_of_null_steps if i <= length(objective_gaps)]
        if !isempty(valid_null)
            scatter!(p, valid_null .- 1, max.(objective_gaps[valid_null], 1e-15),
                    color=:orange, marker=:square, markersize=3,
                    label="Null Steps")
        end
    end

    if !isempty(rpb.indices_of_proximal_doubling_steps)
        valid_doubling = [i for i in rpb.indices_of_proximal_doubling_steps if i <= length(objective_gaps)]
        if !isempty(valid_doubling)
            scatter!(p, valid_doubling .- 1, max.(objective_gaps[valid_doubling], 1e-15),
                    color=:red, marker=:uptriangle, markersize=3,
                    label="Proximal Doubling Steps")
        end
    end

    return p
end

"""
Run experiment on SPD manifold
"""
function run_spd_experiment(dimension=3, N=50)
    println("=" ^ 60)
    println("SPD Manifold Experiment (dimension $dimension)")
    println("=" ^ 60)

    manifold = SymmetricPositiveDefinite(dimension)
    data_points, weights = generate_median_data(manifold, N)

    true_median, true_min_obj = find_true_median(data_points, weights, manifold)

    rpb_result = run_rpb_method(data_points, weights, manifold, true_min_obj)
    gd_result = run_gradient_descent_tracked(data_points, weights, manifold, true_min_obj)

    p1, p2 = create_comparison_plots(rpb_result, gd_result, "SPD", dimension)

    # Print summary
    println()
    println("Results Summary:")
    println("=" ^ 40)
    @printf "RPB - Final gap: %.2e, Iterations: %d\\n" rpb_result.objective_history[end] length(rpb_result.objective_history)-1
    @printf "RPB - Descent: %d, Null: %d, Doubling: %d\\n" length(rpb_result.indices_of_descent_steps) length(rpb_result.indices_of_null_steps) length(rpb_result.indices_of_proximal_doubling_steps)

    gd_final, gd_objectives, gd_gaps = gd_result
    @printf "GD  - Final gap: %.2e, Iterations: %d\\n" gd_gaps[end] length(gd_gaps)

    return rpb_result, gd_result, p1, p2
end

"""
Main function
"""
function main()
    println("Simple Algorithm Comparison for Riemannian Median")
    println("RPB vs Gradient Descent on SPD Manifolds")
    println()

    dimensions = [2, 3, 5]
    results = []

    for dim in dimensions
        rpb_result, gd_result, p1, p2 = run_spd_experiment(dim, 50)
        push!(results, (dim, rpb_result, gd_result, p1, p2))

        # Save plots
        savefig(p1, "simple_rpb_detailed_spd_$dim.png")
        savefig(p2, "simple_algorithm_comparison_spd_$dim.png")
        println("Plots saved for SPD($dim)")
        println()
    end

    println("Simple comparison experiment completed!")
    return results
end

# Run the experiment
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end