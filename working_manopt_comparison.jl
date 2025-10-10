"""
Working Manopt.jl Comparison
Uses only the algorithms that work properly in current Manopt.jl version
"""

using Random
using LinearAlgebra
using Printf
using Plots
using Manifolds
using Manopt

# Include our bundle method
include("src/RiemannianProximalBundle.jl")

# Create wrapper manifold for our bundle method
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
function generate_median_data(manifold, N=50, seed=57)
    Random.seed!(seed)
    data_points = [rand(manifold) for _ in 1:N]
    weights = fill(1.0/N, N)
    return data_points, weights
end

"""
Create proper Manopt.jl functions
"""
function create_manopt_functions(data_points, weights)
    # Manopt objective function: (M, p) -> Real
    function f(M, p)
        return sum(weights[i] * distance(M, p, data_points[i]) for i in 1:length(data_points))
    end

    # Manopt subgradient function: (M, p) -> TangentVector
    function ∂f(M, p)
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

    return f, ∂f
end

"""
Find true minimum using robust gradient descent
"""
function find_true_median(data_points, weights, manifold)
    println("  Finding true median...")

    # Create functions for our method
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

    candidates = []
    # Try multiple starting points
    for i in 1:min(5, length(data_points))
        result = robust_gradient_descent(cost_func, subgrad_func, data_points[i], manifold, 1000)
        obj_val = cost_func(result)
        push!(candidates, (obj_val, result))
    end

    for i in 1:3
        start_point = rand(manifold)
        result = robust_gradient_descent(cost_func, subgrad_func, start_point, manifold, 1000)
        obj_val = cost_func(result)
        push!(candidates, (obj_val, result))
    end

    sort!(candidates, by=x->x[1])
    return candidates[1][2], candidates[1][1]
end

"""
Robust gradient descent helper
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
Run Manopt.jl Subgradient Method (the one that works)
"""
function run_manopt_subgradient(data_points, weights, manifold, true_min_obj, max_iter=500)
    println("  Running Manopt.jl Subgradient Method...")

    f, ∂f = create_manopt_functions(data_points, weights)

    Random.seed!(123)
    p0 = rand(manifold)

    try
        # Run with recording to track convergence
        result = subgradient_method(
            manifold, f, ∂f, p0;
            stopping_criterion=StopAfterIteration(max_iter) | StopWhenGradientNormLess(1e-8),
            record=[:Iteration, :Cost],
            return_state=true
        )

        # Extract recorded data
        records = get_record(result)
        final_point = get_solver_result(result)

        iterations = [r[1] for r in records]
        costs = [r[2] for r in records]
        gaps = costs .- true_min_obj

        println("    Manopt SGM completed successfully")
        return (final_point, iterations, costs, gaps)
    catch e
        println("    Manopt SGM failed: $e")
        return nothing
    end
end

"""
Run our custom gradient descent with tracking
"""
function run_custom_gradient_descent(data_points, weights, manifold, true_min_obj, max_iter=500)
    println("  Running Custom Gradient Descent...")

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

    Random.seed!(123)
    X = rand(manifold)

    iterations = Int[]
    costs = Float64[]
    gaps = Float64[]

    for iter in 1:max_iter
        current_obj = cost_func(X)
        gap = current_obj - true_min_obj

        push!(iterations, iter)
        push!(costs, current_obj)
        push!(gaps, gap)

        if gap < 1e-8
            println("    Custom GD converged at iteration $iter")
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

    return (X, iterations, costs, gaps)
end

"""
Run our RPB method
"""
function run_our_rpb_method(data_points, weights, manifold, true_min_obj, max_iter=500)
    println("  Running Our Riemannian Proximal Bundle Method...")

    # Create functions for our method
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
Create comprehensive comparison plots
"""
function create_comparison_plots(our_rpb, manopt_sgm, custom_gd, manifold_name, dimension)
    # Main comparison plot
    p = plot(xlabel="Iteration Number",
             ylabel="Objective Gap (log scale)",
             title="Algorithm Comparison: $manifold_name($dimension)",
             yscale=:log10,
             grid=true,
             legend=:topright)

    # Plot our RPB
    rpb_gaps = our_rpb.objective_history
    rpb_iters = 0:(length(rpb_gaps)-1)
    plot!(p, rpb_iters, max.(rpb_gaps, 1e-15),
          label="Our RPB Method",
          linewidth=3,
          color=:red)

    # Plot Manopt Subgradient Method
    if manopt_sgm !== nothing
        final_point, iterations, costs, gaps = manopt_sgm
        valid_gaps = max.(gaps, 1e-15)
        plot!(p, iterations .- 1, valid_gaps,  # Convert to 0-based indexing
              label="Manopt Subgradient Method",
              linewidth=2,
              color=:blue)
    end

    # Plot Custom Gradient Descent
    if custom_gd !== nothing
        final_point_gd, iterations_gd, costs_gd, gaps_gd = custom_gd
        valid_gaps_gd = max.(gaps_gd, 1e-15)
        plot!(p, iterations_gd .- 1, valid_gaps_gd,  # Convert to 0-based indexing
              label="Custom Gradient Descent",
              linewidth=2,
              color=:green,
              linestyle=:dash)
    end

    return p
end

"""
Run experiment on SPD manifold
"""
function run_spd_experiment(dimension=3, N=30, max_iter=200)
    println("=" ^ 60)
    println("Working Manopt Comparison: SPD($dimension)")
    println("=" ^ 60)

    manifold = SymmetricPositiveDefinite(dimension)
    data_points, weights = generate_median_data(manifold, N)

    true_median, true_min_obj = find_true_median(data_points, weights, manifold)
    println("    True minimum objective: $true_min_obj")

    our_rpb = run_our_rpb_method(data_points, weights, manifold, true_min_obj, max_iter)
    manopt_sgm = run_manopt_subgradient(data_points, weights, manifold, true_min_obj, max_iter)
    custom_gd = run_custom_gradient_descent(data_points, weights, manifold, true_min_obj, max_iter)

    p = create_comparison_plots(our_rpb, manopt_sgm, custom_gd, "SPD", dimension)

    # Print summary
    println()
    println("Results Summary:")
    println("=" ^ 40)
    @printf "Our RPB     - Final gap: %.2e, Iterations: %d\\n" our_rpb.objective_history[end] length(our_rpb.objective_history)-1
    @printf "Our RPB     - Descent: %d, Null: %d, Doubling: %d\\n" length(our_rpb.indices_of_descent_steps) length(our_rpb.indices_of_null_steps) length(our_rpb.indices_of_proximal_doubling_steps)

    if manopt_sgm !== nothing
        final_point, iterations, costs, gaps = manopt_sgm
        @printf "Manopt SGM  - Final gap: %.2e, Iterations: %d\\n" gaps[end] length(iterations)
    else
        println("Manopt SGM  - Failed to run")
    end

    if custom_gd !== nothing
        final_point_gd, iterations_gd, costs_gd, gaps_gd = custom_gd
        @printf "Custom GD   - Final gap: %.2e, Iterations: %d\\n" gaps_gd[end] length(iterations_gd)
    else
        println("Custom GD   - Failed to run")
    end

    return our_rpb, manopt_sgm, custom_gd, p
end

"""
Main function
"""
function main()
    println("Working Manopt.jl Algorithm Comparison")
    println("Comparing algorithms that actually work in current Manopt.jl")
    println()

    dimensions = [2, 3, 5]  # Test multiple dimensions
    results = []

    for dim in dimensions
        our_rpb, manopt_sgm, custom_gd, p = run_spd_experiment(dim, 30, 150)
        push!(results, (dim, our_rpb, manopt_sgm, custom_gd, p))

        # Save plots
        savefig(p, "working_manopt_comparison_spd_$dim.png")
        println("Plot saved for SPD($dim)")
        println()
    end

    println("Working Manopt comparison completed!")
    return results
end

# Run the experiment
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end