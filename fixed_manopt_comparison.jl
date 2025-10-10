"""
Fixed Manopt.jl Algorithm Comparison
Properly implements Manopt.jl algorithms with correct interfaces
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
Run Manopt.jl algorithms with proper recording
"""
function run_manopt_algorithms(data_points, weights, manifold, true_min_obj, max_iter=1000)
    println("  Running Manopt.jl algorithms...")

    f, ∂f = create_manopt_functions(data_points, weights)

    Random.seed!(123)
    p0 = rand(manifold)

    results = Dict()

    # Subgradient Method
    try
        println("    Running Subgradient Method...")

        # Create recording for tracking
        rec = [:Iteration, :Cost]

        result_sgm = subgradient_method(
            manifold, f, ∂f, p0;
            stepsize=ConstantStepsize(0.01),
            stopping_criterion=StopAfterIteration(max_iter) | StopWhenGradientNormLess(1e-8),
            record=rec,
            return_state=true
        )

        # Extract recorded data
        records = get_record(result_sgm)
        iterations = [r[1] for r in records]
        costs = [r[2] for r in records]
        gaps = costs .- true_min_obj

        results[:SGM] = (get_solver_result(result_sgm), iterations, costs, gaps)
        println("    SGM completed successfully")
    catch e
        println("    SGM failed: $e")
        results[:SGM] = nothing
    end

    # Convex Bundle Method
    try
        println("    Running Convex Bundle Method...")

        rec = [:Iteration, :Cost]

        result_cbm = convex_bundle_method(
            manifold, f, ∂f, p0;
            stopping_criterion=StopAfterIteration(max_iter) | StopWhenLagrangeMultiplierLess(1e-8),
            record=rec,
            return_state=true
        )

        records = get_record(result_cbm)
        iterations = [r[1] for r in records]
        costs = [r[2] for r in records]
        gaps = costs .- true_min_obj

        results[:CBM] = (get_solver_result(result_cbm), iterations, costs, gaps)
        println("    CBM completed successfully")
    catch e
        println("    CBM failed: $e")
        results[:CBM] = nothing
    end

    # Proximal Bundle Method
    try
        println("    Running Proximal Bundle Method...")

        rec = [:Iteration, :Cost]

        result_pbm = proximal_bundle_method(
            manifold, f, ∂f, p0;
            stopping_criterion=StopAfterIteration(max_iter) | StopWhenLagrangeMultiplierLess(1e-8),
            record=rec,
            return_state=true
        )

        records = get_record(result_pbm)
        iterations = [r[1] for r in records]
        costs = [r[2] for r in records]
        gaps = costs .- true_min_obj

        results[:PBM] = (get_solver_result(result_pbm), iterations, costs, gaps)
        println("    PBM completed successfully")
    catch e
        println("    PBM failed: $e")
        results[:PBM] = nothing
    end

    return results
end

"""
Run our RPB method
"""
function run_our_rpb_method(data_points, weights, manifold, true_min_obj, max_iter=1000)
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
function create_comparison_plots(our_rpb, manopt_results, manifold_name, dimension)
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

    # Colors for different algorithms
    colors = Dict(:SGM => :blue, :CBM => :green, :PBM => :purple)
    names = Dict(:SGM => "Subgradient Method", :CBM => "Convex Bundle Method", :PBM => "Proximal Bundle Method")

    # Plot Manopt algorithms
    for (alg_name, result) in manopt_results
        if result !== nothing
            final_point, iterations, costs, gaps = result

            # Only plot positive gaps
            valid_gaps = max.(gaps, 1e-15)

            plot!(p, iterations .- 1, valid_gaps,  # iterations start from 1, plot from 0
                  label=names[alg_name],
                  linewidth=2,
                  color=colors[alg_name])
        end
    end

    return p
end

"""
Run experiment on SPD manifold
"""
function run_spd_experiment(dimension=3, N=50, max_iter=200)
    println("=" ^ 60)
    println("Fixed Manopt Comparison: SPD($dimension)")
    println("=" ^ 60)

    manifold = SymmetricPositiveDefinite(dimension)
    data_points, weights = generate_median_data(manifold, N)

    true_median, true_min_obj = find_true_median(data_points, weights, manifold)
    println("    True minimum objective: $true_min_obj")

    our_rpb = run_our_rpb_method(data_points, weights, manifold, true_min_obj, max_iter)
    manopt_results = run_manopt_algorithms(data_points, weights, manifold, true_min_obj, max_iter)

    p = create_comparison_plots(our_rpb, manopt_results, "SPD", dimension)

    # Print summary
    println()
    println("Results Summary:")
    println("=" ^ 40)
    @printf "Our RPB - Final gap: %.2e, Iterations: %d\\n" our_rpb.objective_history[end] length(our_rpb.objective_history)-1
    @printf "Our RPB - Descent: %d, Null: %d, Doubling: %d\\n" length(our_rpb.indices_of_descent_steps) length(our_rpb.indices_of_null_steps) length(our_rpb.indices_of_proximal_doubling_steps)

    for (alg_name, result) in manopt_results
        if result !== nothing
            final_point, iterations, costs, gaps = result
            @printf "%s - Final gap: %.2e, Iterations: %d\\n" alg_name gaps[end] length(iterations)
        else
            println("$alg_name - Failed to run")
        end
    end

    return our_rpb, manopt_results, p
end

"""
Main function
"""
function main()
    println("Fixed Manopt.jl Algorithm Comparison")
    println("Including proper Manopt.jl algorithm implementations")
    println()

    dimensions = [2, 3]  # Start with smaller dimensions to test
    results = []

    for dim in dimensions
        our_rpb, manopt_results, p = run_spd_experiment(dim, 30, 150)  # Smaller problem sizes
        push!(results, (dim, our_rpb, manopt_results, p))

        # Save plots
        savefig(p, "fixed_manopt_comparison_spd_$dim.png")
        println("Plot saved for SPD($dim)")
        println()
    end

    println("Fixed Manopt comparison completed!")
    return results
end

# Run the experiment
if abspath(PROGRAM_FILE) == @__FILE__
    results = main()
end