"""
Corrected Julia test using Manifolds.jl (equivalent to pymanopt)
with the fixed bundle method implementation
"""

using Random
using LinearAlgebra
using Printf
using Plots
using Manifolds

# Include the fixed bundle method (with Nothing bugs patched)
include("src/RiemannianProximalBundle.jl")

# Create wrapper manifold for bundle method that provides the expected interface
# The bundle method expects manifold.inner_product and manifold.norm functions
struct ManifoldWrapper
    manifold::SymmetricPositiveDefinite
end

function inner_product(wrapper::ManifoldWrapper, X, U, V)
    return inner(wrapper.manifold, X, U, V)
end

function norm(wrapper::ManifoldWrapper, X, V)
    return Manifolds.norm(wrapper.manifold, X, V)
end

function test_corrected_julia_manifolds()
    """Test corrected Julia implementation using robust Manifolds.jl"""
    println("=" ^ 60)
    println("Testing Corrected Julia with Manifolds.jl")
    println("=" ^ 60)

    # Problem setup - match the Python version exactly
    n = 2
    num_data_points = 5
    manifold = SymmetricPositiveDefinite(n)

    println("Problem Setup:")
    println("  Matrix dimension: $(n)×$(n)")
    println("  Number of data points: $(num_data_points)")

    # Generate test data using same approach as Python/pymanopt
    Random.seed!(42)
    base_point = rand(manifold)  # Random SPD point
    data_points = []

    # Generate data around base point (same as corrected Python version)
    for i in 1:num_data_points
        # Generate random tangent vector at base point
        tangent_vec = rand(manifold; vector_at=base_point)
        # Normalize and scale (same scaling as Python version)
        tangent_norm = Manifolds.norm(manifold, base_point, tangent_vec)
        if tangent_norm > 1e-12
            tangent_vec = tangent_vec / tangent_norm
            scale_factor = 0.5 + 1.5 * rand()  # Random scaling [0.5, 2.0]
            tangent_vec = tangent_vec * scale_factor
        end
        # Retract to manifold
        point = exp(manifold, base_point, tangent_vec)
        push!(data_points, point)
    end

    println("  Base point:")
    println("$(base_point)")

    # Define cost function (Riemannian median)
    function cost_function(point)
        return sum([distance(manifold, point, x) for x in data_points]) / length(data_points)
    end

    # Define subgradient function
    function subgradient_function(point)
        grad = zeros(size(point))
        for x in data_points
            log_point = log(manifold, point, x)
            norm_log = Manifolds.norm(manifold, point, log_point)
            if norm_log > 1e-12
                grad += -(log_point) / norm_log
            end
        end
        return grad / length(data_points)
    end

    # Phase 1: Find true median using multiple starting points
    println("\\n=== PHASE 1: Finding True Median (Manifolds.jl) ===")

    candidates = []

    # Try starting from each data point
    for (i, start_point) in enumerate(data_points)
        result = robust_gradient_descent_manifolds(cost_function, subgradient_function, start_point, manifold)
        obj_val = cost_function(result)
        push!(candidates, (obj_val, result, "data_point_$i"))
        @printf "  From data point %d: objective = %.10f\\n" i obj_val
    end

    # Try starting from random points
    for i in 1:3
        start_point = rand(manifold)
        result = robust_gradient_descent_manifolds(cost_function, subgradient_function, start_point, manifold)
        obj_val = cost_function(result)
        push!(candidates, (obj_val, result, "random_$i"))
        @printf "  From random point %d: objective = %.10f\\n" i obj_val
    end

    # Find the best candidate
    sort!(candidates, by=x->x[1])
    best_obj, best_point, best_method = candidates[1]

    @printf "\\nBest result: objective = %.10f (from %s)\\n" best_obj best_method

    # Check gradient norm at best point
    grad = subgradient_function(best_point)
    grad_norm = Manifolds.norm(manifold, best_point, grad)
    @printf "Gradient norm at best point: %.10f\\n" grad_norm

    true_median = best_point
    true_min_obj = best_obj

    # Setup initial point for bundle method test
    Random.seed!(123)  # Different seed for initial point
    initial_point = rand(manifold)
    initial_objective = cost_function(initial_point)
    initial_subgradient = subgradient_function(initial_point)
    initial_gap = initial_objective - true_min_obj

    println("\\nInitial Setup:")
    println("  Initial objective: $(initial_objective)")
    println("  True minimum objective: $(true_min_obj)")
    println("  Initial gap: $(initial_gap)")

    if initial_gap <= 0
        println("❌ ERROR: Initial gap is not positive!")
        return nothing
    end

    # Create wrapper instance
    manifold_wrapper = ManifoldWrapper(manifold)

    # Define wrapper functions for bundle method
    function objective_wrapper(X)
        return cost_function(X)
    end

    function subgradient_wrapper(X)
        return subgradient_function(X)
    end

    function retraction_wrapper(X, V)
        return exp(manifold, X, V)
    end

    function transport_wrapper(X, Y, V)
        return parallel_transport_to(manifold, X, V, Y)
    end

    # Create and run proximal bundle algorithm
    println("\\n" * "=" ^ 40)
    println("Running Proximal Bundle Algorithm (Manifolds.jl)")
    println("=" ^ 40)

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
        proximal_parameter=0.1,  # Same as Python corrected version
        trust_parameter=0.2,     # Same as Python corrected version
        max_iter=50,
        tolerance=1e-8,
        adaptive_proximal=true,  # Same as Python corrected version
        know_minimizer=true
    )

    run!(rpb)

    println("\\nAlgorithm Results:")
    println("  Final objective: $(rpb.raw_objective_history[end])")
    println("  Final gap: $(rpb.objective_history[end])")
    println("  Number of iterations: $(length(rpb.objective_history) - 1)")
    println("  Descent steps: $(length(rpb.indices_of_descent_steps))")
    println("  Null steps: $(length(rpb.indices_of_null_steps))")
    println("  Proximal doubling steps: $(length(rpb.indices_of_proximal_doubling_steps))")

    # Check convergence
    if rpb.objective_history[end] < 1e-6
        println("  ✅ Algorithm converged successfully!")
    else
        println("  ⚠️  Algorithm may not have fully converged")
    end

    println("\\nObjective gap history (first 10): $(rpb.objective_history[1:min(10, end)])")

    # Check if gaps are decreasing
    gaps_decreasing = all(rpb.objective_history[i] >= rpb.objective_history[i+1] for i in 1:(length(rpb.objective_history)-1))
    println("Gaps monotonically decreasing: $gaps_decreasing")

    if !gaps_decreasing
        println("⚠️  Gaps are not monotonically decreasing - may indicate issues")
    else
        println("✅ Gaps are properly decreasing")
    end

    return rpb
end

function robust_gradient_descent_manifolds(cost, subgradient, start_point, manifold, max_iter=500)
    """Robust gradient descent using Manifolds.jl operations"""
    X = copy(start_point)

    for iter_num in 1:max_iter
        grad = subgradient(X)
        grad_norm = Manifolds.norm(manifold, X, grad)

        if grad_norm < 1e-10
            break
        end

        # Adaptive step size with backtracking
        step_size = 0.1 / (1 + iter_num * 0.01)
        current_obj = cost(X)

        for alpha in [step_size, step_size/2, step_size/4]
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

function plot_convergence_manifolds_corrected(rpb)
    """Plot convergence with conditional log scale"""
    objective_gaps = rpb.objective_history
    iterations = 0:(length(objective_gaps)-1)

    # Check if we can use log scale
    use_log_scale = all(objective_gaps .> 0)

    p1 = plot(iterations, objective_gaps,
              label="Objective Gap",
              linewidth=2,
              color=:blue,
              xlabel="Iteration Number",
              ylabel=use_log_scale ? "Objective Gap (log scale)" : "Objective Gap",
              title="Convergence (Julia + Manifolds.jl, Corrected)",
              grid=true,
              gridwidth=1,
              gridalpha=0.3)

    if use_log_scale
        plot!(p1, yscale=:log10)
    end

    # Add markers for different step types
    if !isempty(rpb.indices_of_descent_steps)
        valid_descent = [i for i in rpb.indices_of_descent_steps if i < length(objective_gaps)]
        if !isempty(valid_descent)
            scatter!(p1, valid_descent, objective_gaps[valid_descent .+ 1],
                    color=:green, marker=:circle, markersize=4,
                    label="Descent Steps")
        end
    end

    if !isempty(rpb.indices_of_null_steps)
        valid_null = [i for i in rpb.indices_of_null_steps if i < length(objective_gaps)]
        if !isempty(valid_null)
            scatter!(p1, valid_null, objective_gaps[valid_null .+ 1],
                    color=:orange, marker=:square, markersize=3,
                    label="Null Steps")
        end
    end

    if !isempty(rpb.indices_of_proximal_doubling_steps)
        valid_doubling = [i for i in rpb.indices_of_proximal_doubling_steps if i < length(objective_gaps)]
        if !isempty(valid_doubling)
            scatter!(p1, valid_doubling, objective_gaps[valid_doubling .+ 1],
                    color=:red, marker=:uptriangle, markersize=3,
                    label="Proximal Doubling Steps")
        end
    end

    # Proximal parameter plot
    p2 = plot(0:(length(rpb.proximal_parameter_history)-1), rpb.proximal_parameter_history,
              label="Proximal Parameter (ρ)",
              linewidth=2,
              color=:purple,
              xlabel="Iteration Number",
              ylabel="Proximal Parameter",
              title="Proximal Parameter Evolution",
              grid=true,
              gridwidth=1,
              gridalpha=0.3)

    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))

    # Print summary
    println("\\nConvergence Summary:")
    println("="^50)
    println("Final objective gap: $(objective_gaps[end])")
    println("Total iterations: $(length(objective_gaps) - 1)")
    println("Descent steps: $(length(rpb.indices_of_descent_steps))")
    println("Null steps: $(length(rpb.indices_of_null_steps))")
    println("Proximal doubling steps: $(length(rpb.indices_of_proximal_doubling_steps))")
    println("Final proximal parameter: $(rpb.proximal_parameter_history[end])")
    println("="^50)

    return combined_plot
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    rpb = test_corrected_julia_manifolds()

    if rpb !== nothing
        println("\\n" * "=" ^ 60)
        println("Creating convergence plots...")
        println("=" ^ 60)

        convergence_plot = plot_convergence_manifolds_corrected(rpb)
        savefig(convergence_plot, "corrected_julia_manifolds_rpb_convergence.png")
        println("Plot saved as 'corrected_julia_manifolds_rpb_convergence.png'")

        println("\\n" * "=" ^ 60)
        println("Test completed! Corrected Julia + Manifolds.jl implementation.")
        println("This should now match the Python/pymanopt robustness and behavior.")
        println("=" ^ 60)
    end
end