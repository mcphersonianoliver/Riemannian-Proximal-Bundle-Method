using LinearAlgebra
include("SPDManifold.jl")

"""
    RiemannianMedianProblem

Problem setup for computing the Riemannian median on SPD manifolds.
"""
struct RiemannianMedianProblem
    manifold::SPDManifold
    data_points::Vector{Matrix{Float64}}
    weights::Vector{Float64}

    function RiemannianMedianProblem(manifold::SPDManifold, data_points::Vector{Matrix{Float64}}, weights=nothing)
        if weights === nothing
            weights = ones(length(data_points)) / length(data_points)
        end
        new(manifold, data_points, weights)
    end
end

"""
    objective_function(problem::RiemannianMedianProblem, X)

Compute the Riemannian median objective function:
f(X) = Σᵢ wᵢ * d(X, Yᵢ)
where d(X, Yᵢ) is the Riemannian distance between X and data point Yᵢ
"""
function objective_function(problem::RiemannianMedianProblem, X)
    total = 0.0
    for (i, Y) in enumerate(problem.data_points)
        dist = distance_spd(problem.manifold, X, Y)
        total += problem.weights[i] * dist
    end
    return total
end

"""
    subgradient_function(problem::RiemannianMedianProblem, X)

Compute a subgradient of the Riemannian median objective function.
The subgradient at X is: ∇f(X) = Σᵢ wᵢ * (log_X(Yᵢ) / ||log_X(Yᵢ)||_X)
"""
function subgradient_function(problem::RiemannianMedianProblem, X)
    subgrad = zeros(size(X))

    for (i, Y) in enumerate(problem.data_points)
        log_XY = log_map(problem.manifold, X, Y)
        norm_log_XY = norm(problem.manifold, X, log_XY)

        if norm_log_XY > 1e-12  # Avoid division by zero
            unit_log_XY = log_XY / norm_log_XY
            subgrad += problem.weights[i] * unit_log_XY
        end
    end

    return subgrad
end

"""
    generate_test_data(manifold::SPDManifold, num_points::Int, center_matrix=nothing)

Generate test data points around a center for testing.
Uses the same approach as the previous experiments.
"""
function generate_test_data(manifold::SPDManifold, num_points::Int, center_matrix=nothing)
    if center_matrix === nothing
        center_matrix = random_spd_matrix(manifold.n)  # Random base point
    end

    data_points = Matrix{Float64}[]

    # Generate preset scalings for reproducibility
    Random.seed!(123)  # Different seed for data generation
    preset_scalings = [0.5 + 1.5 * rand() for _ in 1:num_points]  # Uniform(0.5, 2.0)
    Random.seed!(42)  # Reset to original seed

    for i in 1:num_points
        # Generate random tangent vector
        V = randn(manifold.n, manifold.n)
        V = (V + V') / 2  # Make symmetric

        # Normalize and scale the tangent vector
        V_norm = norm(manifold, center_matrix, V)
        if V_norm > 1e-12
            V = V / V_norm  # Normalize
            V = V * preset_scalings[i]  # Scale by preset amount
        end

        # Retract to manifold
        point = exp_map(manifold, center_matrix, V)
        push!(data_points, point)
    end

    return data_points, center_matrix
end

"""
    compute_true_median(problem::RiemannianMedianProblem, initial_guess=nothing; max_iter=100, tol=1e-8)

Compute the true Riemannian median using gradient descent for comparison.
"""
function compute_true_median(problem::RiemannianMedianProblem, initial_guess=nothing; max_iter=100, tol=1e-8)
    if initial_guess === nothing
        initial_guess = Matrix{Float64}(I, problem.manifold.n, problem.manifold.n)
    end

    X = copy(initial_guess)

    for iter in 1:max_iter
        grad = subgradient_function(problem, X)
        grad_norm = norm(problem.manifold, X, grad)

        if grad_norm < tol
            break
        end

        # Simple gradient descent step
        step_size = 0.1 / iter
        X_new = exp_map(problem.manifold, X, -step_size * grad)
        X = X_new
    end

    return X
end