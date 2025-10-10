using LinearAlgebra

"""
    SPDManifold

Simple implementation of the SPD (Symmetric Positive Definite) manifold
for testing the Riemannian Proximal Bundle method.
"""
struct SPDManifold
    n::Int  # dimension of matrices (n×n)
end

"""
    inner_product(M::SPDManifold, X, U, V)

Compute the Riemannian inner product at point X between tangent vectors U and V.
For SPD manifold: ⟨U,V⟩_X = tr(X^{-1} U X^{-1} V)
"""
function inner_product(M::SPDManifold, X, U, V)
    X_inv = inv(X)
    return tr(X_inv * U * X_inv * V)
end

"""
    norm(M::SPDManifold, X, V)

Compute the Riemannian norm of tangent vector V at point X.
"""
function norm(M::SPDManifold, X, V)
    return sqrt(inner_product(M, X, V, V))
end

"""
    exp_map(M::SPDManifold, X, V)

Exponential map: retract tangent vector V at point X to the manifold.
For SPD: Exp_X(V) = X^{1/2} exp(X^{-1/2} V X^{-1/2}) X^{1/2}
"""
function exp_map(M::SPDManifold, X, V)
    sqrt_X = sqrt(X)
    inv_sqrt_X = inv(sqrt_X)
    return sqrt_X * exp(inv_sqrt_X * V * inv_sqrt_X) * sqrt_X
end

"""
    log_map(M::SPDManifold, X, Y)

Logarithmic map: compute tangent vector from X to Y.
For SPD: Log_X(Y) = X^{1/2} log(X^{-1/2} Y X^{-1/2}) X^{1/2}
"""
function log_map(M::SPDManifold, X, Y)
    sqrt_X = sqrt(X)
    inv_sqrt_X = inv(sqrt_X)
    return sqrt_X * log(inv_sqrt_X * Y * inv_sqrt_X) * sqrt_X
end

"""
    parallel_transport(M::SPDManifold, X, Y, V)

Parallel transport tangent vector V from X to Y.
For SPD manifold, we'll use a simple approximation for this test.
"""
function parallel_transport(M::SPDManifold, X, Y, V)
    # Simple approximation: V (identity transport)
    # For a more accurate implementation, use the Schild's ladder or other methods
    return V
end

"""
    random_spd_matrix(n::Int)

Generate a random SPD matrix of size n×n.
"""
function random_spd_matrix(n::Int)
    A = randn(n, n)
    return A * A' + I  # A*A' + I is always SPD
end

"""
    distance_spd(M::SPDManifold, X, Y)

Compute the Riemannian distance between two SPD matrices.
"""
function distance_spd(M::SPDManifold, X, Y)
    return norm(M, X, log_map(M, X, Y))
end