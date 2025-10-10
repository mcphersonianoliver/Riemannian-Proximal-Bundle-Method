"""
Extract exact data points from Julia version for Python comparison
"""

using Random
using LinearAlgebra
using Printf
using Manifolds

function extract_julia_data_points()
    println("Extracting exact data points from Julia experiment")

    # Exact same setup as corrected Julia version
    n = 2
    num_data_points = 5
    manifold = SymmetricPositiveDefinite(n)

    # Generate test data using same approach as Julia version
    Random.seed!(42)
    base_point = rand(manifold)  # Random SPD point
    data_points = []

    println("Base point:")
    for i in 1:size(base_point, 1)
        for j in 1:size(base_point, 2)
            @printf "%.16f " base_point[i, j]
        end
        println()
    end

    # Generate data around base point (same as corrected Julia version)
    for i in 1:num_data_points
        # Generate random tangent vector at base point
        tangent_vec = rand(manifold; vector_at=base_point)
        # Normalize and scale (same scaling as Julia version)
        tangent_norm = Manifolds.norm(manifold, base_point, tangent_vec)
        if tangent_norm > 1e-12
            tangent_vec = tangent_vec / tangent_norm
            scale_factor = 0.5 + 1.5 * rand()  # Random scaling [0.5, 2.0]
            tangent_vec = tangent_vec * scale_factor
        end
        # Retract to manifold
        point = exp(manifold, base_point, tangent_vec)
        push!(data_points, point)

        println("Data point $i:")
        for row in 1:size(point, 1)
            for col in 1:size(point, 2)
                @printf "%.16f " point[row, col]
            end
            println()
        end
    end

    # Also get the initial point for bundle method
    Random.seed!(123)  # Different seed for initial point
    initial_point = rand(manifold)

    println("Initial point:")
    for i in 1:size(initial_point, 1)
        for j in 1:size(initial_point, 2)
            @printf "%.16f " initial_point[i, j]
        end
        println()
    end

    return data_points, base_point, initial_point
end

extract_julia_data_points()