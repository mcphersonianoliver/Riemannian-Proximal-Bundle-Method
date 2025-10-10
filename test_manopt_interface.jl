"""
Test Manopt.jl interface to understand correct usage
"""

using Manifolds
using Manopt
using Random

# Simple test on a small problem
Random.seed!(42)
M = SymmetricPositiveDefinite(2)

# Generate simple test data
data = [rand(M) for _ in 1:3]
weights = [1/3, 1/3, 1/3]

# Simple objective function
f(M, p) = sum(weights[i] * distance(M, p, data[i]) for i in 1:length(data))

# Simple subgradient function
function ∂f(M, p)
    grad = zero_vector(M, p)
    for i in 1:length(data)
        d = distance(M, p, data[i])
        if d > 1e-12
            v = log(M, p, data[i])
            grad += -weights[i] * v / Manifolds.norm(M, p, v)
        end
    end
    return grad
end

# Test initial point
p0 = rand(M)
println("Initial objective: ", f(M, p0))

# Test 1: Try subgradient method with different stepsize approaches
println("\\nTesting Subgradient Method...")

try
    # Try with default stepsize
    result1 = subgradient_method(M, f, ∂f, p0)
    println("✅ SGM with default stepsize: success")
    println("   Final objective: ", f(M, result1))
catch e
    println("❌ SGM with default stepsize failed: ", e)
end

try
    # Try with explicit stepsize parameter
    result2 = subgradient_method(M, f, ∂f, p0; stepsize=0.01)
    println("✅ SGM with stepsize=0.01: success")
    println("   Final objective: ", f(M, result2))
catch e
    println("❌ SGM with stepsize=0.01 failed: ", e)
end

# Test 2: Try convex bundle method
println("\\nTesting Convex Bundle Method...")

try
    result3 = convex_bundle_method(M, f, ∂f, p0)
    println("✅ CBM with defaults: success")
    println("   Final objective: ", f(M, result3))
catch e
    println("❌ CBM with defaults failed: ", e)
end

# Test 3: Try proximal bundle method
println("\\nTesting Proximal Bundle Method...")

try
    result4 = proximal_bundle_method(M, f, ∂f, p0)
    println("✅ PBM with defaults: success")
    println("   Final objective: ", f(M, result4))
catch e
    println("❌ PBM with defaults failed: ", e)
end

# Test recording functionality
println("\\nTesting with recording...")

try
    result5 = subgradient_method(
        M, f, ∂f, p0;
        record=[:Iteration, :Cost],
        return_state=true
    )

    records = get_record(result5)
    println("✅ SGM with recording: success")
    println("   Recorded ", length(records), " iterations")
    if length(records) > 0
        println("   First record: ", records[1])
        println("   Last record: ", records[end])
    end
catch e
    println("❌ SGM with recording failed: ", e)
end