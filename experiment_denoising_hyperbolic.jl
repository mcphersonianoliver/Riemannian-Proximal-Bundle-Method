using PrettyTables
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots, Plots.PlotMeasures, LaTeXStrings
using QuadraticModels, RipQP
using Random, LinearAlgebra, LRUCache
using Statistics
using ManifoldDiff, Manifolds, Manopt, ManoptExamples, ManifoldsBase


include("src/RiemannianProximalBundle.jl")
using .Main: RProximalBundle, run!, plot_objective_versus_iter

pgfplotsx()

# Add inner product function for the RPB solver
inner_product(M, p, v, w) = inner(M, p, v, w)

# --- feature flags (top of file) ---
# const benchmarking  = "--bench" in ARGS || get(ENV, "BENCHMARKING", "") == "1"
# const export_result = "--export-result" in ARGS || get(ENV, "EXPORT_RESULT", "") == "1"
# const export_table  = "--export-table" in ARGS || get(ENV, "EXPORT_TABLE", "") == "1"
# const export_orig   = "--export-orig" in ARGS || get(ENV, "EXPORT_ORIG", "") == "1"
# const create_gifs   = "--create-gifs" in ARGS || get(ENV, "CREATE_GIFS", "") == "1"

# Experiment configuration - Two-phase mode only
const PHASE1_MULTIPLIER = 1  # Run Phase 1 for this many times the normal iterations
const PHASE2_GAP_TOL = 1e-8  # Stop Phase 2 when objective gap reaches this tolerance
const SAVE_FINAL_ITERATIONS = true  # Save final iterations for transport/retraction testing
const PLOT_CURRENT_OBJECTIVE = false  # true: plot current objective gap, false: plot minimum objective gap

println("Configuration:")
# println("  Benchmarking: $benchmarking")
# println("  Export result: $export_result")
# println("  Export tables: $export_table")
# println("  Export original: $export_orig")
# println("  Create GIFs: $create_gifs")
println("  Phase 1 multiplier: $PHASE1_MULTIPLIER")
println("  Phase 2 gap tolerance: $PHASE2_GAP_TOL")
println("  Save final iterations: $SAVE_FINAL_ITERATIONS")
println("  Plot current objective: $PLOT_CURRENT_OBJECTIVE")
println()

# --- experiment setup ---
experiment_name = "denoising_TV_hyperbolic"
results_folder = joinpath(@__DIR__, "Denoising TV Hyperbolic Results")
isdir(results_folder) || mkpath(results_folder)

# --- experiment parameters ---
Random.seed!(42)
num_points = 496
sigma = 0.3 # noise level
alpha = 0.5 # regularization parameter
atol = 1e-8
k_max = 0.0 # maximum sectional curvature
k_min = -1.0 # minimum sectional curvature
max_iter = 100_000 # maximum iterations
extended_max_iter = 1_000_000 # extended maximum iterations for sgm and rbm to find true minimum
back_tracking_fact = 2 # backtracking factor

# --- our algorithm's parameters ---
# RPB solver configuration
rpb_kwargs = (
true_min_obj = 0.0,
proximal_parameter = 1.0/num_points,
trust_parameter = 0.001,
max_iter = max_iter,
tolerance = atol,
know_minimizer = false,
relative_error = false,
retraction_error = 1.0,
transport_error = 1.0,
sectional_curvature = -1.0,
back_tracking_factor = back_tracking_fact,
)

data_color = RGBA{Float64}(colorant"#BBBBBB")
noise_color = RGBA{Float64}(colorant"#33BBEE") # Tol Vibrant Teal
result_color = RGBA{Float64}(colorant"#EE7733") # Tol Vibrant Orange

# --- helper functions for signal (clean and noisy), objective, subgradient, and proximal computations  ---
"""
        artificial_H2_signal(pts, a, b, T)
    Generate a synthetic signal on the Hyperbolic manifold H². 
    The signal consists of segments of geodesics connected at "jump" points.
    Arguments
    - `pts`: Total number of samples in the final discretized signal.
    - `a, b`: The start and end of the signal's domain (e.g., time).
    - `T`: The period of the "jumps" (how often the signal changes state).
"""

function artificial_H2_signal(pts::Integer = 100, a::Real = 0.0, b::Real = 1.0, T::Real = (b-a)/2)
    # 1. Create a temporal grid
    t = range(a, b; length=pts)
    
    # 2. Define a square-wave-like signal in R²: [time, sign(oscillation)]
    # This creates a signal that flips between y=1 and y=-1.
    x = [[s, sign((2*pi/T) *s)] for s in t]
    
    # 3. Identify "Change Points"
    # We only keep points where the signal flips sign (the 'corners').
    # This reduces the signal to just the critical jump locations.
    y = vcat([x[1]],
             [x[i] for i in 2:(length(x) - 1) if (x[i][2] != x[i + 1][2] || x[i][2] != x[i - 1][2])],
             [x[end]])

    # 4. Map R² points to the Hyperbolic Manifold (Lorentz Model)
    # _hyperbolize lifts the [s, ±1] coordinates into R³ such that they lie 
    # on the hyperboloid: -x₀² + x₁² + x₂² = -1.
    y = map(z -> Manifolds._hyperbolize(Hyperbolic(2), z), y)
    
    data = []
    geodesics = []
    
    # 5. Interpolate between Jump Points
    # Calculate the number of points per segment based on period T
    l = Int(round(pts * T / (2 * (b - a))))
    
    for i in 1:2:(length(y) - 1)
        # Connect key points y[i] and y[i+1] with a shortest geodesic.
        # This fills in the 'flat' parts of the signal with hyperbolic straight lines.
        segment = shortest_geodesic(Hyperbolic(2), y[i], y[i + 1], range(0.0, 1.0; length=l))
        append!(data, segment)
        
        # Store segments separately for visualization purposes
        if i + 2 ≤ length(y) - 1
            append!(geodesics, segment)
            
            # Connect to the next jump segment
            next_seg = shortest_geodesic(
                Hyperbolic(2), y[i + 1], y[i + 2], range(0.0, 1.0; length=l)
            )
            append!(geodesics, next_seg)
        end
    end
    
    # data: The full discretized signal (the 'ground truth')
    # geodesics: The segments used to construct the signal
    return data, geodesics
end

"""
        matrixify_Poincare_ball(input)

    Converts an array of point objects (where each point has a `.value` field) 
    into an N x 2 matrix of coordinates.

    This is useful for visualizing points in the Poincaré disk model.
"""

function matrixify_Poincare_ball(input)
    input_x = []
    input_y = []
    
    # Iterate through the manifold points (e.g., points on H²)
    for p in input
        # Extract the 1st and 2nd coordinates from the point's value field
        # Note: In the Poincaré model, these are the (u, v) coordinates 
        # inside the unit disk.
        push!(input_x, p.value[1])
        push!(input_y, p.value[2])
    end
    
    # Horizontally concatenate the x and y vectors into a single N x 2 matrix
    return hcat(input_x, input_y)
end

"""
        f(M, p)
    Objective function for denoising experiment on manifold M.
    Combines a fidelity term (squared distance to noisy data) and a total variation regularization term.
"""

function f(M, p)
    # distance(M, noise, p) on a PowerManifold computes sqrt(sum(d(n_i, p_i)^2))
    # So 0.5 * distance^2 is the standard L2 fidelity term.
    return 1 / length(noise) * (
        0.5 * distance(M, noise, p)^2 + 
        alpha * ManoptExamples.Total_Variation(M, p)
    )
end

# 2. Domain Constraint
# Useful for the Poincaré model to ensure points don't approach the boundary (infinity)
domf(M, p) = distance(M, p, noise) < diameter / 2 ? true : false

# 3. The Subgradient (for Riemannian Subgradient Method)
function subgradient_of_f(M, p)
    subgradient = 1 / length(noise) * (
        # Grad of 0.5 * d(p,q)^2 is -log_p(q)
        ManifoldDiff.grad_distance(M, noise, p) +
        # Subgrad of d(p_i, p_{i+1}) is a unit vector pointing along the geodesic
        # or a zero-vector if points are within 'atol'
        alpha * ManoptExamples.subgrad_Total_Variation(M, p; atol=atol))

    # for numerical stability, project the subgradient onto the tangent space at each point just in case
    project_subgradient = project(M, p, subgradient)

    return project_subgradient
end

# 4. The Proximal Maps (for Cyclic Proximal Point Algorithm)
# Note: In CPPA, λ is the step size provided by the solver.
proxes = (
    # Prox for (1/2n) * d^2: Results in geodesic interpolation between p and noise
    (M, lambda, p) -> ManifoldDiff.prox_distance(M, lambda, noise, p, 2),

    # Prox for (α/n) * TV: This is a complex internal solver (like Condat-Pock)
    # that 'shrinks' the entire signal to be piecewise constant.
    (M, lambda, p) -> ManoptExamples.prox_Total_Variation(M, (alpha * lambda), p),
)

# ============================================================================
# QP Solver Diagnostics - Override Manopt's RipQP extension to intercept
# every QP solve during RCBM and PBA runs.
# ============================================================================
import Manopt: convex_bundle_method_subsolver!, proximal_bundle_method_subsolver!
using SparseArrays: sparse
using LinearAlgebra: tril, cond

mutable struct QPDiagnostics
    call_count::Int
    method_label::String          # "RCBM" or "PBA" -- set before each run
    print_frequency::Int
    # Per-call logs
    status_history::Vector{Symbol}
    iter_history::Vector{Int}
    primal_feas_history::Vector{Float64}
    dual_feas_history::Vector{Float64}
    objective_history::Vector{Float64}
    elapsed_time_history::Vector{Float64}
    # Problem diagnostics
    bundle_size_history::Vector{Int}
    H_cond_history::Vector{Float64}
    # Solution diagnostics
    has_nan_history::Vector{Bool}
    has_inf_history::Vector{Bool}
    lambda_min_history::Vector{Float64}
    lambda_max_history::Vector{Float64}
    simplex_violation_history::Vector{Float64}   # |sum(λ) - 1|
    neg_lambda_count_history::Vector{Int}        # number of λ_i < -eps
    # Pruning diagnostics: track weight of newest element (last index)
    newest_lambda_history::Vector{Float64}       # λ[end] -- weight QP assigns to newest bundle element
    newest_would_be_pruned_history::Vector{Bool}  # whether λ[end] ≤ eps() (i.e., will be pruned)
end

function QPDiagnostics(; print_frequency::Int=100)
    QPDiagnostics(0, "QP", print_frequency,
        Symbol[], Int[], Float64[], Float64[], Float64[], Float64[],
        Int[], Float64[],
        Bool[], Bool[], Float64[], Float64[], Float64[], Int[],
        Float64[], Bool[])
end

function reset_qp_diagnostics!(d::QPDiagnostics; label::String="QP", print_frequency::Int=d.print_frequency)
    d.call_count = 0
    d.method_label = label
    d.print_frequency = print_frequency
    for fn in fieldnames(QPDiagnostics)
        v = getfield(d, fn)
        v isa AbstractVector && empty!(v)
    end
end

const QP_DIAG = QPDiagnostics(print_frequency=100)

function log_qp_solve!(d::QPDiagnostics, stats, λ, H, bundle_size::Int)
    d.call_count += 1

    # --- Solver output ---
    push!(d.status_history, stats.status)
    push!(d.iter_history, stats.iter)
    push!(d.primal_feas_history, Float64(stats.primal_feas))
    push!(d.dual_feas_history, Float64(stats.dual_feas))
    push!(d.objective_history, Float64(stats.objective))
    push!(d.elapsed_time_history, stats.elapsed_time)

    # --- Problem conditioning ---
    push!(d.bundle_size_history, bundle_size)
    H_cond = try Float64(cond(H)) catch; NaN end
    push!(d.H_cond_history, H_cond)

    # --- Solution health ---
    push!(d.has_nan_history, any(isnan, λ))
    push!(d.has_inf_history, any(isinf, λ))
    push!(d.lambda_min_history, minimum(λ))
    push!(d.lambda_max_history, maximum(λ))
    push!(d.simplex_violation_history, abs(sum(λ) - 1.0))
    push!(d.neg_lambda_count_history, count(x -> x < -1e-12, λ))

    # --- Pruning diagnostics: newest element is at index end ---
    # The last element in the bundle was added most recently (previous iteration).
    # If QP assigns it near-zero weight, it will be pruned by the λ ≤ atol_λ rule,
    # meaning the newest information is being discarded.
    newest_λ = λ[end]
    would_be_pruned = newest_λ ≤ eps()
    push!(d.newest_lambda_history, newest_λ)
    push!(d.newest_would_be_pruned_history, would_be_pruned)

    # --- Immediate warnings for critical issues ---
    is_critical = false
    if stats.status == :infeasible
        @warn "[$(d.method_label) QP #$(d.call_count)] INFEASIBLE -- QP declared infeasible by RipQP"
        is_critical = true
    end
    if stats.status == :unbounded
        @warn "[$(d.method_label) QP #$(d.call_count)] UNBOUNDED -- QP declared unbounded by RipQP"
        is_critical = true
    end
    if any(isnan, λ)
        @warn "[$(d.method_label) QP #$(d.call_count)] NaN in solution vector! λ = $λ"
        is_critical = true
    end
    if any(isinf, λ)
        @warn "[$(d.method_label) QP #$(d.call_count)] Inf in solution vector! λ = $λ"
        is_critical = true
    end
    if stats.status ∉ (:first_order, :acceptable) && !is_critical
        @warn "[$(d.method_label) QP #$(d.call_count)] Non-optimal: status=$(stats.status), iter=$(stats.iter), pfeas=$(stats.primal_feas), dfeas=$(stats.dual_feas)"
    end

    # --- Per-iteration bundle size and newest element weight (throttled) ---
    if d.call_count % d.print_frequency == 0 || d.call_count == 1
        pruned_marker = would_be_pruned ? " ⚠ NEWEST PRUNED" : ""
        println("  [$(d.method_label) iter $(d.call_count)] bundle_size=$bundle_size, λ[end]=$(round(newest_λ, sigdigits=4))$pruned_marker")
    end
end

function print_qp_diagnostics_summary(d::QPDiagnostics)
    println("\n" * "="^70)
    println("  QP SOLVER DIAGNOSTICS SUMMARY -- $(d.method_label)")
    println("="^70)
    println("Total QP solves: $(d.call_count)")

    isempty(d.status_history) && (println("  No QP calls were logged."); println("="^70 * "\n"); return)

    # --- Status breakdown ---
    status_counts = Dict{Symbol, Int}()
    for s in d.status_history
        status_counts[s] = get(status_counts, s, 0) + 1
    end
    println("\nStatus breakdown:")
    for (status, cnt) in sort(collect(status_counts), by=x->-x[2])
        pct = round(100 * cnt / length(d.status_history), digits=1)
        marker = status ∈ (:first_order, :acceptable) ? "  " : "!!"
        println("  $marker $status: $cnt ($pct%)")
    end

    # --- Feasibility ---
    pf = d.primal_feas_history
    df = d.dual_feas_history
    println("\nFeasibility (primal): max=$(round(maximum(pf), sigdigits=3)), " *
            "mean=$(round(mean(pf), sigdigits=3)), median=$(round(median(pf), sigdigits=3))")
    println("Feasibility (dual):   max=$(round(maximum(df), sigdigits=3)), " *
            "mean=$(round(mean(df), sigdigits=3)), median=$(round(median(df), sigdigits=3))")

    # --- Problem conditioning ---
    valid_conds = filter(!isnan, d.H_cond_history)
    if !isempty(valid_conds)
        println("\nGram matrix condition number: max=$(round(maximum(valid_conds), sigdigits=3)), " *
                "mean=$(round(mean(valid_conds), sigdigits=3)), median=$(round(median(valid_conds), sigdigits=3))")
        ill_cond = count(c -> c > 1e10, valid_conds)
        # ill_cond > 0 && println("  WARNING: $ill_cond / $(length(valid_conds)) QP solves had cond(H) > 1e10")
    end

    # --- Solution health ---
    n_nan = count(d.has_nan_history)
    n_inf = count(d.has_inf_history)
    n_neg = count(x -> x > 0, d.neg_lambda_count_history)
    max_simplex_viol = maximum(d.simplex_violation_history)
    println("\nSolution health:")
    println("  NaN in solution:     $n_nan / $(d.call_count)")
    println("  Inf in solution:     $n_inf / $(d.call_count)")
    println("  Negative lambda:     $n_neg / $(d.call_count) solves had lambda_i < -1e-12")
    println("  Simplex violation:   max |sum(lambda)-1| = $(round(max_simplex_viol, sigdigits=3))")
    println("  Lambda range:        [$(round(minimum(d.lambda_min_history), sigdigits=3)), $(round(maximum(d.lambda_max_history), sigdigits=3))]")

    # --- Bundle sizes ---
    println("\nBundle sizes: min=$(minimum(d.bundle_size_history)), " *
            "max=$(maximum(d.bundle_size_history)), " *
            "mean=$(round(mean(d.bundle_size_history), digits=1))")
    println("Bundle size per iteration:")
    println("  ", d.bundle_size_history)

    # --- Non-optimal detail ---
    non_opt_indices = findall(s -> s ∉ (:first_order, :acceptable), d.status_history)
    if !isempty(non_opt_indices)
        println("\nNon-optimal QP solves ($(length(non_opt_indices)) total):")
        show_n = min(20, length(non_opt_indices))
        for i in non_opt_indices[1:show_n]
            println("  QP #$i: status=$(d.status_history[i]), iter=$(d.iter_history[i]), " *
                    "pfeas=$(round(d.primal_feas_history[i], sigdigits=3)), " *
                    "dfeas=$(round(d.dual_feas_history[i], sigdigits=3)), " *
                    "cond(H)=$(round(d.H_cond_history[i], sigdigits=3)), " *
                    "bundle_size=$(d.bundle_size_history[i])")
        end
        length(non_opt_indices) > show_n && println("  ... and $(length(non_opt_indices) - show_n) more")
    else
        println("\nAll QP solves converged to first-order or acceptable optimality.")
    end

    # --- Pruning diagnostics: newest element survival ---
    if !isempty(d.newest_would_be_pruned_history)
        n_pruned = count(d.newest_would_be_pruned_history)
        n_total = length(d.newest_would_be_pruned_history)
        pct_pruned = round(100 * n_pruned / n_total, digits=1)
        println("\nNewest element pruning (λ[end] ≤ eps → pruned next iter):")
        println("  Pruned: $n_pruned / $n_total ($pct_pruned%)")
        if !isempty(d.newest_lambda_history)
            println("  λ[end] range: [$(round(minimum(d.newest_lambda_history), sigdigits=3)), $(round(maximum(d.newest_lambda_history), sigdigits=3))]")
            println("  λ[end] mean:  $(round(mean(d.newest_lambda_history), sigdigits=3))")
            println("  λ[end] median: $(round(median(d.newest_lambda_history), sigdigits=3))")
        end
        if n_pruned > 0
            # Show first few iterations where newest was pruned
            pruned_iters = findall(d.newest_would_be_pruned_history)
            show_n = min(20, length(pruned_iters))
            println("  First pruned at iterations: $(pruned_iters[1:show_n])" *
                    (length(pruned_iters) > show_n ? " ... and $(length(pruned_iters) - show_n) more" : ""))
            # Check for consecutive pruning (persistent stalling pattern)
            max_consecutive = 0
            current_consecutive = 0
            for pruned in d.newest_would_be_pruned_history
                if pruned
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else
                    current_consecutive = 0
                end
            end
            println("  Max consecutive prunings: $max_consecutive")
            if pct_pruned > 50
                println("  ⚠ WARNING: Newest subgradient is pruned >50% of the time!")
                println("    This means RCBM is discarding new information and may explain stalling.")
            end
        end
    end

    # --- Infeasible / unbounded specifically ---
    n_infeasible = count(s -> s == :infeasible, d.status_history)
    n_unbounded = count(s -> s == :unbounded, d.status_history)
    n_stalled = count(s -> s == :stalled, d.status_history)
    n_max_iter = count(s -> s == :max_iter, d.status_history)
    if n_infeasible + n_unbounded + n_stalled + n_max_iter > 0
        println("\nCritical status counts:")
        n_infeasible > 0 && println("  INFEASIBLE: $n_infeasible")
        n_unbounded > 0  && println("  UNBOUNDED:  $n_unbounded")
        n_stalled > 0    && println("  STALLED:    $n_stalled")
        n_max_iter > 0   && println("  MAX_ITER:   $n_max_iter")
    end

    println("="^70 * "\n")
end

# ============================================================================
# Tangent Space Violation Diagnostics - Track whether transported subgradients
# (pre-QP) and the aggregate direction (post-QP) lie in the tangent space.
# ============================================================================
mutable struct TangentSpaceDiagnostics
    method_label::String
    print_frequency::Int
    call_count::Int
    # Pre-QP: max tangent space violation across input subgradients per call
    pre_qp_max_violation_history::Vector{Float64}
    pre_qp_mean_violation_history::Vector{Float64}
    # Post-QP: tangent space violation of the aggregate direction
    post_qp_violation_history::Vector{Float64}
end

function TangentSpaceDiagnostics(; print_frequency::Int=100)
    TangentSpaceDiagnostics("TS", print_frequency, 0, Float64[], Float64[], Float64[])
end

function reset_ts_diagnostics!(d::TangentSpaceDiagnostics; label::String="TS", print_frequency::Int=d.print_frequency)
    d.method_label = label
    d.print_frequency = print_frequency
    d.call_count = 0
    empty!(d.pre_qp_max_violation_history)
    empty!(d.pre_qp_mean_violation_history)
    empty!(d.post_qp_violation_history)
end

"""
    tangent_violation(M, p, X)

Compute how far X is from lying in T_p M, measured as ‖X - project(M, p, X)‖_p.
"""
function tangent_violation(M, p, X)
    X_proj = project(M, p, X)
    return norm(M, p, X .- X_proj)
end

function log_ts_check!(d::TangentSpaceDiagnostics, M, p, transported_subgradients, λ;
                       scale_factor::Float64=1.0)
    d.call_count += 1

    # --- Pre-QP: check each transported subgradient ---
    violations = [tangent_violation(M, p, X) for X in transported_subgradients]
    max_viol = maximum(violations)
    mean_viol = mean(violations)
    push!(d.pre_qp_max_violation_history, max_viol)
    push!(d.pre_qp_mean_violation_history, mean_viol)

    # --- Post-QP: check aggregate direction ---
    aggregate = scale_factor .* sum(λ[j] .* transported_subgradients[j] for j in eachindex(λ))
    post_viol = tangent_violation(M, p, aggregate)
    push!(d.post_qp_violation_history, post_viol)

    # --- Immediate warnings for large violations ---
    if max_viol > 1e-8
        @warn "[$(d.method_label) TS #$(d.call_count)] Pre-QP tangent violation: max=$(max_viol), mean=$(mean_viol)"
    end
    if post_viol > 1e-8
        @warn "[$(d.method_label) TS #$(d.call_count)] Post-QP aggregate tangent violation: $(post_viol)"
    end
end

function print_ts_diagnostics_summary(d::TangentSpaceDiagnostics)
    println("\n" * "="^70)
    println("  TANGENT SPACE DIAGNOSTICS SUMMARY -- $(d.method_label)")
    println("="^70)
    println("Total QP calls checked: $(d.call_count)")

    isempty(d.pre_qp_max_violation_history) && (println("  No calls were logged."); println("="^70 * "\n"); return)

    # --- Pre-QP summary ---
    pre_max = d.pre_qp_max_violation_history
    pre_mean = d.pre_qp_mean_violation_history
    println("\nPre-QP (transported subgradients entering QP):")
    println("  Max violation per call:  max=$(round(maximum(pre_max), sigdigits=3)), " *
            "mean=$(round(mean(pre_max), sigdigits=3)), median=$(round(median(pre_max), sigdigits=3))")
    println("  Mean violation per call: max=$(round(maximum(pre_mean), sigdigits=3)), " *
            "mean=$(round(mean(pre_mean), sigdigits=3)), median=$(round(median(pre_mean), sigdigits=3))")
    n_pre_bad = count(v -> v > 1e-12, pre_max)
    println("  Calls with max violation > 1e-12: $n_pre_bad / $(d.call_count)")

    # --- Post-QP summary ---
    post = d.post_qp_violation_history
    println("\nPost-QP (aggregate direction after QP weighting):")
    println("  Violation: max=$(round(maximum(post), sigdigits=3)), " *
            "mean=$(round(mean(post), sigdigits=3)), median=$(round(median(post), sigdigits=3))")
    n_post_bad = count(v -> v > 1e-12, post)
    println("  Calls with violation > 1e-12: $n_post_bad / $(d.call_count)")

    println("="^70 * "\n")
end

const TS_DIAG = TangentSpaceDiagnostics(print_frequency=100)

# --- Override Manopt's RCBM subsolver to intercept RipQP ---
function convex_bundle_method_subsolver!(
        M::A, λ, p_last_serious, linearization_errors, transported_subgradients
    ) where {A <: AbstractManifold}
    d = length(linearization_errors)
    H = [inner(M, p_last_serious, X, Y) for X in transported_subgradients, Y in transported_subgradients]
    qm = QuadraticModel(linearization_errors, sparse(tril(H));
        A = reshape(ones(d), 1, d),
        lcon = [one(eltype(linearization_errors))], ucon = [one(eltype(linearization_errors))],
        lvar = zeros(d), uvar = [Inf for i in 1:d], c0 = zero(eltype(linearization_errors)))

    stats = RipQP.ripqp(qm; display = false)
    λ .= stats.solution
    return λ
end

# --- Override Manopt's PBA subsolver to intercept RipQP ---
function proximal_bundle_method_subsolver!(
        M::A, λ, p_last_serious, μ, approximation_errors, transported_subgradients
    ) where {A <: AbstractManifold}
    d = length(approximation_errors)
    H = 1 / μ * [inner(M, p_last_serious, X, Y) for X in transported_subgradients, Y in transported_subgradients]
    qm = QuadraticModel(approximation_errors, sparse(tril(H));
        A = reshape(ones(d), 1, d),
        lcon = [one(eltype(approximation_errors))], ucon = [one(eltype(approximation_errors))],
        lvar = zeros(d), uvar = [Inf for i in 1:d], c0 = zero(eltype(approximation_errors)))

    stats = RipQP.ripqp(qm; display = false)
    λ .= stats.solution
    return λ
end

"""
    inspect_pba_state(st)

Deep inspection of bundle method state to find QP solver location.
Call this once after running to understand the state structure.
"""
function inspect_pba_state(st; depth::Int=0, max_depth::Int=3, prefix::String="")
    depth > max_depth && return

    println("$(prefix)Type: $(typeof(st))")

    try
        for fn in fieldnames(typeof(st))
            fv = getfield(st, fn)
            ft = typeof(fv)
            println("$(prefix)  .$fn :: $ft")

            # Recurse into solver/state/sub fields
            if fv !== nothing && (occursin("solver", lowercase(string(fn))) ||
                                   occursin("state", lowercase(string(fn))) ||
                                   occursin("sub", lowercase(string(fn))))
                inspect_pba_state(fv; depth=depth+1, max_depth=max_depth, prefix=prefix*"    ")
            end
        end
    catch
        println("$(prefix)  (cannot inspect)")
    end
end

# ============================================================================

# --- helper functions for plotting convergence ---
"""
        plot_objective_gap_convergence(records, method_names, true_min_estimate; kwargs...)

    Creates a log-scale convergence plot comparing multiple optimization methods.
    It plots the "Objective Gap" (f(x) - f_min) vs. the iteration count or wall clock time.

    # Arguments
    - `records`: A list of iteration logs. Each log is expected to be a list of
                tuples/arrays where [1] is the iteration index and [2] is the objective value.
    - `method_names`: A list of strings for the plot legend.
    - `true_min_estimate`: A scalar representing the (estimated) global minimum value.

    # Optional keyword arguments
    - `wallclock`: Boolean flag to enable wall clock time plotting (default: false)
    - `wallclock_times`: Array of total wall clock times for each algorithm (required if wallclock=true)
    - `xscale_log`: Boolean flag to enable log scale on x-axis (default: false)
    - `manifold_constraint_mode`: Integer flag for manifold constraint checking:
        1 = ignore manifold constraints (default)
        2 = check algebraic manifold constraints
    - `show_reference`: Boolean flag to show/hide the O(1/√k) reference line (default: true, disabled for wall clock plots)
"""
function plot_objective_gap_convergence(records, method_names, true_min_estimate;
                                   xlabel=nothing,
                                   ylabel=nothing,
                                   filename=nothing,
                                   plot_current=false,
                                   xscale_log=false,
                                   manifold_constraint_mode=1,
                                   manifold=nothing,
                                   manifold_tolerance=1e-12,
                                   show_reference=true,
                                   wallclock = false,
                                   wallclock_times = nothing,
                                   ylims = (0.00001, Inf),
                                   max_x = nothing
                                   )
    # Set default titles and labels based on plot_current flag
    if ylabel === nothing
        ylabel = plot_current ? L"\textrm{Current Objective Gap}" : L"\textrm{Minimum Objective Gap}"
    end
    if xlabel === nothing
        xlabel = wallclock ? L"\textrm{Time (seconds)}" : L"\textrm{Oracle Calls}"
    end

    # 1. Initialize the Plot
    # We use a log10 scale for the Y-axis to clearly see convergence across orders of magnitude.
    # X-axis can optionally also be log scale based on xscale_log parameter.
    p = plot(
        xlabel=xlabel,
        ylabel=ylabel,
        yscale=:log10,
        xscale=xscale_log ? :log10 : :identity,
        ylims=ylims,
        legend=:topright,
        size=(600, 400),
        guidefontsize=12,
        tickfontsize=10,
        legendfontsize=10,
        legendtitlefontsize=10,
        background_color_legend=:transparent,
        foreground_color_legend=:transparent,
        grid=true,
        gridcolor=:lightgray,
        gridwidth=0.5,
        gridalpha=0.3,
        linewidth=1,
        margin=5mm,
        extra_kwargs=Dict(:subplot => Dict("width" => raw"12cm", "height" => raw"8cm")),
    )

    # 2. Define a color palette (IBM/High-Contrast style for accessibility)
    # Map algorithm names to specific colors
    color_map = Dict(
        L"\textrm{RPB}" => "#785ef0",              # Purple
        L"\textrm{RPB (Ours)}" => "#785ef0",        # Purple
        L"\textrm{PBA}" => "#dc267f",              # Pink/Magenta
        L"\textrm{RCBM}" => "#fe6100",             # Orange
        L"\textrm{SGM}" => "#ffb000",              # Gold/Yellow
    )
    # Fallback colors if algorithm not found in map
    fallback_colors = ["#785ef0", "#dc267f", "#fe6100", "#ffb000", :brown]

    # 3. Determine the X-axis limit
    # We find the longest record to ensure the plot window stays consistent.
    max_iterations_across_all = 0
    for record in records
        if !isempty(record)
            max_iterations_across_all = max(max_iterations_across_all, length(record))
        end
    end

    # Note: This multiplier (currently 1.0) can be changed to 0.15 to zoom into 
    # the first 15% of the iterations if the convergence happens very quickly.
    max_iter_idx_global = max(1, Int(floor(1 * max_iterations_across_all)))

    # Helper function to check manifold constraint based on mode
    function check_manifold_constraint_local(M::PowerManifold, iterate, search_direction, mode; atol=1e-12)
        if M === nothing || mode == 1
            return true, ""  # Skip check if no manifold provided or mode 1 (ignore constraints)
        end

        base_H = M.manifold # The Hyperbolic(2) manifold
        n_comp = length(iterate) # Get number of components

        for i in 1:n_comp
            p_i = get_component(M, iterate, i)

            if mode == 2
                # Mode 2: Check algebraic manifold constraints
                # Check Point: <p, p>_m should be -1
                p_val = inner(base_H, p_i, p_i, p_i)
                p_drift = abs(p_val + 1.0)

                if p_drift > atol
                    return false, "point membership (drift: $(p_drift))"
                end

                # Check tangent space membership of search direction if available
                if search_direction !== nothing && length(search_direction) == n_comp
                    v_i = get_component(M, search_direction, i)
                    # Check Tangency: <p, ξ>_m should be 0
                    v_drift = abs(inner(base_H, p_i, p_i, v_i))
                    if v_drift > atol
                        return false, "tangent space membership (drift: $(v_drift))"
                    end
                end
            end
        end

        return true, ""
    end

    # 4. Process and Plot each method
    first_method_data = nothing  # Store first method's data for reference line calculation

    for (i, (record, name)) in enumerate(zip(records, method_names))
        if !isempty(record)
            # Check for manifold violations and track first occurrence of each type
            valid_record = record
            violation_index = nothing
            violation_type = ""
            manifold_violation_index = nothing
            tangent_space_violation_index = nothing
            if manifold_constraint_mode > 1 && manifold !== nothing
                for j in 1:length(record)
                    iterate = length(record[j]) >= 3 ? record[j][3] : nothing
                    search_direction = length(record[j]) >= 4 ? record[j][4] : nothing

                    if iterate !== nothing
                        # Check manifold membership separately from tangent space membership
                        base_H = manifold.manifold
                        n_comp = length(iterate)

                        # Check manifold membership constraint (iterate must be on manifold)
                        if manifold_violation_index === nothing
                            manifold_violation_found = false
                            for k in 1:n_comp
                                p_k = get_component(manifold, iterate, k)
                                p_val = inner(base_H, p_k, p_k, p_k)
                                p_drift = abs(p_val + 1.0)
                                if p_drift > manifold_tolerance
                                    manifold_violation_index = j
                                    manifold_violation_found = true
                                    println("⚠️  First manifold membership constraint violation at iteration $j for $name (drift: $(p_drift))")
                                    break
                                end
                            end
                        end

                        # Check tangent space membership: search direction must be in T_{base_point}
                        # For bundle methods (5th record = p_last_serious), skip this check:
                        # Manopt records state AFTER step_solver!, so p_last_serious is post-update
                        # while g/d was computed at the pre-update p_last_serious. This causes
                        # false violations. QP-level diagnostics (TangentSpaceDiagnostics) check
                        # at the correct moment instead.
                        is_bundle_method = length(record[j]) >= 5
                        if !is_bundle_method && tangent_space_violation_index === nothing && search_direction !== nothing && iterate !== nothing && length(search_direction) == n_comp
                            tangent_violation_found = false
                            for k in 1:n_comp
                                p_k = get_component(manifold, iterate, k)
                                v_k = get_component(manifold, search_direction, k)
                                v_drift = abs(inner(base_H, p_k, p_k, v_k))
                                if v_drift > manifold_tolerance
                                    tangent_space_violation_index = j
                                    tangent_violation_found = true
                                    println("⚠️  First tangent space membership constraint violation at iteration $j for $name (drift: $(v_drift))")
                                    break
                                end
                            end
                        end

                        # Set overall violation index to the first one encountered (for backward compatibility)
                        if violation_index === nothing
                            if manifold_violation_index !== nothing && tangent_space_violation_index !== nothing
                                violation_index = min(manifold_violation_index, tangent_space_violation_index)
                                violation_type = manifold_violation_index <= tangent_space_violation_index ? "point membership" : "tangent space membership"
                            elseif manifold_violation_index !== nothing
                                violation_index = manifold_violation_index
                                violation_type = "point membership"
                            elseif tangent_space_violation_index !== nothing
                                violation_index = tangent_space_violation_index
                                violation_type = "tangent space membership"
                            end
                        end
                    end
                end
            end

            # Extract x-axis values and raw objective values
            if wallclock && wallclock_times !== nothing && i <= length(wallclock_times)
                # For wall clock time: create uniform time grid based on original record length
                total_time = wallclock_times[i]
                original_num_iters = length(record)  # Use original record length, not valid_record
                time_per_iter = original_num_iters > 1 ? total_time / (original_num_iters - 1) : 0.0
                # Create time values for the valid iterations only
                x_values = [(r[1]) * time_per_iter for r in valid_record]  # r[1] is iteration number
                objective_gaps = [max(r[2] - true_min_estimate, 1e-16) for r in valid_record]
            else
                # For iteration-based plotting: use iteration numbers
                x_values = [r[1] + 1 for r in valid_record]
                objective_gaps = [max(r[2] - true_min_estimate, 1e-16) for r in valid_record]
            end

            # 5. Compute objective values to plot based on toggle
            if plot_current
                # Use current objective gaps at each iteration
                gaps_to_plot = objective_gaps
            else
                # Compute Cumulative Minimum (default behavior)
                # Many algorithms (like Subgradient Descent) are not monotonic.
                # This ensures we plot the "Best value found up to iteration k".
                gaps_to_plot = [minimum(objective_gaps[1:j]) for j in 1:length(objective_gaps)]
            end

            # Apply the iteration limit (the 'actual_limit' prevents indexing out of bounds)
            actual_limit = min(max_iter_idx_global, length(x_values))
            x_values_limited = x_values[1:actual_limit]
            gaps_limited = gaps_to_plot[1:actual_limit]

            # 6. Sanitize data for log-scale plotting
            # Filters out any non-positive values that would break the log axis.
            valid_indices = (x_values_limited .> 0) .& (gaps_limited .> 0)
            valid_x_values = x_values_limited[valid_indices]
            valid_gaps = gaps_limited[valid_indices]

            # Store first method's data for reference line calculation
            if i == 1 && !isempty(valid_x_values)
                first_method_data = (valid_x_values, valid_gaps)
            end

            # Downsample to at most 2000 points per line
            n_pts = length(valid_x_values)
            data_stride = n_pts > 2000 ? cld(n_pts, 2000) : 1
            if data_stride > 1
                valid_x_values = valid_x_values[1:data_stride:end]
                valid_gaps = valid_gaps[1:data_stride:end]
            end

            if !isempty(valid_x_values)
                # Get color from map or use fallback
                plot_color = get(color_map, name, fallback_colors[mod1(i, length(fallback_colors))])

                if violation_index === nothing
                    # No violations - plot normally with solid line
                    plot!(p, valid_x_values, valid_gaps,
                        label=name,
                        color=plot_color,
                        linewidth=1.75,
                        markershape=:circle,
                        markersize=0,
                        markerstrokewidth=0)
                else
                    # Violations detected - plot regular line until violation, then muted dotted line
                    # Find the violation point in the valid data
                    violation_data_index = min(cld(violation_index, data_stride), length(valid_x_values))

                    if violation_data_index > 1
                        # Plot regular line up to violation point
                        plot!(p, valid_x_values[1:violation_data_index], valid_gaps[1:violation_data_index],
                            label=name,
                            color=plot_color,
                            linewidth=1.75,
                            markershape=:circle,
                            markersize=0,
                            markerstrokewidth=0)
                    end

                    if violation_data_index <= length(valid_x_values)
                        # Plot muted dotted line from violation point onwards
                        # Create muted version with 50% alpha
                        muted_color = plot_color  # Use original color with default transparency handled by plotting backend

                        plot!(p, valid_x_values[violation_data_index:end], valid_gaps[violation_data_index:end],
                            label="", # No label for the dotted portion
                            color=muted_color,
                            linewidth=1.5,
                            linestyle=:dot,
                            markershape=:circle,
                            markersize=0,
                            markerstrokewidth=0)
                    end
                end
            end
        end
    end

    # 4.1. Add O(1/√k) theoretical reference line (togglable)
    if show_reference && first_method_data !== nothing && !wallclock
        first_x_values, first_gaps = first_method_data

        # Define x-range based on the plotted data range
        x_min = maximum([1, minimum(first_x_values)])
        x_max = maximum(first_x_values)
        x_range = range(x_min, x_max, length=100)

        # Calculate constant C so that reference line starts at same height as first data point
        # y = C * x^(-0.5), so C = y₁ * x₁^(0.5)
        first_gap = first_gaps[1]
        first_x = first_x_values[1]
        C = first_gap * sqrt(first_x)

        # Create reference line: y = C * x^(-0.5)
        reference_line = C .* (x_range .^ (-0.5))

        # Only plot within current axis limits and ensure positive values
        valid_ref_indices = (x_range .>= x_min) .& (x_range .<= x_max) .& (reference_line .> 0)
        x_ref = x_range[valid_ref_indices]
        y_ref = reference_line[valid_ref_indices]

        if !isempty(x_ref)
            plot!(p, x_ref, y_ref,
                label=L"O(1/\sqrt{k})",
                color=:gray,
                linestyle=:dot,
                linewidth=1)
        end
    end

    # 7. Apply x-axis limits if specified
    if max_x !== nothing
        current_xlims = Plots.xlims(p)
        xlims!(p, (current_xlims[1], max_x))
    end

    # 8. Save and Return
    if filename !== nothing
        savefig(p, filename)
        println("Plot saved to: $filename")
    end

    return p
end

# --- generate data for objective ---
H = Hyperbolic(2)
signal, geodesics = artificial_H2_signal(num_points, -6.0, 6.0, 3)
noise = map(p -> exp(H, p, rand(H; vector_at=p, σ=sigma)), signal)
diameter = 3 * maximum([distance(H, noise[i], noise[j]) for i in 1:length(noise), j in 1:length(noise)])
Hn = PowerManifold(H, NestedPowerRepresentation(), length(noise))

# --- polish parallel transport with a projection to ensure numerical stability ---
# Defined on Hyperbolic (base manifold) so that PowerManifold's component-wise
# dispatch calls this for each H^2 factor without ambiguity.
struct ProjectedParallelTransport <: AbstractVectorTransportMethod end

function ManifoldsBase.vector_transport_to!(M::Hyperbolic, Y, p, X, q, ::ProjectedParallelTransport)
    parallel_transport_to!(M, Y, p, X, q)
    project!(M, Y, q, Y)
    return Y
end

# --- Polish the default ExponentialRetraction for Hyperbolic ---
# Override retract! so that exp is always followed by a hyperboloid normalization.
# This ensures ALL solvers (RCBM, PBA, SGM, RPB) get a polished retraction automatically,
# including RCBM's internal DomainBackTrackingStepsize which ignores user-provided
# retraction_method and uses default_retraction_method(M) = ExponentialRetraction().
function ManifoldsBase.retract!(M::Hyperbolic, q, p, X, ::ExponentialRetraction)
    exp!(M, q, p, X)
    # Normalize back onto hyperboloid: enforce ⟨q, q⟩_M = -1
    # where ⟨a,b⟩_M = -a[end]*b[end] + dot(a[1:end-1], b[1:end-1])
    mq = dot(@view(q[1:end-1]), @view(q[1:end-1])) - q[end]^2  # ≈ -1
    q ./= sqrt(-mq)
    return q
end
# --- Plot of ball

export_orig = false  # Set to true if you want to export original data

global ball_scene = plot()

if export_orig
    ball_signal = convert.(PoincareBallPoint, signal)
    ball_noise = convert.(PoincareBallPoint, noise)
    ball_geodesics = convert.(PoincareBallPoint, geodesics)
    plot!(ball_scene, H, ball_signal; geodesic_interpolation=100, label="Geodesics")
    plot!(
        ball_scene,
        H,
        ball_signal;
        markercolor=data_color,
        markerstrokecolor=data_color,
        label="Signal",
    )
    plot!(
        ball_scene,
        H,
        ball_noise;
        markercolor=noise_color,
        markerstrokecolor=noise_color,
        label="Noise",
    )
    matrix_data = matrixify_Poincare_ball(ball_signal)
    matrix_noise = matrixify_Poincare_ball(ball_noise)
    matrix_geodesics = matrixify_Poincare_ball(ball_geodesics)
    CSV.write(
        joinpath(results_folder, experiment_name * "-signal.csv"),
        DataFrame(matrix_data, :auto);
        header=["x", "y"],
    )
    CSV.write(
        joinpath(results_folder, experiment_name * "-noise.csv"),
        DataFrame(matrix_noise, :auto);
        header=["x", "y"],
    )
    CSV.write(
        joinpath(results_folder, experiment_name * "-geodesics.csv"),
        DataFrame(matrix_geodesics, :auto);
        header=["x", "y"],
    )
    display(ball_scene)
end

# --- set up and run of optimization algorithms ---

rcbm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :diameter => diameter,
    :debug => [
        :Iteration,
        (:Cost, "F(p): %1.8f "),
        (:ξ, "ξ: %1.16f "),
        (:ε, "ε: %1.16f "),
        :WarnBundle,
        :Stop,
        1000,
        "\n",
        ],
    :domain => domf,
    :k_max => k_max,
    :k_min => k_min,
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iter),
]
rcbm_bm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :diameter => diameter,
    :domain => domf,
    :k_max => k_max,
    :k_min => k_min,
    :stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iter),
]
pba_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :debug => [
        :Iteration,
        :Stop,
        (:Cost, "F(p): %1.16f "),
        (:ν, "ν: %1.16f "),
        (:c, "c: %1.16f "),
        (:μ, "μ: %1.8f "),
        :Stop,
        1000,
        "\n",
    ],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iter),
]
pba_bm_kwargs = [
    :cache =>(:LRU, [:Cost, :SubGradient], 50),
    :stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iter),
]
sgm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
    :stopping_criterion => StopWhenSubgradientNormLess(√atol) | StopAfterIteration(max_iter),
]
sgm_bm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stopping_criterion => StopWhenSubgradientNormLess(√atol) | StopAfterIteration(max_iter),
]
cppa_kwargs = [
    #=
    :debug => [
        :Iteration,
        " | ",
        DebugProximalParameter(),
        " | ",
        (:Cost, "F(p): %1.16f "),
        " | ",
        :Change,
        "\n",
        1000,
        :Stop,
    ],
    =#
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stopping_criterion => StopWhenAny(StopAfterIteration(max_iter), StopWhenChangeLess(Hn, atol)),
]
cppa_bm_kwargs = [
    :stopping_criterion => StopWhenAny(StopAfterIteration(max_iter), StopWhenChangeLess(Hn, atol)),
]

# --- Phase 1: Find approximation of true minimum ---
println("=== PHASE 1: Finding true minimum ===")
# Print initial objective value
initial_objective = f(Hn, noise)
println("Initial objective value: $initial_objective")
println()

# Create initial entries for plotting
initial_entry = (0, initial_objective, noise)
initial_entry_no_iterate = (0, initial_objective)

# Set up manifold tools for RPB
retraction_map = (p, v) -> retract(Hn, p, v, ExponentialRetraction())
# Standard transport map (for comparison)
transport_map = (p, q, v) -> vector_transport_to(Hn, p, v, q, ParallelTransport())

# Projection-based transport map
transport_map_projection = (p, q, v) -> vector_transport_to(Hn, p, v, q, ProjectionTransport())

# PHASE 1: Find the true minimum by running methods for extended iterations
println("=== PHASE 1: Finding true minimum ===")
phase1_maxiter = max_iter* PHASE1_MULTIPLIER
initial_subgradient = subgradient_of_f(Hn, noise)

# Phase 1 solver configurations with extended iterations for all algorithms
cppa_kwargs_phase1 = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stopping_criterion => StopAfterIteration(phase1_maxiter),
]

cppa_phase1 = cyclic_proximal_point(Hn, f, proxes, noise; cppa_kwargs_phase1...)
cppa_min_obj = minimum([r[2] for r in get_record(cppa_phase1)])
println("    CPPA minimum objective: $cppa_min_obj")

# Use the best result among all Phase 1 algorithms as the true minimum estimate
phase1_results = [cppa_min_obj]
phase1_names = ["CPPA"]
true_min_estimate = minimum(phase1_results)
best_algorithm_idx = argmin(phase1_results)
best_algorithm = phase1_names[best_algorithm_idx]

println("Phase 1 complete. Best result from $best_algorithm: $true_min_estimate")
println("Phase 1 results summary:")
for (name, result) in zip(phase1_names, phase1_results)
    println("  $name: $result")
end

# Free Phase 1 solver state (true_min_estimate is already extracted)
cppa_phase1 = nothing
GC.gc()

# --- PHASE 2: Run main experiments with RPB and other methods ---
println("=== PHASE 2: Convergence comparison with objective gap ===")
initial_subgradient = subgradient_of_f(Hn, noise)

# Create data folder early so we can save results immediately after each run
data_folder = joinpath(@__DIR__, "Denoising TV Hyperbolic Data")
isdir(data_folder) || mkpath(data_folder)

function save_record_csv(data_folder, experiment_name, method_name, record)
    df = DataFrame(iteration = [r[1] for r in record], objective = [r[2] for r in record])
    CSV.write(joinpath(data_folder, "$(experiment_name)_$(method_name)_objectives.csv"), df)
    println("  Saved $(method_name) objectives to CSV")
end

# Configurations for Phase 2
# Phase 2 stopping criterion: stop when objective gap is small enough
gap_stopping_criterion = StopWhenCostLess(true_min_estimate + PHASE2_GAP_TOL)

# Phase 2 solver configurations
rcbm_kwargs_phase2 = [
:cache => (:LRU, [:Cost, :SubGradient], 50),
:debug => [
    :Iteration,
    (:Cost, "F(p): %.10f "),
    :Stop,
    10_000,
    "\n",
],
:diameter => diameter,
:domain => domf,
:k_max => k_max,
:k_min => k_min,
:record => [:Iteration, :Cost],
:return_state => true,
:stopping_criterion => gap_stopping_criterion | StopAfterIteration(max_iter),
:vector_transport_method => ProjectedParallelTransport(),
# :atol_λ => 1e-12,  # Tighten inner QP solve for better convergence in Phase 2
]

sgm_kwargs_phase2 = [
:cache => (:LRU, [:Cost, :SubGradient], 50),
:record => [:Iteration, :Cost, :Iterate],
:return_state => true,
:stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
:stopping_criterion => gap_stopping_criterion | StopAfterIteration(max_iter),
# retraction_method not needed: ExponentialRetraction (default) is now polished with hyperboloid projection
]

pba_kwargs_phase2 = [
:bundle_size => 25,
:cache => (:LRU, [:Cost, :SubGradient], 50),
:debug => [
    :Iteration,
    (:Cost, "F(p): %.10f "),
    # DebugRipQPStatus(frequency=100),  # Print RipQP status every 100 iterations
    :Stop,
    10_000,
    "\n",
],
:record => [:Iteration, :Cost],
:return_state => true,
:stopping_criterion => gap_stopping_criterion | StopAfterIteration(max_iter),
:vector_transport_method => ProjectedParallelTransport(),
# retraction_method not needed: ExponentialRetraction (default) is now polished with hyperboloid projection
]

# Run Phase 2 experiments with detailed timing and QP diagnostics
println("Running RCBM...")
rcbm_start_time = time()
rcbm = convex_bundle_method(Hn, f, subgradient_of_f, noise; rcbm_kwargs_phase2...)
rcbm_end_time = time()
rcbm_total_time = rcbm_end_time - rcbm_start_time
print("RCBM total time: $rcbm_total_time seconds\n")
rcbm_result = get_solver_result(rcbm)
rcbm_record = get_record(rcbm)
rcbm_record = [initial_entry_no_iterate; rcbm_record]
save_record_csv(data_folder, experiment_name, "rcbm", rcbm_record)
rcbm = nothing; rcbm_result = nothing; GC.gc()

println("Running PBA...")
pba_start_time = time()
pba = proximal_bundle_method(Hn, f, subgradient_of_f, noise; pba_kwargs_phase2...)
pba_end_time = time()
pba_total_time = pba_end_time - pba_start_time
print("PBA total time: $pba_total_time seconds\n")
pba_result = get_solver_result(pba)
pba_record = get_record(pba)
pba_record = [initial_entry_no_iterate; pba_record]
save_record_csv(data_folder, experiment_name, "pba", pba_record)
pba = nothing; pba_result = nothing; GC.gc()

println("Running SGM...")
sgm_start_time = time()
sgm = subgradient_method(Hn, f, subgradient_of_f, noise; sgm_kwargs_phase2...)
sgm_end_time = time()

sgm_total_time = sgm_end_time - sgm_start_time
print("SGM total time: $sgm_total_time seconds\n")
sgm_result = get_solver_result(sgm)
sgm_record_raw = get_record(sgm)
# Strip iterates from SGM record to free memory (only iteration + objective needed)
sgm_record = [(r[1], r[2]) for r in sgm_record_raw]
sgm_record = [initial_entry_no_iterate; sgm_record]
save_record_csv(data_folder, experiment_name, "sgm", sgm_record)
sgm = nothing; sgm_result = nothing; sgm_record_raw = nothing; GC.gc()

println("Running RPB...")
rpb_start_time = time()
rpb_solver = RProximalBundle(
    Hn, retraction_map, transport_map_projection,
    (x) -> f(Hn, x), (x) -> subgradient_of_f(Hn, x),
    noise, initial_objective, initial_subgradient;
    max_iter=max_iter, tolerance=PHASE2_GAP_TOL,
    proximal_parameter=1.0/num_points, trust_parameter=0.001, know_minimizer=true, relative_error=false,
    true_min_obj=true_min_estimate
)
run!(rpb_solver)
rpb_end_time = time()

print("RPB total time: $(rpb_end_time - rpb_start_time) seconds\n")

# Convert RPB results to match expected format (iteration, objective, iterate) tuples
rpb_iterations = collect(0:length(rpb_solver.raw_objective_history)-1)
rpb_objectives = rpb_solver.raw_objective_history
rpb_iterates = rpb_solver.proximal_center_history
rpb_record = [(iter, obj, iterate) for (iter, obj, iterate) in zip(rpb_iterations, rpb_objectives, rpb_iterates)]
rpb_result = rpb_solver.current_proximal_center

# Create wall clock time record for RPB with per-iteration average timing
rpb_total_time = rpb_end_time - rpb_start_time

# After running RPB
println("\nRPB Diagnostics:")
println("  Final proximal parameter: $(rpb_solver.proximal_parameter)")
println("  Total iterations: $(length(rpb_solver.iteration) - 1)")
println("  Total descent steps: $(length(rpb_solver.indices_of_descent_steps))")
println("  Total null steps: $(length(rpb_solver.indices_of_null_steps))")

# check if any objective increases occurred at proximal center
if !isempty(rpb_solver.objective_increase_flags)
    num_increases = count(rpb_solver.objective_increase_flags)
    println("  Objective increases at proximal center: $num_increases")
    if num_increases > 0
        increase_percentage = round(100 * num_increases / length(rpb_solver.objective_increase_flags), digits=2)
        println("  Objective increase percentage: $increase_percentage%")
        increase_iterations = findall(rpb_solver.objective_increase_flags)
        println("  Iterations with objective increases: $increase_iterations")
    end
end
save_record_csv(data_folder, experiment_name, "rpb", rpb_record)
rpb_solver = nothing; rpb_result = nothing; GC.gc()

# Print the wall-clock times for all methods
println("\nWall-clock times for all methods:")
println("  RPB total time: $rpb_total_time seconds")
println("  RCBM total time: $rcbm_total_time seconds")
println("  PBA total time: $pba_total_time seconds")
println("  SGM total time: $sgm_total_time seconds")

# --- Save remaining experiment metadata (objective CSVs already saved after each run) ---
metadata = DataFrame(
    method = ["RPB", "RCBM", "PBA", "SGM"],
    wallclock_seconds = [rpb_total_time, rcbm_total_time, pba_total_time, sgm_total_time],
)
CSV.write(joinpath(data_folder, "$(experiment_name)_wallclock.csv"), metadata)

params = DataFrame(
    key = ["true_min_estimate", "num_points", "sigma", "alpha", "max_iter", "k_min", "k_max"],
    value = [true_min_estimate, num_points, sigma, alpha, max_iter, k_min, k_max],
)
CSV.write(joinpath(data_folder, "$(experiment_name)_params.csv"), params)

println("\nExperiment data saved to: $data_folder")

# # --- Create Plots from Experiment ---
# # SGM vs Our Bundle Method Log-Log Plots (reference and no reference line), along with current objective vs minimum objective plots
# records = [rpb_record, sgm_record]
# method_names = [L"\textrm{RPB (Ours)}", L"\textrm{SGM}"]
# plot1 = plot_objective_gap_convergence(
#     records, method_names, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-sgm-vs-rpb-loglog.pdf"),
#     xscale_log=true,

#     show_reference=true,
# )

# plot2 = plot_objective_gap_convergence(
#     records, method_names, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-sgm-vs-rpb-loglog-noref.pdf"),
#     xscale_log=true,

#     show_reference=false,
# )

# plot1_current = plot_objective_gap_convergence(
#     records, method_names, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-sgm-vs-rpb-current-loglog.pdf"),
#     xscale_log=true,
#     plot_current=true,

#     show_reference=false,
# )

# plot2_current = plot_objective_gap_convergence(
#     records, method_names, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-sgm-vs-rpb-current-loglog-noref.pdf"),
#     xscale_log=true,
#     plot_current=true,

#     show_reference=false,
# )

# # All Bundle Methods Wall Clock Time Plot and Log-Log Plots
# records_bundles = [rpb_record, rcbm_record, pba_record]
# method_names_bundles = [L"\textrm{RPB (Ours)}", L"\textrm{RCBM}", L"\textrm{PBA}"]

# plot3 = plot_objective_gap_convergence(
#     records_bundles, method_names_bundles, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-all-bundles-wallclock.pdf"),
#     wallclock=true,
#     wallclock_times=[rpb_total_time, rcbm_total_time, pba_total_time],

#     show_reference=false,
# )

# # Short wallclock: x-axis limited to 5× the fastest finisher
# bundles_fastest_time = minimum([rpb_total_time, rcbm_total_time, pba_total_time])
# plot3_short = plot_objective_gap_convergence(
#     records_bundles, method_names_bundles, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-all-bundles-wallclock-short.pdf"),
#     wallclock=true,
#     wallclock_times=[rpb_total_time, rcbm_total_time, pba_total_time],

#     show_reference=false,
#     max_x=5 * bundles_fastest_time,
# )

# plot4 = plot_objective_gap_convergence(
#     records_bundles, method_names_bundles, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-all-bundles-loglog.pdf"),
#     xscale_log=true,

#     show_reference=false,
# )

# # All Methods Semilogx Plot and Log-Log Plot, and Wall Clock Plot
# records_all = [rpb_record, rcbm_record, pba_record, sgm_record]
# method_names_all = [L"\textrm{RPB (Ours)}", L"\textrm{RCBM}", L"\textrm{PBA}", L"\textrm{SGM}"]

# plot5 = plot_objective_gap_convergence(
#     records_all, method_names_all, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-all-methods-semilogx.pdf"),

#     show_reference=false,
# )

# plot6 = plot_objective_gap_convergence(
#     records_all, method_names_all, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-all-methods-loglog.pdf"),
#     xscale_log=true,

#     show_reference=false,
# )

# plot7 = plot_objective_gap_convergence(
#     records_all, method_names_all, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-all-methods-wallclock.pdf"),
#     wallclock=true,
#     wallclock_times=[rpb_total_time, rcbm_total_time, pba_total_time, sgm_total_time],

#     show_reference=false,
# )

# # Short wallclock: x-axis limited to 5× the fastest finisher
# all_fastest_time = minimum([rpb_total_time, rcbm_total_time, pba_total_time, sgm_total_time])
# plot7_short = plot_objective_gap_convergence(
#     records_all, method_names_all, true_min_estimate;
#     filename=joinpath(results_folder, experiment_name * "-all-methods-wallclock-short.pdf"),
#     wallclock=true,
#     wallclock_times=[rpb_total_time, rcbm_total_time, pba_total_time, sgm_total_time],

#     show_reference=false,
#     max_x=5 * all_fastest_time,
# )

# # --- Bug check for outputs ---
# # Helper function to verify if iterates and directions lie on the manifold and tangent spaces, respectively.
# """
#     verify_geometry(M, iterates, directions; atol=1e-12)

# Verifies if a sequence of points lies on the manifold M and if the corresponding
# search directions lie in the tangent spaces at those points.
# Supports Power Manifolds of Hyperbolic space.
# """
# function verify_geometry(M::PowerManifold, iterates, directions; atol=1e-12)
#     num_iters = min(length(iterates), length(directions))
#     base_H = M.manifold # The Hyperbolic(2) manifold
#     n_comp = length(iterates[1]) # Get actual number of components from first iterate

#     println("--- Geometry Verification Report ---")
#     println("Total Iterations to check: $num_iters")
#     println("Components per Iteration: $n_comp")
#     println("Tolerance: $atol")
#     println("-"^35)

#     p_ensemble_rmse_history = Float64[]
#     v_ensemble_rmse_history = Float64[]

#     violation_found = false

#     for k in 1:num_iters
#         p_full = iterates[k]
#         v_full = directions[k]

#         # Keep track of all errors
#         p_errors = zeros(Float64, n_comp)
#         v_errors = zeros(Float64, n_comp)

#         # Track the worst offending component in this iteration
#         max_p_drift = 0.0
#         max_v_drift = 0.0

#         for i in 1:n_comp
#             p_i = get_component(M, p_full, i)
#             v_i = get_component(M, v_full, i)

#             # 1. Check Point: <p, p>_m should be -1
#             p_val = inner(base_H, p_i, p_i, p_i)
#             p_drift = abs(p_val + 1.0)
#             max_p_drift = max(max_p_drift, p_drift)
#             p_errors[i] = p_drift

#             # 2. Check Tangency: <p, ξ>_m should be 0
#             # Note: inner(M, p, v, w) calculates the Minkowski product for Hyperbolic
#             v_drift = abs(inner(base_H, p_i, p_i, v_i))
#             max_v_drift = max(max_v_drift, v_drift)
#             v_errors[i] = v_drift
#         end

#         # ensemble metric
#         p_ensemble_rmse = sqrt(mean(p_errors .^ 2))
#         v_ensemble_rmse = sqrt(mean(v_errors .^ 2))

#         # report if above iteration 500 and divisible by 50 to reduce log spam
#         if k > 500 && mod(k, 50) == 0
#             # Report if this iteration exceeds tolerance
#             if max_p_drift > atol || max_v_drift > atol || p_ensemble_rmse > atol || v_ensemble_rmse > atol
#                 violation_found = true
#                 @info "Violation at Iteration $(k-1):" max_p_drift max_v_drift p_ensemble_rmse v_ensemble_rmse
#             end
#         end
#         push!(p_ensemble_rmse_history, p_ensemble_rmse)
#         push!(v_ensemble_rmse_history, v_ensemble_rmse)
#     end

#     if !violation_found
#         println("✅ All points and directions are geometrically valid.")
#     else
#         println("❌ Geometric violations detected (see log above).")
#     end
#     return p_ensemble_rmse_history, v_ensemble_rmse_history
# end


# # recover list of iterates (proximal centers)
# pba_iterates = get_record(pba, :Iteration, :Iterate)
# pba_search_directions = get_record(pba, :Iteration, :X)
# rcbm_iterates = get_record(rcbm, :Iteration, :Iterate)
# rcbm_search_directions = get_record(rcbm, :Iteration, :X)

# # check if iterates are actually on the manifold for other bundle methods
# println("\n--- Verifying PBA Geometry ---")
# pba_p_rmse_history, pba_v_rmse_history = verify_geometry(Hn, pba_iterates, pba_search_directions; atol=1e-12)
# println("\n--- Verifying RCBM Geometry ---")
# rcbm_p_rmse_history, rcbm_v_rmse_history = verify_geometry(Hn, rcbm_iterates, rcbm_search_directions; atol=1e-12)

# # --- Run Phase 2 Again for SGM and RPB for much longer since they are fast ---
# println("\n=== PHASE 3: Extended runs for SGM and RPB ===")
# sgm_kwargs_phase3 = [
#     :cache => (:LRU, [:Cost, :SubGradient], 50),
#     :record => [:Iteration, :Cost, :Iterate],
#     :return_state => true,
#     :stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
#     :stopping_criterion => StopWhenCostLess(true_min_estimate + PHASE2_GAP_TOL) | StopAfterIteration(extended_max_iter),
# ]

# rpb_kwargs_phase3 = [
#     :max_iter => extended_max_iter,
#     :tolerance => PHASE2_GAP_TOL,
#     :proximal_parameter => 1.0/num_points,
#     :trust_parameter => 0.001,
#     :know_minimizer => true,
#     :relative_error => false,
#     :true_min_obj => true_min_estimate,
# ]

# # Run extended SGM
# println("Running extended SGM...")
# sgm_extended_start_time = time()
# sgm_extended = subgradient_method(Hn, f, subgradient_of_f, noise;
#     sgm_kwargs_phase3...)
# sgm_extended_end_time = time()
# sgm_extended_total_time = sgm_extended_end_time - sgm_extended_start_time
# println("Extended SGM total time: $sgm_extended_total_time seconds")
# sgm_extended_result = get_solver_result(sgm_extended)
# sgm_extended_record = get_record(sgm_extended)
# # Add initial entry to SGM record to match RPB
# sgm_extended_record = [initial_entry; sgm_extended_record]

# # Run extended RPB
# println("Running extended RPB...")
# rpb_extended_start_time = time()
# rpb_extended_solver = RProximalBundle(
#     Hn, retraction_map, transport_map_projection,
#     (x) -> f(Hn, x), (x) -> subgradient_of_f(Hn, x),
#     noise, initial_objective, initial_subgradient; 
#     max_iter=extended_max_iter, tolerance=PHASE2_GAP_TOL,
#     proximal_parameter=1.0/num_points, trust_parameter=0.001, know_minimizer=true, relative_error=false,
#     true_min_obj=true_min_estimate
# )
# run!(rpb_extended_solver)
# rpb_extended_end_time = time()
# rpb_extended_total_time = rpb_extended_end_time - rpb_extended_start_time
# println("Extended RPB total time: $rpb_extended_total_time seconds")

# # Convert RPB results to match expected format (iteration, objective, iterate) tuples
# rpb_extended_iterations = collect(0:length(rpb_extended_solver.raw_objective_history)-1)
# rpb_extended_objectives = rpb_extended_solver.raw_objective_history
# rpb_extended_iterates = rpb_extended_solver.proximal_center_history
# rpb_extended_record = [(iter, obj, iterate) for (iter, obj, iterate) in zip(rpb_extended_iterations, rpb_extended_objectives, rpb_extended_iterates)]
# rpb_extended_result = rpb_extended_solver.current_proximal_center
# # Add initial entry to RPB record to match SGM
# rpb_extended_record = [initial_entry; rpb_extended_record]
# # Create wall clock time record for extended RPB with per-iteration average timing
# rpb_extended_total_time = rpb_extended_end_time - rpb_extended_start_time

# # --- Create Plots from Extended Experiment ---
# # SGM vs Our Bundle Method Log-Log Plots (reference and no reference line)
# records_extended = [rpb_extended_record, sgm_extended_record]
# method_names_extended = ["RPB (Ours)", "SGM"]
# plot7 = plot_objective_gap_convergence(
#     records_extended, method_names_extended, true_min_estimate;
#     title="Hyperbolic Signal Denoising",
#     filename=joinpath(results_folder, experiment_name * "-extended-sgm-vs-rpb-loglog.png"),
#     xscale_log=true,
#     manifold_constraint_mode=2,
#     manifold=Hn,
#     manifold_tolerance=1e-9,
#     show_reference=true,
# )   
# plot8 = plot_objective_gap_convergence(
#     records_extended, method_names_extended, true_min_estimate;
#     title="Hyperbolic Signal Denoising",
#     filename=joinpath(results_folder, experiment_name * "-extended-sgm-vs-rpb-loglog-noref.png"),
#     xscale_log=true,
#     manifold_constraint_mode=2,
#     manifold=Hn,
#     manifold_tolerance=1e-9,
#     show_reference=false,
# )   

# # print wall-clock times from phase 2 and phase 3
# println("\nWall-clock times for all methods in Phase 2:")
# println("  RPB total time: $rpb_total_time seconds")
# println("  RCBM total time: $rcbm_total_time seconds")
# println("  PBA total time: $pba_total_time seconds")
# println("  SGM total time: $sgm_total_time seconds")

# println("\nWall-clock times for extended methods:")
# println("  Extended RPB total time: $rpb_extended_total_time seconds")
# println("  Extended SGM total time: $sgm_extended_total_time seconds")
