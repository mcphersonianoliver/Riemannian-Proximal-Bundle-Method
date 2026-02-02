using PrettyTables
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots
using QuadraticModels, RipQP
using Random, LinearAlgebra, LRUCache
using Statistics
using ManifoldDiff, Manifolds, Manopt, ManoptExamples

# Include our custom RiemannianProximalBundle solver
include("src/RiemannianProximalBundle.jl")
using .Main: RProximalBundle, run!, plot_objective_versus_iter, plot_model_proximal_gap

# Add inner product function for the RPB solver
inner_product(M, p, v, w) = inner(M, p, v, w)

# --- feature flags (top of file) ---
const benchmarking  = "--bench" in ARGS || get(ENV, "BENCHMARKING", "") == "1"
const export_result = "--export-result" in ARGS || get(ENV, "EXPORT_RESULT", "") == "1"
const export_table  = "--export-table" in ARGS || get(ENV, "EXPORT_TABLE", "") == "1"
const export_orig   = "--export-orig" in ARGS || get(ENV, "EXPORT_ORIG", "") == "1"
const create_gifs   = "--create-gifs" in ARGS || get(ENV, "CREATE_GIFS", "") == "1"

# Experiment configuration - Two-phase mode only
const PHASE1_MULTIPLIER = 2  # Run Phase 1 for this many times the normal iterations
const PHASE2_GAP_TOL = 1e-8  # Stop Phase 2 when objective gap reaches this tolerance
const SAVE_FINAL_ITERATIONS = true  # Save final iterations for transport/retraction testing

println("Configuration:")
println("  Benchmarking: $benchmarking")
println("  Export result: $export_result")
println("  Export tables: $export_table")
println("  Export original: $export_orig")
println("  Create GIFs: $create_gifs")
println("  Phase 1 multiplier: $PHASE1_MULTIPLIER")
println("  Phase 2 gap tolerance: $PHASE2_GAP_TOL")
println("  Save final iterations: $SAVE_FINAL_ITERATIONS")
println()

# --- experiment setup ---
experiment_name = "denoising_h2_tv"
results_folder = joinpath(@__DIR__, "Denoising-H2-TV")
isdir(results_folder) || mkdir(results_folder)

# initialize experiemnt parameter and utility functions
Random.seed!(33)
n = 456 
σ = 0.1 # Noise parameter
α =0.5 # TV parameter
atol = 1e-6  # Relaxed for testing
k_max = 0.0
k_min = -1.0
max_iters = 400  
back_tracking_fact = 2

#
# Colors
data_color = RGBA{Float64}(colorant"#BBBBBB")
noise_color = RGBA{Float64}(colorant"#33BBEE") # Tol Vibrant Teal
result_color = RGBA{Float64}(colorant"#EE7733") # Tol Vibrant Orange

# helper functions
function artificial_H2_signal(pts::Integer=100; a::Real=0.0, b::Real=1.0, T::Real=(b - a) / 2)

    t = range(a, b; length=pts)
    x = [[s, sign(sin(2 * π / T * s))] for s in t]
    y = [
        [x[1]]
        [
            x[i] for
            i in 2:(length(x) - 1) if (x[i][2] != x[i + 1][2] || x[i][2] != x[i - 1][2])
        ]
        [x[end]]
    ]

    y = map(z -> Manifolds._hyperbolize(Hyperbolic(2), z), y)
    data = []
    geodesics = []
    l = Int(round(pts * T / (2 * (b - a))))
    for i in 1:2:(length(y) - 1)
        append!(
            data,
            shortest_geodesic(Hyperbolic(2), y[i], y[i + 1], range(0.0, 1.0; length=l)),
        )
        if i + 2 ≤ length(y) - 1
            append!(
                geodesics,
                shortest_geodesic(Hyperbolic(2), y[i], y[i + 1], range(0.0, 1.0; length=l)),
            )
            append!(
                geodesics,
                shortest_geodesic(
                    Hyperbolic(2), y[i + 1], y[i + 2], range(0.0, 1.0; length=l)
                ),
            )
        end
    end
    return data, geodesics
end

function matrixify_Poincare_ball(input)
    input_x = []
    input_y = []
    for p in input
        push!(input_x, p.value[1])
        push!(input_y, p.value[2])
    end
    return hcat(input_x, input_y)
end

function plot_loglog_convergence(records, method_names;
                            title="Algorithm Convergence Comparison",
                            xlabel="Iteration + 1",
                            ylabel="Objective Value",
                            filename=nothing)
p = plot(
    xlabel=xlabel,
    ylabel=ylabel,
    title=title,
    yscale=:log10,
    legend=:topright, 
    grid=true,
    linewidth=2,
    dpi=300
)

colors = ["#fe6100", "#dc267f", "#785ef0", "#ffb000", "#648fff", :brown]

for (i, (record, name)) in enumerate(zip(records, method_names))
    if !isempty(record)
        iterations = [r[1] + 1 for r in record]
        objectives = [r[2] for r in record]

        # Filter out non-positive values for log scale
        valid_indices = (iterations .> 0) .& (objectives .> 0)
        valid_iterations = iterations[valid_indices]
        valid_objectives = objectives[valid_indices]

        if !isempty(valid_iterations)
            plot!(p, valid_iterations, valid_objectives,
                  label=name,
                  color=colors[mod1(i, length(colors))],
                  linewidth=2,
                  markershape=:circle,
                  markersize=1,
                  markerstrokewidth=0)
        end
    end
end

if filename !== nothing
    savefig(p, filename)
    println("Plot saved to: $filename")
end

return p
end

function plot_objective_gap_convergence(records, method_names, true_min_estimate;
                                   title="Algorithm Convergence Comparison (Minimum Objective Gap)",
                                   xlabel="Iteration + 1",
                                   ylabel="Minimum Objective Gap",
                                   filename=nothing)
p = plot(
    xlabel=xlabel,
    ylabel=ylabel,
    title=title,
    yscale=:log10,
    legend=:topright,
    grid=true,
    linewidth=2,
    dpi=300
)

# Custom colors: RCBM=#fe6100, PBA=#dc267f, RPB=#785ef0, SGM=#ffb000
colors = ["#fe6100", "#dc267f", "#785ef0", "#ffb000", :brown]

# Find the maximum number of iterations across all methods
max_iterations_across_all = 0
for record in records
    if !isempty(record)
        max_iterations_across_all = max(max_iterations_across_all, length(record))
    end
end

# Calculate 15% of the longest iteration amount
# max_iter_idx_global = max(1, Int(ceil(0.15 * max_iterations_across_all)))
max_iter_idx_global = max(1, Int(floor(1 * max_iterations_across_all)))

for (i, (record, name)) in enumerate(zip(records, method_names))
    if !isempty(record)
        iterations = [r[1] + 1 for r in record]
        objective_gaps = [max(r[2] - true_min_estimate, 1e-16) for r in record]  # Ensure positive for log scale

        # Calculate minimum objective gap seen so far (cumulative minimum)
        min_gaps_so_far = [minimum(objective_gaps[1:j]) for j in 1:length(objective_gaps)]

        # Limit to first 15% of the LONGEST iteration amount across all methods
        actual_limit = min(max_iter_idx_global, length(iterations))
        iterations_limited = iterations[1:actual_limit]
        min_gaps_limited = min_gaps_so_far[1:actual_limit]

        # Filter out non-positive values for log scale
        valid_indices = (iterations_limited .> 0) .& (min_gaps_limited .> 0)
        valid_iterations = iterations_limited[valid_indices]
        valid_gaps = min_gaps_limited[valid_indices]

        if !isempty(valid_iterations)
            plot!(p, valid_iterations, valid_gaps,
                  label=name,
                  color=colors[mod1(i, length(colors))],
                  linewidth=2,
                  markershape=:circle,
                  markersize=1,
                  markerstrokewidth=0)
        end
    end
end

if filename !== nothing
    savefig(p, filename)
    println("Plot saved to: $filename")
end

return p
end

function plot_wallclock_objective_gap(time_records, method_names, true_min_estimate;
                                 title="Wall-Clock Time vs Minimum Objective Gap Comparison",
                                 xlabel="Wall-Clock Time (seconds)",
                                 ylabel="Log Minimum Objective Gap",
                                 filename=nothing)
p = plot(
    xlabel=xlabel,
    ylabel=ylabel,
    title=title,
    yscale=:log10,
    legend=:topright,
    grid=true,
    linewidth=2,
    dpi=300
)

# Custom colors: RCBM=#fe6100, PBA=#dc267f, RPB=#785ef0, SGM=#ffb000
colors = ["#fe6100", "#dc267f", "#785ef0", "#ffb000", :brown]

# Find the maximum time duration across all methods
max_time_across_all = 0.0
for time_record in time_records
    if !isempty(time_record)
        times = [r[1] for r in time_record]
        max_time_across_all = max(max_time_across_all, maximum(times))
    end
end

# Calculate 15% of the longest time duration
time_cutoff_global = 0.15 * max_time_across_all

for (i, (time_record, name)) in enumerate(zip(time_records, method_names))
    if !isempty(time_record)
        times = [r[1] for r in time_record]
        objective_gaps = [max(r[2] - true_min_estimate, 1e-16) for r in time_record]  # Ensure positive for log scale

        # Calculate minimum objective gap seen so far (cumulative minimum)
        min_gaps_so_far = [minimum(objective_gaps[1:j]) for j in 1:length(objective_gaps)]

        # Limit to first 15% of the LONGEST time duration across all methods
        time_indices = times .<= time_cutoff_global
        times_limited = times[time_indices]
        min_gaps_limited = min_gaps_so_far[time_indices]

        # Filter out non-positive values for log scale
        valid_indices = (times_limited .>= 0) .& (min_gaps_limited .> 0)
        valid_times = times_limited[valid_indices]
        valid_gaps = min_gaps_limited[valid_indices]

        if !isempty(valid_times)
            plot!(p, valid_times, valid_gaps,
                  label=name,
                  color=colors[mod1(i, length(colors))],
                  linewidth=2,
                  markershape=:circle,
                  markersize=1,
                  markerstrokewidth=0)
        end
    end
end

if filename !== nothing
    savefig(p, filename)
    println("Plot saved to: $filename")
end

return p
end

function plot_error_with_clean_signal(records, method_names, clean_signal, manifold;
                                 title="Error with Clean Signal",
                                 xlabel="Iteration + 1",
                                 ylabel="Distance to Clean Signal",
                                 filename=nothing)
p = plot(
    xlabel=xlabel,
    ylabel=ylabel,
    title=title,
    yscale=:log10,
    legend=:topright,
    grid=true,
    linewidth=2,
    dpi=300
)

# Custom colors: RCBM=#fe6100, PBA=#dc267f, RPB=#785ef0, SGM=#ffb000
colors = ["#fe6100", "#dc267f", "#785ef0", "#ffb000", :brown]

for (i, (record, name)) in enumerate(zip(records, method_names))
    if !isempty(record)
        iterations = []
        error_values = []

        for r in record
            iteration = r[1]
            # Check if the record has iterate information (3rd element)
            if length(r) >= 3
                current_iterate = r[3]
                # Calculate (1/n) * distance(clean_signal, current_iterate)
                error = (1 / length(clean_signal)) * distance(manifold, clean_signal, current_iterate)
                push!(iterations, iteration + 1)  # +1 for log scale
                push!(error_values, error)
            end
        end

        # Filter out non-positive values for log scale
        valid_indices = (iterations .> 0) .& (error_values .> 0)
        valid_iterations = iterations[valid_indices]
        valid_errors = error_values[valid_indices]

        if !isempty(valid_iterations)
            plot!(p, valid_iterations, valid_errors,
                  label=name,
                  color=colors[mod1(i, length(colors))],
                  linewidth=2,
                  markershape=:circle,
                  markersize=1,
                  markerstrokewidth=0)
        end
    end
end

if filename !== nothing
    savefig(p, filename)
    println("Plot saved to: $filename")
end

return p
end

# fix data for objective
H = Hyperbolic(2)
signal, geodesics = artificial_H2_signal(n; a=-6.0, b=6.0, T=3)
noise = map(p -> exp(H, p, rand(H; vector_at=p, σ=σ)), signal)
diameter = 3 * maximum([distance(H, noise[i], noise[j]) for i in 1:length(noise), j in 1:length(noise)])
Hn = PowerManifold(H, NestedPowerRepresentation(), length(noise))

function f(M, p)
return 1 / length(noise) *
       (1 / 2 * distance(M, noise, p)^2 + α * ManoptExamples.Total_Variation(M, p))
end
domf(M, p) = distance(M, p, noise) < diameter / 2 ? true : false
function ∂f(M, p)
return 1 / length(noise) * (
    ManifoldDiff.grad_distance(M, noise, p) +
    α * ManoptExamples.subgrad_Total_Variation(M, p; atol=atol)
)
end
proxes = (
(M, λ, p) -> ManifoldDiff.prox_distance(M, λ, p, 2),
(M, λ, p) -> ManoptExamples.prox_Total_Variation(M, (α * λ), p),
)

global ball_scene = plot()
if export_orig
ball_signal = convert.(PoincareBallPoint, signal)
ball_noise = convert.(PoincareBallPoint, noise)
ball_geodesics = convert.(PoincareBallPoint, geodesics)
plot!(ball_scene, H, ball_signal;
      geodesic_interpolation=100,
      color=data_color,
      linewidth=2,
      label="Clean Signal")
scatter!(ball_scene,
        [pt.value[1] for pt in ball_noise],
        [pt.value[2] for pt in ball_noise];
        markercolor=noise_color,
        markerstrokecolor=noise_color,
        markersize=1,
        label="Noisy Signal")
matrix_data = matrixify_Poincare_ball(ball_signal)
matrix_noise = matrixify_Poincare_ball(ball_noise)
matrix_geodesics = matrixify_Poincare_ball(ball_geodesics)
CSV.write(
    joinpath(results_folder, experiment_name * "-noise.csv"),
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
:stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iters),
]
rcbm_bm_kwargs = [
:cache => (:LRU, [:Cost, :SubGradient], 50),
:diameter => diameter,
:domain => domf,
:k_max => k_max,
:k_min => k_min,
:stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iters),
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
:stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iters),
]
pba_bm_kwargs = [
:cache =>(:LRU, [:Cost, :SubGradient], 50),
:stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iters),
]
sgm_kwargs = [
:cache => (:LRU, [:Cost, :SubGradient], 50),
:debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
:record => [:Iteration, :Cost, :Iterate],
:return_state => true,
:stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
:stopping_criterion => StopWhenSubgradientNormLess(√atol) | StopAfterIteration(max_iters),
]
sgm_bm_kwargs = [
:cache => (:LRU, [:Cost, :SubGradient], 50),
:stopping_criterion => StopWhenSubgradientNormLess(√atol) | StopAfterIteration(max_iters),
]
cppa_kwargs = [
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
:record => [:Iteration, :Cost, :Iterate],
:return_state => true,
:stopping_criterion => StopWhenAny(StopAfterIteration(max_iters), StopWhenChangeLess(Hn, atol)),
]
cppa_bm_kwargs = [
:stopping_criterion => StopWhenAny(StopAfterIteration(max_iters), StopWhenChangeLess(Hn, atol)),
]

# RPB solver configuration
rpb_kwargs = (
true_min_obj = 0.0,
proximal_parameter = 1.0/n,
trust_parameter = 0.001,
max_iter = max_iters,
tolerance = atol,
adaptive_proximal = true,
know_minimizer = false,
relative_error = false,
retraction_error = 1.0,
transport_error = 1.0,
sectional_curvature = -1.0,
back_tracking_factor = back_tracking_fact,
)

# Print initial objective value
initial_objective = f(Hn, noise)
println("Initial objective value: $initial_objective")
println()

# Create initial entry for plotting
initial_entry = (0, initial_objective, noise)

# Set up manifold tools for RPB
retraction_map = (p, v) -> retract(Hn, p, v, ExponentialRetraction())

# Standard transport map (for comparison)
transport_map = (p, q, v) -> vector_transport_to(Hn, p, v, q, ParallelTransport())

# Projection-based transport map
transport_map_projection = (p, q, v) -> vector_transport_to(Hn, p, v, q, ProjectionTransport())

# PHASE 1: Find the true minimum by running methods for extended iterations
println("=== PHASE 1: Finding true minimum ===")
phase1_maxiter = max_iters * PHASE1_MULTIPLIER

# Phase 1 solver configuration with extended iterations (CPPA only)
cppa_kwargs_phase1 = [
:record => [:Iteration, :Cost, :Iterate],
:return_state => true,
:stopping_criterion => StopAfterIteration(phase1_maxiter),
]

# Run Phase 1 to find approximate true minimum using CPPA only
println("Phase 1: CPPA (finding true minimum)")
cppa_phase1 = cyclic_proximal_point(Hn, f, proxes, noise; cppa_kwargs_phase1...)
cppa_min_obj = minimum([r[2] for r in get_record(cppa_phase1)])

# Use CPPA result as the true minimum estimate
true_min_estimate = cppa_min_obj
println("Phase 1 complete. Estimated true minimum: $true_min_estimate")

# PHASE 2: Run methods until gap tolerance is reached
println("=== PHASE 2: Convergence comparison with objective gap ===")

# Phase 2 stopping criterion: stop when objective gap is small enough
gap_stopping_criterion = StopWhenCostLess(true_min_estimate + PHASE2_GAP_TOL)

# Phase 2 solver configurations
rcbm_kwargs_phase2 = [
:cache => (:LRU, [:Cost, :SubGradient], 50),
:diameter => diameter,
:domain => domf,
:k_max => k_max,
:k_min => k_min,
:record => [:Iteration, :Cost, :Iterate],
:return_state => true,
:stopping_criterion => gap_stopping_criterion | StopAfterIteration(max_iters),
]

sgm_kwargs_phase2 = [
:cache => (:LRU, [:Cost, :SubGradient], 50),
:record => [:Iteration, :Cost, :Iterate],
:return_state => true,
:stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
:stopping_criterion => gap_stopping_criterion | StopAfterIteration(max_iters),
]

pba_kwargs_phase2 = [
:cache => (:LRU, [:Cost, :SubGradient], 50),
:record => [:Iteration, :Cost, :Iterate],
:return_state => true,
:stopping_criterion => gap_stopping_criterion | StopAfterIteration(max_iters),
]

# Run Phase 2 methods with timing
println("Phase 2: RCBM")
rcbm_start_time = time()
rcbm = convex_bundle_method(Hn, f, ∂f, noise; rcbm_kwargs_phase2...)
rcbm_result = get_solver_result(rcbm)
rcbm_record_raw = get_record(rcbm)
rcbm_record = vcat([initial_entry], rcbm_record_raw)

# Create time-based record for RCBM
rcbm_times = [0.0]  # Start at time 0
rcbm_time_step = (time() - rcbm_start_time) / length(rcbm_record_raw)
for i in 1:length(rcbm_record_raw)
    push!(rcbm_times, i * rcbm_time_step)
end
rcbm_time_record = [(rcbm_times[i], rcbm_record[i][2]) for i in 1:length(rcbm_record)]

println("Phase 2: SGM")
sgm_start_time = time()
sgm = subgradient_method(Hn, f, ∂f, noise; sgm_kwargs_phase2...)
sgm_result = get_solver_result(sgm)
sgm_record_raw = get_record(sgm)
sgm_record = vcat([initial_entry], sgm_record_raw)

# Create time-based record for SGM
sgm_times = [0.0]  # Start at time 0
sgm_time_step = (time() - sgm_start_time) / length(sgm_record_raw)
for i in 1:length(sgm_record_raw)
    push!(sgm_times, i * sgm_time_step)
end
sgm_time_record = [(sgm_times[i], sgm_record[i][2]) for i in 1:length(sgm_record)]

println("Phase 2: PBA")
pba_start_time = time()
pba = proximal_bundle_method(Hn, f, ∂f, noise; pba_kwargs_phase2...)
pba_result = get_solver_result(pba)
pba_record_raw = get_record(pba)
pba_record = vcat([initial_entry], pba_record_raw)

# Create time-based record for PBA
pba_times = [0.0]  # Start at time 0
pba_time_step = (time() - pba_start_time) / length(pba_record_raw)
for i in 1:length(pba_record_raw)
    push!(pba_times, i * pba_time_step)
end
pba_time_record = [(pba_times[i], pba_record[i][2]) for i in 1:length(pba_record)]

println("Phase 2: RPB Projection Transport")
initial_subgradient = ∂f(Hn, noise)

# RPB with gap tolerance as stopping criterion (using projection transport)
rpb_start_time = time()
rpb_solver = RProximalBundle(
    Hn, retraction_map, transport_map_projection,
    (x) -> f(Hn, x), (x) -> ∂f(Hn, x),
    noise, initial_objective, initial_subgradient;
    max_iter=max_iters, tolerance=PHASE2_GAP_TOL,
    proximal_parameter=1.0/n, trust_parameter=0.001,
    adaptive_proximal=true, know_minimizer=true, relative_error=false,
    true_min_obj=true_min_estimate
)
rpb_solver.debugging = true  # Enable debugging for transport constant tracking and objective increase warnings

# Run the RPB solver
run!(rpb_solver)

# Convert RPB results to match expected format (iteration, objective, iterate) tuples
rpb_iterations = collect(0:length(rpb_solver.raw_objective_history)-1)
rpb_objectives = rpb_solver.raw_objective_history
rpb_iterates = rpb_solver.proximal_center_history
rpb_record = [(iter, obj, iterate) for (iter, obj, iterate) in zip(rpb_iterations, rpb_objectives, rpb_iterates)]
rpb_result = rpb_solver.current_proximal_center

# Create time-based record for RPB
rpb_total_time = time() - rpb_start_time
rpb_time_step = rpb_total_time / (length(rpb_solver.raw_objective_history) - 1)
rpb_time_record = [(i * rpb_time_step, rpb_solver.raw_objective_history[i+1]) for i in 0:length(rpb_solver.raw_objective_history)-1]

println("Phase 2 complete.")

# Save final iterations from bundle methods when they stall out
if SAVE_FINAL_ITERATIONS
    println("\n=== SAVING FINAL ITERATIONS ===")

    # Save final iterations for each bundle method
    saved_iterations = Dict()

    # RCBM final iteration
    if !isempty(rcbm_record)
        rcbm_final = rcbm_record[end][3]  # Get final iterate
        saved_iterations["RCBM"] = rcbm_final
        println("Saved RCBM final iteration (iteration $(rcbm_record[end][1]))")
    end

    # PBA final iteration
    if !isempty(pba_record)
        pba_final = pba_record[end][3]  # Get final iterate
        saved_iterations["PBA"] = pba_final
        println("Saved PBA final iteration (iteration $(pba_record[end][1]))")
    end

    # RPB final iteration
    if !isempty(rpb_record)
        rpb_final = rpb_record[end][3]  # Get final iterate
        saved_iterations["RPB"] = rpb_final
        println("Saved RPB final iteration (iteration $(rpb_record[end][1]))")
    end

    # SGM final iteration
    if !isempty(sgm_record)
        sgm_final = sgm_record[end][3]  # Get final iterate
        saved_iterations["SGM"] = sgm_final
        println("Saved SGM final iteration (iteration $(sgm_record[end][1]))")
    end

    # Add initialization point for comparison
    saved_iterations["INITIALIZATION"] = noise
    println("Saved initialization point (noisy signal)")

    println("Final iterations saved for $(length(saved_iterations)) methods")

    # Test transport and retraction errors around these saved iterations
    println("\n=== TESTING TRANSPORT AND RETRACTION ERRORS ===")

    # Test parameters
    num_test_points = 50
    max_distance = 0.1  # Maximum distance for test vectors

    # Storage for transport/retraction error results
    transport_errors = Dict()
    retraction_errors = Dict()

    # Storage for constant approximations
    retraction_constants = Dict()  # C_R estimates
    transport_constants = Dict()   # C_T estimates

    for (method_name, final_point) in saved_iterations
        println("\nTesting transport/retraction errors around $method_name final iteration...")

        method_transport_errors = Float64[]
        method_retraction_errors = Float64[]

        # For constant estimation
        method_retraction_ratios = Float64[]  # |d_M(exp_x(v), R_x(v))| / ||v||
        method_transport_ratios = Float64[]   # ||transp[v] - parallel_transp[v]|| / (||v|| * transport_distance)

        # Store detailed data for plotting
        method_tangent_norms = Float64[]
        method_transport_distances = Float64[]

        for i in 1:num_test_points
            # Generate random tangent vector at the final point
            random_tangent = rand(Hn; vector_at=final_point)

            # Scale to desired magnitude
            tangent_norm = norm(Hn, final_point, random_tangent)
            if tangent_norm > 0
                scaled_tangent = (max_distance * rand()) * random_tangent / tangent_norm
            else
                continue  # Skip zero tangent vectors
            end

            # Get the actual norm of scaled tangent
            v_norm = norm(Hn, final_point, scaled_tangent)
            if v_norm < 1e-12
                continue  # Skip very small vectors
            end

            # Test point nearby via exponential map
            try
                nearby_point = exp(Hn, final_point, scaled_tangent)
                transport_distance = distance(Hn, final_point, nearby_point)

                # Test transport error: parallel transport vs projection transport
                try
                    # Parallel transport
                    parallel_transported = vector_transport_to(Hn, final_point, scaled_tangent, nearby_point, ParallelTransport())

                    # Projection transport
                    projection_transported = vector_transport_to(Hn, final_point, scaled_tangent, nearby_point, ProjectionTransport())

                    # Calculate transport error
                    transport_error = norm(Hn, nearby_point, parallel_transported - projection_transported)
                    push!(method_transport_errors, transport_error)

                    # Store data for plotting
                    push!(method_tangent_norms, v_norm)
                    push!(method_transport_distances, transport_distance)

                    # Calculate C_T ratio: ||transp[v] - parallel_transp[v]|| / (||v|| * transport_distance)
                    if v_norm > 0 && transport_distance > 0
                        transport_ratio = transport_error / (v_norm * transport_distance)
                        push!(method_transport_ratios, transport_ratio)
                    end
                catch e
                    # Skip if transport fails
                    continue
                end

                # Test retraction error: exponential vs first-order retraction approximation
                try
                    # Exponential retraction (exact)
                    exp_retracted = retract(Hn, final_point, scaled_tangent, ExponentialRetraction())

                    # First-order approximation: R_x(v) ≈ x + v (project to manifold)
                    # For PowerManifold, we can use a simpler retraction
                    first_order_retracted = try
                        retract(Hn, final_point, scaled_tangent, ProjectionRetraction())
                    catch
                        # Fallback to exponential if projection retraction not available
                        exp_retracted
                    end

                    # Calculate retraction error d_M(exp_x(v), R_x(v))
                    retraction_error = distance(Hn, exp_retracted, first_order_retracted)
                    push!(method_retraction_errors, retraction_error)

                    # Calculate C_R ratio: |d_M(exp_x(v), R_x(v))| / ||v||
                    if v_norm > 0
                        retraction_ratio = retraction_error / v_norm
                        push!(method_retraction_ratios, retraction_ratio)
                    end
                catch e
                    # Skip if retraction fails
                    continue
                end

            catch e
                # Skip if exponential map fails
                continue
            end
        end

        # Store results including detailed data for plotting
        transport_errors[method_name] = (
            errors = method_transport_errors,
            tangent_norms = method_tangent_norms,
            transport_distances = method_transport_distances,
            ratios = method_transport_ratios
        )
        retraction_errors[method_name] = (
            errors = method_retraction_errors,
            ratios = method_retraction_ratios
        )

        # Calculate constant estimates
        if !isempty(method_retraction_ratios)
            C_R_estimate = maximum(method_retraction_ratios)
            retraction_constants[method_name] = C_R_estimate
        end

        if !isempty(method_transport_ratios)
            C_T_estimate = maximum(method_transport_ratios)
            transport_constants[method_name] = C_T_estimate
        end

        # Print statistics
        if !isempty(method_transport_errors)
            println("  $method_name transport errors ($(length(method_transport_errors)) tests):")
            println("    Mean: $(round(mean(method_transport_errors), digits=8))")
            println("    Max:  $(round(maximum(method_transport_errors), digits=8))")
            println("    Min:  $(round(minimum(method_transport_errors), digits=8))")
            if !isempty(method_transport_ratios)
                println("    C_T estimate: $(round(transport_constants[method_name], digits=6))")
            end
        else
            println("  $method_name: No valid transport error tests")
        end

        if !isempty(method_retraction_errors)
            println("  $method_name retraction errors ($(length(method_retraction_errors)) tests):")
            println("    Mean: $(round(mean(method_retraction_errors), digits=8))")
            println("    Max:  $(round(maximum(method_retraction_errors), digits=8))")
            println("    Min:  $(round(minimum(method_retraction_errors), digits=8))")
            if !isempty(method_retraction_ratios)
                println("    C_R estimate: $(round(retraction_constants[method_name], digits=6))")
            end
        else
            println("  $method_name: No valid retraction error tests")
        end
    end

    # Create plots for transport and retraction errors
    println("\nCreating transport and retraction error plots...")

    # Plot 1: Transport errors vs tangent vector norm
    transport_plot = plot(
        xlabel="Tangent Vector Norm ||v||",
        ylabel="Transport Error ||T_parallel[v] - T_projection[v]||",
        title="Transport Errors Around Final Bundle Method Iterations",
        legend=:topright,
        grid=true,
        dpi=300
    )

    colors = ["#fe6100", "#dc267f", "#785ef0", "#ffb000", "#648fff", :brown]
    for (i, (method_name, data)) in enumerate(transport_errors)
        if !isempty(data.errors)
            scatter!(transport_plot, data.tangent_norms, data.errors,
                    label=method_name,
                    color=colors[mod1(i, length(colors))],
                    alpha=0.7,
                    markersize=3)
        end
    end

    transport_plot_filename = joinpath(results_folder, "$(experiment_name)_transport_errors.png")
    savefig(transport_plot, transport_plot_filename)
    println("Transport error plot saved: $transport_plot_filename")

    # Plot 2: Retraction errors vs tangent vector norm
    retraction_plot = plot(
        xlabel="Tangent Vector Norm ||v||",
        ylabel="Retraction Error d_M(exp_x(v), R_x(v))",
        title="Retraction Errors Around Final Bundle Method Iterations",
        legend=:topright,
        grid=true,
        dpi=300
    )

    for (i, (method_name, data)) in enumerate(retraction_errors)
        if !isempty(data.errors)
            # Need to get tangent norms from transport_errors since they're stored together
            if haskey(transport_errors, method_name) && !isempty(transport_errors[method_name].tangent_norms)
                tangent_norms = transport_errors[method_name].tangent_norms[1:length(data.errors)]
                scatter!(retraction_plot, tangent_norms, data.errors,
                        label=method_name,
                        color=colors[mod1(i, length(colors))],
                        alpha=0.7,
                        markersize=3)
            end
        end
    end

    retraction_plot_filename = joinpath(results_folder, "$(experiment_name)_retraction_errors.png")
    savefig(retraction_plot, retraction_plot_filename)
    println("Retraction error plot saved: $retraction_plot_filename")

    # Plot 3: Constant estimates visualization
    constants_plot = plot(layout=(1, 2), size=(1000, 400), dpi=300)

    # C_R constants
    if !isempty(retraction_constants)
        method_names_cr = collect(keys(retraction_constants))
        cr_values = [retraction_constants[name] for name in method_names_cr]
        bar!(constants_plot[1], method_names_cr, cr_values,
             title="Retraction Constant C_R Estimates",
             ylabel="C_R",
             color=colors[1:length(method_names_cr)],
             legend=false)
    end

    # C_T constants
    if !isempty(transport_constants)
        method_names_ct = collect(keys(transport_constants))
        ct_values = [transport_constants[name] for name in method_names_ct]
        bar!(constants_plot[2], method_names_ct, ct_values,
             title="Transport Constant C_T Estimates",
             ylabel="C_T",
             color=colors[1:length(method_names_ct)],
             legend=false)
    end

    constants_plot_filename = joinpath(results_folder, "$(experiment_name)_constants_estimates.png")
    savefig(constants_plot, constants_plot_filename)
    println("Constants estimates plot saved: $constants_plot_filename")

    # Plot 4: Normalized ratios to show constant bounds
    ratios_plot = plot(layout=(1, 2), size=(1000, 400), dpi=300)

    # C_R ratios
    subplot1 = plot(
        xlabel="Test Point Index",
        ylabel="||d_M(exp_x(v), R_x(v))|| / ||v||",
        title="Retraction Error Ratios (C_R bounds)",
        legend=:topright,
        grid=true
    )

    for (i, (method_name, data)) in enumerate(retraction_errors)
        if !isempty(data.ratios)
            plot!(subplot1, 1:length(data.ratios), data.ratios,
                  label=method_name,
                  color=colors[mod1(i, length(colors))],
                  linewidth=2,
                  alpha=0.8)
        end
    end

    # C_T ratios
    subplot2 = plot(
        xlabel="Test Point Index",
        ylabel="||T_diff|| / (||v|| * d)",
        title="Transport Error Ratios (C_T bounds)",
        legend=:topright,
        grid=true
    )

    for (i, (method_name, data)) in enumerate(transport_errors)
        if !isempty(data.ratios)
            plot!(subplot2, 1:length(data.ratios), data.ratios,
                  label=method_name,
                  color=colors[mod1(i, length(colors))],
                  linewidth=2,
                  alpha=0.8)
        end
    end

    ratios_combined_plot = plot(subplot1, subplot2, layout=(1, 2), size=(1000, 400), dpi=300)
    ratios_plot_filename = joinpath(results_folder, "$(experiment_name)_error_ratios.png")
    savefig(ratios_combined_plot, ratios_plot_filename)
    println("Error ratios plot saved: $ratios_plot_filename")

    # Save transport/retraction error data to files
    if export_table
        println("\nSaving transport/retraction error data...")

        for (method_name, data) in transport_errors
            if !isempty(data.errors)
                df = DataFrame(
                    transport_error = data.errors,
                    tangent_norm = data.tangent_norms[1:length(data.errors)],
                    transport_distance = data.transport_distances[1:length(data.errors)],
                    normalized_ratio = data.ratios[1:length(data.errors)]
                )
                CSV.write(
                    joinpath(results_folder, "$(experiment_name)_$(method_name)_transport_errors.csv"),
                    df
                )
            end
        end

        for (method_name, data) in retraction_errors
            if !isempty(data.errors)
                df = DataFrame(
                    retraction_error = data.errors,
                    normalized_ratio = data.ratios[1:length(data.errors)]
                )
                CSV.write(
                    joinpath(results_folder, "$(experiment_name)_$(method_name)_retraction_errors.csv"),
                    df
                )
            end
        end

        # Save constant estimates
        if !isempty(retraction_constants) || !isempty(transport_constants)
            constants_df = DataFrame(
                method = String[],
                C_R_estimate = Float64[],
                C_T_estimate = Float64[]
            )

            all_methods = unique([collect(keys(retraction_constants)); collect(keys(transport_constants))])
            for method in all_methods
                c_r = get(retraction_constants, method, NaN)
                c_t = get(transport_constants, method, NaN)
                push!(constants_df, (method, c_r, c_t))
            end

            CSV.write(
                joinpath(results_folder, "$(experiment_name)_constants_estimates.csv"),
                constants_df
            )
        end

        println("Transport/retraction error data saved to CSV files")
    end

    # Print summary of constant estimates
    println("\n=== CONSTANT ESTIMATES SUMMARY ===")
    println("Retraction constants C_R (d_M(exp_x(v), R_x(v))/||v|| ≤ C_R):")
    for (method, c_r) in retraction_constants
        println("  $method: C_R ≤ $(round(c_r, digits=6))")
    end

    println("\nTransport constants C_T (||T_diff||/(||v||*d) ≤ C_T):")
    for (method, c_t) in transport_constants
        println("  $method: C_T ≤ $(round(c_t, digits=6))")
    end

    println("\nTransport and retraction error testing completed!")
end

records = [rcbm_record, pba_record, rpb_record, sgm_record]
time_records = [rcbm_time_record, pba_time_record, rpb_time_record, sgm_time_record]

# After running RPB
println("\nRPB Diagnostics:")
println("  Final proximal parameter: $(rpb_solver.proximal_parameter)")
println("  Total iterations: $(length(rpb_solver.iteration) - 1)")
println("  Total descent steps: $(length(rpb_solver.indices_of_descent_steps))")
println("  Total null steps: $(length(rpb_solver.indices_of_null_steps))")
println("  Total proximal parameter doubling steps: $(length(rpb_solver.indices_of_proximal_doubling_steps))")

# Check for objective increases at proximal centers
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

# Plot RPB step types
println("\nCreating RPB step types plot...")
step_plot_filename = joinpath(results_folder, "$(experiment_name)_rpb_step_types.png")

# Create and save the plot using the built-in function with semi-log scale
plot_objective_versus_iter(rpb_solver; save_path=step_plot_filename)

# Plot RPB model proximal gap
println("\nCreating RPB model proximal gap plot...")
model_gap_plot_filename = joinpath(results_folder, "$(experiment_name)_rpb_model_proximal_gap.png")

# Create and save the model proximal gap plot (semi-log scale, not loglog)
plot_model_proximal_gap(rpb_solver; save_path=model_gap_plot_filename, use_loglog=false)

# Plot transport constant estimates
println("\nCreating RPB transport constant plot...")
transport_plot_filename = joinpath(results_folder, "$(experiment_name)_rpb_transport_constants.png")

# Create and save the transport constant plot
plot_transport_constant_versus_iter(rpb_solver; save_path=transport_plot_filename)


method_names_plot = ["RCBM", "PBA", "RPB (Projection)", "SGM"]

plot_filename_gap = joinpath(results_folder, "$(experiment_name)_convergence_gap.png")
plot_objective_gap_convergence(records, method_names_plot, true_min_estimate; filename=plot_filename_gap)

# Create wall-clock time vs objective gap plot
plot_filename_wallclock = joinpath(results_folder, "$(experiment_name)_wallclock_gap.png")
plot_wallclock_objective_gap(time_records, method_names_plot, true_min_estimate; filename=plot_filename_wallclock)


# Function to create denoising progress gif
function create_denoising_gif(records, method_names, results_folder, experiment_name;
                        period=5, duration=500, filename=nothing)
"""
Create a GIF showing the denoising progress for multiple algorithms.

Args:
    records: Vector of algorithm records (each containing (iteration, objective, iterate))
    method_names: Vector of method names corresponding to records
    results_folder: Folder to save the gif
    experiment_name: Base name for the experiment
    period: Show every period-th frame (default: 5)
    duration: Duration per frame in milliseconds (default: 500)
    filename: Custom filename (optional)
"""

# Set default filename if not provided
if filename === nothing
    filename = joinpath(results_folder, "$(experiment_name)_denoising_progress.gif")
end

println("Creating denoising progress GIF...")

# Find the record with the most iterations to use as reference
max_iterations = maximum(length(record) for record in records)
reference_idx = findfirst(record -> length(record) == max_iterations, records)
reference_record = records[reference_idx]
reference_name = method_names[reference_idx]

# Extract iterations that are multiples of period
frame_indices = [i for i in 1:length(reference_record) if (i-1) % period == 0 || i == length(reference_record)]

println("  Creating $(length(frame_indices)) frames (every $period iterations)")

# Create frames
frames = []

# Colors for different methods
method_colors = ["#fe6100", "#dc267f", "#785ef0", "#ffb000", "#648fff", :brown]

for frame_idx in frame_indices
    # Create figure for this frame
    p = plot(size=(1200, 800), layout=(2, 2))

    iteration = reference_record[frame_idx][1]

    # Plot 1: Current denoised signals comparison
    # Plot original signal and noise as reference
    ball_signal = convert.(PoincareBallPoint, signal)
    ball_noise = convert.(PoincareBallPoint, noise)

    plot!(p, H, ball_signal,
          color=:gray, linewidth=2, alpha=0.7,
          label="Clean Signal", subplot=1)
    scatter!(p, [pt.value[1] for pt in ball_noise], [pt.value[2] for pt in ball_noise],
            color=noise_color, markersize=1, alpha=0.5,
            label="Noisy Signal", subplot=1)

    # Plot current denoised results for each method
    for (i, (record, name)) in enumerate(zip(records, method_names))
        if frame_idx <= length(record)
            # Handle different record formats
            if length(record[frame_idx]) >= 3
                current_result = record[frame_idx][3]  # Extract iterate
                ball_result = convert.(PoincareBallPoint, current_result)
                plot!(p, H, ball_result,
                      color=method_colors[mod1(i, length(method_colors))],
                      linewidth=3, alpha=0.8,
                      label="$name (iter $iteration)", subplot=1)
            else
                # Skip methods without iterate information (like RPB in current format)
                continue
            end
        end
    end

    title!(p, "Denoised Signals at Iteration $iteration", subplot=1)

    # Plot 2: Convergence curves up to current iteration
    for (i, (record, name)) in enumerate(zip(records, method_names))
        current_frame = min(frame_idx, length(record))
        if current_frame > 1
            iters = [r[1] + 1 for r in record[1:current_frame]]
            objs = [r[2] for r in record[1:current_frame]]

            # Filter positive values for log scale
            valid_mask = (iters .> 0) .& (objs .> 0)
            if any(valid_mask)
                plot!(p, iters[valid_mask], objs[valid_mask],
                      color=method_colors[mod1(i, length(method_colors))],
                      linewidth=2, marker=:circle, markersize=1,
                      markerstrokewidth=0,
                      label=name, subplot=2)
            end
        end
    end

    xlabel!(p, "Iteration + 1", subplot=2)
    ylabel!(p, "Objective Value", subplot=2)
    title!(p, "Convergence Progress", subplot=2)
    plot!(p, yscale=:log10, subplot=2)

    # Plot 3: Focus on the current best method's result
    if frame_idx <= length(reference_record)
        current_result = reference_record[frame_idx][3]
        ball_result = convert.(PoincareBallPoint, current_result)
        ball_signal = convert.(PoincareBallPoint, signal)

        plot!(p, H, ball_signal,
              color=:gray, linewidth=2, alpha=0.7,
              label="Target", subplot=3)
        plot!(p, H, ball_result,
              color=result_color, linewidth=3, alpha=0.9,
              label="$reference_name Result", subplot=3)
    end

    title!(p, "$reference_name - Iteration $iteration", subplot=3)

    # Plot 4: Statistics table (text-based)
    # Clear the subplot and add text
    plot!(p, [], [], subplot=4)
    xlims!(p, (0, 1), subplot=4)
    ylims!(p, (0, 1), subplot=4)

    # Add statistics text
    stats_text = "Progress Summary\n\n"
    for (i, (record, name)) in enumerate(zip(records, method_names))
        current_frame = min(frame_idx, length(record))
        if current_frame > 0
            current_obj = record[current_frame][2]
            stats_text *= "$name:\n"
            stats_text *= "  Iteration: $(record[current_frame][1])\n"
            stats_text *= "  Objective: $(round(current_obj, digits=6))\n\n"
        end
    end

    annotate!(p, 0.05, 0.95, text(stats_text, :left, :top, 10), subplot=4)
    title!(p, "Algorithm Statistics", subplot=4)
    plot!(p, axis=false, grid=false, subplot=4)

    # Save this frame as a temporary plot
    temp_filename = joinpath(results_folder, "temp_frame_$(lpad(frame_idx, 3, '0')).png")
    savefig(p, temp_filename)
    push!(frames, temp_filename)

    if length(frames) % 5 == 0
        println("    Created frame $(length(frames))/$(length(frame_indices))")
    end
end

# Try using Plots.jl animation instead of ImageMagick
println("  Assembling GIF using Plots.jl animation...")

try
    # Create animation using @animate macro
    gif_filename_anim = replace(filename, ".gif" => "_animation.gif")

    anim = @animate for frame_idx in frame_indices
        # Create figure for this frame (simplified single plot version)
        p = plot(size=(800, 600))

        iteration = reference_record[frame_idx][1]

        # Plot original signal and noise as reference
        ball_signal = convert.(PoincareBallPoint, signal)
        ball_noise = convert.(PoincareBallPoint, noise)

        plot!(p, H, ball_signal,
              color=:gray, linewidth=2, alpha=0.7,
              label="Clean Signal")
        scatter!(p, [pt.value[1] for pt in ball_noise], [pt.value[2] for pt in ball_noise],
                color=noise_color, markersize=1, alpha=0.5,
                label="Noisy Signal")

        # Plot current denoised results for each method (focus on best ones)
        best_methods = [1, 2, 3, 4, 5]  # Methods with iterate information (skip RPB index 5)
        for i in best_methods
            if i <= length(records) && frame_idx <= length(records[i])
                record = records[i]
                name = method_names[i]
                # Check if record has iterate information
                if length(record[frame_idx]) >= 3
                    current_result = record[frame_idx][3]  # Extract iterate
                    ball_result = convert.(PoincareBallPoint, current_result)
                    plot!(p, H, ball_result,
                          color=method_colors[mod1(i, length(method_colors))],
                          linewidth=3, alpha=0.8,
                          label="$name")
                end
            end
        end

        title!(p, "Denoising Progress - Iteration $iteration")
        p
    end

    # Save the GIF with doubled frame rate
    gif(anim, gif_filename_anim, fps=6)  # 6 fps for faster playback
    println("  GIF created successfully using Plots.jl: $gif_filename_anim")

    # Clean up temporary PNG files
    for frame_file in frames
        rm(frame_file, force=true)
    end
    println("  Temporary files cleaned up")

    return gif_filename_anim

catch e
    println("  Error creating GIF with Plots.jl animation: $e")
    println("  Individual frames saved as PNG files:")
    for (i, frame_file) in enumerate(frames)
        println("    Frame $i: $frame_file")
    end
    println("  You can create a GIF manually using ImageMagick:")
    println("    brew install imagemagick")
    frames_string = join(frames, " ")
    println("    convert -delay $(duration÷10) -loop 0 $frames_string $filename")
    return frames
end
end

# Function to create individual method GIFs
function create_individual_method_gifs(records, method_names, results_folder, experiment_name;
                                 period=5, duration=500)
"""
Create separate GIF files for each optimization method.

Args:
    records: Vector of algorithm records (each containing (iteration, objective, iterate))
    method_names: Vector of method names corresponding to records
    results_folder: Folder to save the gifs
    experiment_name: Base name for the experiment
    period: Show every period-th frame (default: 5)
    duration: Duration per frame in milliseconds (default: 500)
"""

println("Creating individual method GIFs...")
created_gifs = []

# Colors for clean signal, noise, and method result
clean_color = :gray
noise_color = RGBA{Float64}(colorant"#33BBEE")
method_colors = ["#fe6100", "#dc267f", "#785ef0", "#ffb000", "#648fff", :brown]

for (method_idx, (record, method_name)) in enumerate(zip(records, method_names))
    # Skip methods without iterate information
    if isempty(record) || length(record[1]) < 3
        println("  Skipping $method_name (no iterate information)")
        continue
    end

    println("  Creating GIF for $method_name...")

    # Extract iterations that are multiples of period
    frame_indices = [i for i in 1:length(record) if (i-1) % period == 0 || i == length(record)]

    try
        # Create animation for this method
        gif_filename = joinpath(results_folder, "$(experiment_name)_$(method_name)_progress.gif")

        anim = @animate for frame_idx in frame_indices
            # Create figure for this frame
            p = plot(size=(1000, 600), layout=(1, 2))

            iteration = record[frame_idx][1]
            current_objective = record[frame_idx][2]
            current_result = record[frame_idx][3]

            # Plot 1: Signal visualization
            ball_signal = convert.(PoincareBallPoint, signal)
            ball_noise = convert.(PoincareBallPoint, noise)
            ball_result = convert.(PoincareBallPoint, current_result)

            plot!(p, H, ball_signal,
                  color=clean_color, linewidth=2, alpha=0.7,
                  label="Clean Signal", subplot=1)
            scatter!(p, [pt.value[1] for pt in ball_noise], [pt.value[2] for pt in ball_noise],
                    color=noise_color, markersize=1, alpha=0.5,
                    label="Noisy Signal", subplot=1)
            plot!(p, H, ball_result,
                  color=method_colors[mod1(method_idx, length(method_colors))],
                  linewidth=3, alpha=0.9,
                  label="$method_name Result", subplot=1)

            title!(p, "$method_name - Iteration $iteration\nObjective: $(round(current_objective, digits=6))", subplot=1)

            # Plot 2: Convergence curve up to current iteration
            if frame_idx > 1
                iters = [r[1] + 1 for r in record[1:frame_idx]]
                objs = [r[2] for r in record[1:frame_idx]]

                # Filter positive values for log scale
                valid_mask = (iters .> 0) .& (objs .> 0)
                if any(valid_mask)
                    plot!(p, iters[valid_mask], objs[valid_mask],
                          color=method_colors[mod1(method_idx, length(method_colors))],
                          linewidth=2, marker=:circle, markersize=3,
                          markerstrokewidth=0,
                          label="$method_name", subplot=2)
                end
            end

            xlabel!(p, "Iteration + 1", subplot=2)
            ylabel!(p, "Objective Value", subplot=2)
            title!(p, "$method_name Convergence", subplot=2)
            plot!(p, yscale=:log10, subplot=2)

            p
        end

        # Save the GIF with doubled frame rate
        gif(anim, gif_filename, fps=4)  # 4 fps for faster playback
        println("    GIF created: $gif_filename")
        push!(created_gifs, gif_filename)

    catch e
        println("    Error creating GIF for $method_name: $e")
    end
end

println("Individual method GIFs creation completed!")
println("Created $(length(created_gifs)) GIF files:")
for gif_file in created_gifs
    println("  - $gif_file")
end

return created_gifs
end

# Create log-log convergence plot
println("\nCreating convergence plot...")
plot_filename = joinpath(results_folder, "$(experiment_name)_convergence_loglog.png")

# Use appropriate method names
final_method_names = ["RCBM", "PBA", "RPB (Projection)", "SGM"]

convergence_plot = plot_loglog_convergence(records, final_method_names;
                                     title="Denoising H2-TV Algorithm Convergence",
                                     filename=plot_filename)
display(convergence_plot)

# Create error with clean signal plot
println("\nCreating error with clean signal plot...")
error_plot_filename = joinpath(results_folder, "$(experiment_name)_error_with_clean_signal.png")
error_plot = plot_error_with_clean_signal(records, final_method_names, signal, Hn;
                                     title="Error with Clean Signal",
                                     filename=error_plot_filename)
display(error_plot)

# Skip subgradient plotting for now - focus on objective gap and convergence plots

if create_gifs
# Create denoising progress GIF
println("\nCreating denoising progress GIF...")
gif_filename = create_denoising_gif(
    records, final_method_names, results_folder, experiment_name;
    period=5, duration=500
)
println("GIF creation completed: $gif_filename")

# Create individual method GIFs
println("\nCreating individual method GIFs...")
individual_gifs = create_individual_method_gifs(
    records, final_method_names, results_folder, experiment_name;
    period=5, duration=500
)
println("Individual GIF creation completed!")
else
println("\nSkipping GIF creation (create_gifs = false)")
end

if benchmarking
# pba_bm = @benchmark proximal_bundle_method($Hn, $f, $∂f, $noise; $pba_bm_kwargs...)
rcbm_bm = @benchmark convex_bundle_method($Hn, $f, $∂f, $noise; $rcbm_bm_kwargs...)
# sgm_bm = @benchmark subgradient_method($Hn, $f, $∂f, $noise; $sgm_bm_kwargs...)
# cppa_bm = @benchmark cyclic_proximal_point($Hn, $f, $proxes, $noise; $cppa_bm_kwargs...)


# Benchmark RPB solver
rpb_bm = @benchmark begin
    retract_func = (p, v) -> retract($Hn, p, v, ExponentialRetraction())
    transport_func = (p, q, v) -> vector_transport_to($Hn, p, v, q, ProjectionTransport())
    init_subgrad = ∂f($Hn, $noise)
    init_obj = f($Hn, $noise)

    rpb = RProximalBundle($Hn, retract_func, transport_func,
                         (x) -> f($Hn, x), (x) -> ∂f($Hn, x),
                         $noise, init_obj, init_subgrad;
                         $rpb_kwargs...)
    run!(rpb)
end
#
experiments = ["RCBM", "RPB"]
records = [rcbm_record, rpb_record]
results = [rcbm_result, rpb_result]
times = [
    median(rcbm_bm).time * 1e-9,
    median(rpb_bm).time * 1e-9,
]
#
global B = cat(
    experiments,
    [maximum(first.(record)) for record in records],
    [t for t in times],
    [minimum([r[2] for r in record]) for record in records],
    [distance(Hn, noise, result) / length(noise) for result in results];
    dims=2,
)
#
global header = ["Algorithm", "Iterations", "Time (s)", "Objective", "Error"]
#
# Finalize - export costs
if export_table
    for (time, record, result, experiment) in zip(times, records, results, experiments)
        A = cat(first.(record), [r[2] for r in record]; dims=2)
        CSV.write(
            joinpath(results_folder, experiment_name * "_" * experiment * "-result.csv"),
            DataFrame(A, :auto);
            header=["i", "cost"],
        )
    end
    CSV.write(
        joinpath(results_folder, experiment_name * "-comparisons.csv"),
        DataFrame(B, :auto);
        header=header,
    )
end
end



if export_result
# Convert hyperboloid points to Poincaré ball points
ball_b = convert.(PoincareBallPoint, rcbm_result)
ball_p = convert.(PoincareBallPoint, pba_result)
ball_s = convert.(PoincareBallPoint, sgm_result)
ball_j = convert.(PoincareBallPoint, rpb_result)
#
# Plot results
plot!(
    ball_scene,
    H,
    ball_b;
    markercolor=result_color,
    markerstrokecolor=result_color,
    label="Convex Bundle Method",
)

# Add PBA result to plot
plot!(
    ball_scene,
    H,
    ball_p;
    markercolor="#dc267f",
    markerstrokecolor="#dc267f",
    label="Proximal Bundle Method",
)

# Add RPB result to plot
plot!(
    ball_scene,
    H,
    ball_j;
    markercolor="#785ef0",
    markerstrokecolor="#785ef0",
    label="RPB Method",
)
#
# Write csv files
matrix_b = matrixify_Poincare_ball(ball_b)
CSV.write(
    joinpath(results_folder, experiment_name * "-bundle_optimum.csv"),
    DataFrame(matrix_b, :auto);
    header=["x", "y"],
)

# Write PBA result to CSV
matrix_p = matrixify_Poincare_ball(ball_p)
CSV.write(
    joinpath(results_folder, experiment_name * "-pba_optimum.csv"),
    DataFrame(matrix_p, :auto);
    header=["x", "y"],
)

# Write RPB result to CSV
matrix_j = matrixify_Poincare_ball(ball_j)
CSV.write(
    joinpath(results_folder, experiment_name * "-rpb_optimum.csv"),
    DataFrame(matrix_j, :auto);
    header=["x", "y"],
)
#
# Suppress some plots for clarity, since they are visually indistinguishable
plot!(ball_scene, H, ball_s; label="Subgradient Method")
display(ball_scene)
println("\nPress Enter to continue...")
readline()
end