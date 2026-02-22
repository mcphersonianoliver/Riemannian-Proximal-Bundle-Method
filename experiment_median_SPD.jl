# used packages
using PrettyTables
using BenchmarkTools, Statistics
using CSV, DataFrames
using QuadraticModels, RipQP
using LinearAlgebra, LRUCache, Random
using ManifoldDiff, Manifolds, Manopt, ManoptExamples

# Include custom Riemannian Proximal Bundle solver
include("src/RiemannianProximalBundle.jl")

# Helper function for inner product (used by RPB solver)
function inner_product(M, p, X, Y)
    return inner(M, p, X, Y)
end

# Custom first-order retraction for SPD manifold: R_p(X) = proj_SPD(p + X)
# Note: ManifoldsBase v2 dispatch is retract! → _retract! (5-arg only, no t parameter).
# The caller pre-scales X, so we define the 5-arg form to intercept before _retract!.
struct FirstOrderSPDRetraction <: AbstractRetractionMethod end

function Manifolds.retract!(::SymmetricPositiveDefinite, q, p, X, ::FirstOrderSPDRetraction)
    @. q = p + X
    q .= (q + q') / 2
    try
        cholesky(q)
    catch
        F = eigen(q)
        F.values .= max.(F.values, 1e-12)
        q .= F.vectors * Diagonal(F.values) * F.vectors'
    end
    return q
end

# --- feature flags (top of file) ---
const benchmarking  = "--bench" in ARGS || get(ENV, "BENCHMARKING", "") == "1"
const export_table  = "--export" in ARGS || get(ENV, "EXPORT_TABLE", "") == "1"

# Two-phase experiment configuration
const TWO_PHASE_MODE = true  # Set to false to run old single-phase experiment
const PHASE1_MULTIPLIER = 2  # Run Phase 1 for this many times the normal iterations
const PHASE2_GAP_TOL = 1e-12 # Stop Phase 2 when objective gap reaches this tolerance
maxiter = 1000  # Maximum iterations for single-phase or Phase 2

println("Configuration:")
println("  Benchmarking: $benchmarking")
println("  Export tables: $export_table")
println("  Two-phase mode: $TWO_PHASE_MODE")
if TWO_PHASE_MODE
    println("  Phase 1 multiplier: $PHASE1_MULTIPLIER")
    println("  Phase 2 gap tolerance: $PHASE2_GAP_TOL")
end
println()


# initialize parameters of experiment
experiment_name = "RCBM-Median-20-Points"
results_folder = joinpath(@__DIR__, experiment_name)
!isdir(results_folder) && mkdir(results_folder)
seed_argument = 57

atol = 1e-12
N = 20 # number of data points
spd_dims = [3, 5, 15, 30, 55]

# Data folder for saving objective gaps (separated from plotting)
data_folder = joinpath(@__DIR__, "RCBM Median $N Points")
isdir(data_folder) || mkpath(data_folder)

function save_record_csv(data_folder, prefix, method_name, record)
    df = DataFrame(iteration = [r[1] for r in record], objective = [r[2] for r in record])
    CSV.write(joinpath(data_folder, "$(prefix)_$(method_name)_objectives.csv"), df)
    println("  Saved $(method_name) objectives to CSV")
end

# Generate a point that is at most `tol` close to the point `p` on `M`
function close_point(M, p, tol; retraction_method=Manifolds.default_retraction_method(M, typeof(p)))
    X = rand(M; vector_at = p)
    X .= tol * rand() * X / norm(M, p, X)
    return retract(M, p, X, retraction_method)
end

# Objective and subdifferential
f(M, p, data) = sum(1 / length(data) * distance.(Ref(M), Ref(p), data))
domf(M, p, centroid, diameter) = distance(M, p, centroid) < diameter / 2 ? true : false
function ∂f(M, p, data, atol=atol)
    return sum(
        1 / length(data) *
        ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 1; atol=atol),
    )
end

# Solver parameters
rcbm_kwargs(diameter, domf, k_max, k_min) = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :count => [:Cost, :SubGradient],
    :domain => domf,
    :debug => [
        :Iteration,
        (:Cost, "F(p): %1.16f "),
        (:ξ, "ξ: %1.8f "),
        (:last_stepsize, "step size: %1.8f"),
        :Stop,
        1000,
        "\n",
    ],
    :diameter => diameter,
    :k_max => k_max,
    :k_min => k_min,
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
]
rcbm_bm_kwargs(diameter, domf, k_max, k_min) = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :diameter => diameter,
    :domain => domf,
    :k_max => k_max,
    :k_min => k_min,
]
pba_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :count => [:Cost, :SubGradient],
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
    :stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(maxiter),
]
pba_bm_kwargs = [:cache => (:LRU, [:Cost, :SubGradient], 50),]
sgm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :count => [:Cost, :SubGradient],
    :debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
    :stopping_criterion => StopWhenSubgradientNormLess(√atol) | StopAfterIteration(maxiter),
]
sgm_bm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
    :stopping_criterion => StopWhenSubgradientNormLess(√atol) | StopAfterIteration(maxiter),
]

# Initialize dataframes for results
global col_names_1 = [
    :Dimension,
    :Iterations_1,
    :Time_1,
    :Objective_1,
    :Iterations_2,
    :Time_2,
    :Objective_2,
    :Iterations_3,
    :Time_3,
    :Objective_3,
]
col_types_1 = [
    Int64,
    Int64,
    Float64,
    Float64,
    Int64,
    Float64,
    Float64,
    Int64,
    Float64,
    Float64,
]
named_tuple_1 = (; zip(col_names_1, type[] for type in col_types_1 )...)

global col_names_2 = [
    :Dimension,
    :Iterations,
    :Time,
    :Objective,
]
col_types_2 = [
    Int64,
    Int64,
    Float64,
    Float64,
]

named_tuple_2 = (; zip(col_names_2, type[] for type in col_types_2 )...)

function initialize_dataframes(results_folder, experiment_name, subexperiment_name, named_tuple_1, named_tuple_2)
    A1 = DataFrame(named_tuple_1)
    CSV.write(
        joinpath(
            results_folder,
            experiment_name * "_$subexperiment_name" * "-Comparisons-Convex-Prox.csv",
        ),
        A1;
        header=false,
    )
    A2 = DataFrame(named_tuple_2)
    CSV.write(
        joinpath(
            results_folder,
            experiment_name * "_$subexperiment_name" * "-Comparisons-Subgrad.csv",
        ),
        A2;
        header=false,
    )
    return A1, A2
end

function export_dataframes(M, records, times, results_folder, experiment_name, subexperiment_name, col_names_1, col_names_2)
    B1 = DataFrame(;
        Dimension=manifold_dimension(M),
        Iterations_1=maximum(first.(records[1])),
        Time_1=times[1],
        Objective_1=minimum([r[2] for r in records[1]]),
        Iterations_2=maximum(first.(records[2])),
        Time_2=times[2],
        Objective_2=minimum([r[2] for r in records[2]]),
        Iterations_3=length(records) > 3 ? maximum(first.(records[3])) : 0,
        Time_3=length(times) > 3 ? times[3] : 0.0,
        Objective_3=length(records) > 3 ? minimum([r[2] for r in records[3]]) : 0.0,
    )
    B2 = DataFrame(;
        Dimension=manifold_dimension(M),
        Iterations=maximum(first.(records[length(records)])),
        Time=times[length(times)],
        Objective=minimum([r[2] for r in records[length(records)]]),
    )
    return B1, B2
end

function write_dataframes(B1, B2, results_folder, experiment_name, subexperiment_name)
    CSV.write(
        joinpath(
            results_folder,
            experiment_name *
            "_$subexperiment_name" *
            "-Comparisons-Convex-Prox.csv",
        ),
        B1;
        append=true,
    )
    CSV.write(
        joinpath(
            results_folder,
            experiment_name *
            "_$subexperiment_name" *
            "-Comparisons-Subgrad.csv",
        ),
        B2;
        append=true,
    )
end

# --- Riemannian Median Experiment on SPD Manifold ---
subexperiment_name = "SPD"
k_max_spd = 0.0
k_min_spd = -1/2

global A1_SPD, A2_SPD = initialize_dataframes(
    results_folder,
    experiment_name,
    subexperiment_name,
    named_tuple_1,
    named_tuple_2
)

# Run experiments for each SPD dimension
for n in spd_dims
    Random.seed!(seed_argument)

    M = SymmetricPositiveDefinite(Int(n))
    data_spd = [rand(M) for _ in 1:N]
    dists = [distance(M, z, y) for z in data_spd, y in data_spd]
    diameter_spd = 2 * maximum(dists)
    p0 = data_spd[minimum(Tuple(findmax(dists)[2]))]
    
    f_spd(M, p) = f(M, p, data_spd)
    domf_spd(M, p) = domf(M, p, p0, diameter_spd)
    ∂f_spd(M, p) = ∂f(M, p, data_spd, atol)

    initial_obj_spd = f_spd(M, p0)
    initial_entry = (0, initial_obj_spd, p0)

    # Set up manifold-specific functions for RPB
    function retraction_exp(x, v)
        return exp(M, x, v)
    end

    function transport_exp(x, y, v)
        return vector_transport_to(M, x, v, y, ParallelTransport())
    end

    # Define first-order retraction and projection transport for additional RPB run
    function retraction_first_order(x, v)
        # First-order Taylor approximation: R_P(X) = P + X
        temp = x + v
        # Make symmetric
        temp = (temp + temp') / 2

        try
            # Check if positive definite
            F = cholesky(temp)
            return temp
        catch
            # If not positive definite, use eigenvalue projection
            F = eigen(temp)
            # Ensure all eigenvalues are positive
            min_eig = 1e-12
            F.values .= max.(F.values, min_eig)
            return F.vectors * Diagonal(F.values) * F.vectors'
        end
    end

    function transport_projection(x, y, v)
        return vector_transport_to(M, x, v, y, ProjectionTransport())
    end

    if TWO_PHASE_MODE
        # PHASE 1: Find the true minimum by running methods for extended iterations
        println("=== PHASE 1: Finding true minimum (SPD dimension $n) ===")
        phase1_maxiter = maxiter * PHASE1_MULTIPLIER

        # Phase 1 solver configurations with extended iterations
        rcbm_kwargs_phase1 = [
            :cache => (:LRU, [:Cost, :SubGradient], 50),
            :count => [:Cost, :SubGradient],
            :domain => domf_spd,
            :debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
            :diameter => diameter_spd,
            :k_max => k_max_spd,
            :k_min => k_min_spd,
            :record => [:Iteration, :Cost, :Iterate],
            :return_state => true,
            :stopping_criterion => StopAfterIteration(phase1_maxiter),
        ]

        pba_kwargs_phase1 = [
            :cache => (:LRU, [:Cost, :SubGradient], 50),
            :count => [:Cost, :SubGradient],
            :debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
            :record => [:Iteration, :Cost, :Iterate],
            :return_state => true,
            :stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(phase1_maxiter),
        ]

        sgm_kwargs_phase1 = [
            :cache => (:LRU, [:Cost, :SubGradient], 50),
            :count => [:Cost, :SubGradient],
            :debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
            :record => [:Iteration, :Cost, :Iterate],
            :return_state => true,
            :stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
            :stopping_criterion => StopWhenSubgradientNormLess(√atol) | StopAfterIteration(phase1_maxiter),
        ]

        # Run Phase 1 to find approximate true minimum
        # println("Phase 1: RCBM")
        # rcbm_phase1 = convex_bundle_method(M, f_spd, ∂f_spd, p0; rcbm_kwargs_phase1...)
        # rcbm_min_obj = minimum([r[2] for r in get_record(rcbm_phase1)])

        println("Phase 1: PBA")
        pba_phase1 = proximal_bundle_method(M, f_spd, ∂f_spd, p0; pba_kwargs_phase1...)
        pba_min_obj = minimum([r[2] for r in get_record(pba_phase1)])

        println("Phase 1: RPB (Exponential)")
        initial_obj = f_spd(M, p0)
        initial_subgrad = ∂f_spd(M, p0)

        rpb_solver_phase1 = RProximalBundle(
            M, retraction_exp, transport_exp,
            (x) -> f_spd(M, x), (x) -> ∂f_spd(M, x),
            p0, initial_obj, initial_subgrad;
            max_iter=phase1_maxiter, tolerance=atol,
            proximal_parameter=1.0, trust_parameter=0.1,
            adaptive_proximal=true, know_minimizer=false, relative_error=false
        )
        run!(rpb_solver_phase1)
        rpb_min_obj = minimum(rpb_solver_phase1.objective_history)

        println("Phase 1: RPB (First-Order + Projection)")
        rpb_fo_solver_phase1 = RProximalBundle(
            M, retraction_first_order, transport_projection,
            (x) -> f_spd(M, x), (x) -> ∂f_spd(M, x),
            p0, initial_obj, initial_subgrad;
            max_iter=phase1_maxiter, tolerance=atol,
            proximal_parameter=1.0, trust_parameter=0.1,
            adaptive_proximal=true, know_minimizer=false, relative_error=false
        )
        run!(rpb_fo_solver_phase1)
        rpb_fo_min_obj = minimum(rpb_fo_solver_phase1.objective_history)

        println("Phase 1: SGM")
        sgm_phase1 = subgradient_method(M, f_spd, ∂f_spd, p0; sgm_kwargs_phase1...)
        sgm_min_obj = minimum([r[2] for r in get_record(sgm_phase1)])

        # Find the best approximation to the true minimum
        true_min_estimate = min(pba_min_obj, rpb_min_obj, rpb_fo_min_obj, sgm_min_obj)
        println("Phase 1 complete. Estimated true minimum: $true_min_estimate")

        # PHASE 2: Run methods until gap tolerance is reached
        println("=== PHASE 2: Convergence comparison with objective gap (SPD dimension $n) ===")

        # Phase 2 stopping criterion: stop when objective gap is small enough
        gap_stopping_criterion = StopWhenCostLess(true_min_estimate + PHASE2_GAP_TOL)

        # Phase 2 solver configurations
        rcbm_kwargs_phase2 = [
            :cache => (:LRU, [:Cost, :SubGradient], 50),
            :count => [:Cost, :SubGradient],
            :domain => domf_spd,
            :debug => [:Iteration, (:Cost, "F(p): %1.16f "), (:ξ, "ξ: %1.8f "), (:last_stepsize, "step size: %1.8f"), :Stop, 1000, "\n"],
            :diameter => diameter_spd,
            :k_max => k_max_spd,
            :k_min => k_min_spd,
            :record => [:Iteration, :Cost, :Iterate],
            :return_state => true,
            :stopping_criterion => gap_stopping_criterion | StopAfterIteration(maxiter),
        ]

        pba_kwargs_phase2 = [
            :cache => (:LRU, [:Cost, :SubGradient], 50),
            :count => [:Cost, :SubGradient],
            :debug => [:Iteration, (:Cost, "F(p): %1.16f "), (:ν, "ν: %1.16f "), (:c, "c: %1.16f "), (:μ, "μ: %1.8f "), :Stop, 1000, "\n"],
            :record => [:Iteration, :Cost, :Iterate],
            :return_state => true,
            :stopping_criterion => (StopWhenLagrangeMultiplierLess(atol) | gap_stopping_criterion | StopAfterIteration(maxiter)),
        ]

        sgm_kwargs_phase2 = [
            :cache => (:LRU, [:Cost, :SubGradient], 50),
            :count => [:Cost, :SubGradient],
            :debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
            :record => [:Iteration, :Cost, :Iterate],
            :return_state => true,
            :stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
            :stopping_criterion => (StopWhenSubgradientNormLess(√atol) | gap_stopping_criterion | StopAfterIteration(maxiter)),
        ]

        # Run Phase 2 methods with timing
        println("Phase 2: RCBM")
        rcbm_start_time = time()
        rcbm = convex_bundle_method(M, f_spd, ∂f_spd, p0; rcbm_kwargs_phase2...)
        rcbm_end_time = time()
        rcbm_result = get_solver_result(rcbm)
        rcbm_record_raw = get_record(rcbm)
        rcbm_record = vcat([initial_entry], rcbm_record_raw)

        # Create time-based record for RCBM
        rcbm_times = [0.0]  # Start at time 0
        rcbm_time_step = (rcbm_end_time - rcbm_start_time) / length(rcbm_record_raw)
        for i in 1:length(rcbm_record_raw)
            push!(rcbm_times, i * rcbm_time_step)
        end
        rcbm_time_record = [(rcbm_times[i], rcbm_record[i][2]) for i in 1:length(rcbm_record)]

        println("Phase 2: PBA")
        pba_start_time = time()
        pba = proximal_bundle_method(M, f_spd, ∂f_spd, p0; pba_kwargs_phase2...)
        pba_end_time = time()
        pba_result = get_solver_result(pba)
        pba_record_raw = get_record(pba)
        pba_record = vcat([initial_entry], pba_record_raw)

        # Create time-based record for PBA
        pba_times = [0.0]  # Start at time 0
        pba_time_step = (pba_end_time - pba_start_time) / length(pba_record_raw)
        for i in 1:length(pba_record_raw)
            push!(pba_times, i * pba_time_step)
        end
        pba_time_record = [(pba_times[i], pba_record[i][2]) for i in 1:length(pba_record)]

        println("Phase 2: RPB (Exponential)")
        initial_obj = f_spd(M, p0)
        initial_subgrad = ∂f_spd(M, p0)

        # RPB with gap tolerance as stopping criterion
        rpb_solver = RProximalBundle(
            M, retraction_exp, transport_exp,
            (x) -> f_spd(M, x), (x) -> ∂f_spd(M, x),
            p0, initial_obj, initial_subgrad;
            max_iter=maxiter, tolerance=PHASE2_GAP_TOL,
            proximal_parameter=1.0, trust_parameter=0.1,
            adaptive_proximal=true, know_minimizer=true, relative_error=false,
            true_min_obj=true_min_estimate
        )
        rpb_start_time = time()
        run!(rpb_solver)
        rpb_end_time = time()

        # Convert RPB results to match expected format (iteration, objective) pairs
        # Use raw_objective_history instead of objective_history (which contains gaps)
        rpb_iterations = collect(0:length(rpb_solver.raw_objective_history)-1)
        rpb_objectives = rpb_solver.raw_objective_history
        rpb_record = [(iter, obj) for (iter, obj) in zip(rpb_iterations, rpb_objectives)]

        # Create time-based record for RPB
        rpb_total_time = rpb_end_time - rpb_start_time
        rpb_time_step = rpb_total_time / (length(rpb_solver.raw_objective_history) - 1)
        rpb_time_record = [(i * rpb_time_step, rpb_solver.raw_objective_history[i+1]) for i in 0:length(rpb_solver.raw_objective_history)-1]

        println("Phase 2: RPB (First-Order + Projection)")
        # RPB with first-order retraction and projection transport
        rpb_fo_solver = RProximalBundle(
            M, retraction_first_order, transport_projection,
            (x) -> f_spd(M, x), (x) -> ∂f_spd(M, x),
            p0, initial_obj, initial_subgrad;
            max_iter=maxiter, tolerance=PHASE2_GAP_TOL,
            retraction_error = 1.0, transport_error = 1.0,
            proximal_parameter=1.0, trust_parameter=0.1,
            adaptive_proximal=true, know_minimizer=true, relative_error=false,
            true_min_obj=true_min_estimate
        )
        rpb_fo_start_time = time()
        run!(rpb_fo_solver)
        rpb_fo_end_time = time()

        # Convert RPB first-order results to match expected format
        rpb_fo_iterations = collect(0:length(rpb_fo_solver.raw_objective_history)-1)
        rpb_fo_objectives = rpb_fo_solver.raw_objective_history
        rpb_fo_record = [(iter, obj) for (iter, obj) in zip(rpb_fo_iterations, rpb_fo_objectives)]

        # Create time-based record for RPB first-order
        rpb_fo_total_time = rpb_fo_end_time - rpb_fo_start_time
        rpb_fo_time_step = rpb_fo_total_time / (length(rpb_fo_solver.raw_objective_history) - 1)
        rpb_fo_time_record = [(i * rpb_fo_time_step, rpb_fo_solver.raw_objective_history[i+1]) for i in 0:length(rpb_fo_solver.raw_objective_history)-1]

        println("Phase 2: SGM")
        sgm_start_time = time()
        sgm = subgradient_method(M, f_spd, ∂f_spd, p0; sgm_kwargs_phase2...)
        sgm_end_time = time()
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

        # --- First-order retraction Phase 2 runs ---
        println("Phase 2: RCBM-FO")
        rcbm_fo_start_time = time()
        rcbm_fo = convex_bundle_method(M, f_spd, ∂f_spd, p0; rcbm_kwargs_phase2...,
            retraction_method=FirstOrderSPDRetraction(), vector_transport_method=ProjectionTransport())
        rcbm_fo_end_time = time()
        rcbm_fo_result = get_solver_result(rcbm_fo)
        rcbm_fo_record_raw = get_record(rcbm_fo)
        rcbm_fo_record = vcat([initial_entry], rcbm_fo_record_raw)

        # Create time-based record for RCBM-FO
        rcbm_fo_times = [0.0]
        rcbm_fo_time_step = (rcbm_fo_end_time - rcbm_fo_start_time) / length(rcbm_fo_record_raw)
        for i in 1:length(rcbm_fo_record_raw)
            push!(rcbm_fo_times, i * rcbm_fo_time_step)
        end
        rcbm_fo_time_record = [(rcbm_fo_times[i], rcbm_fo_record[i][2]) for i in 1:length(rcbm_fo_record)]

        println("Phase 2: PBA-FO")
        pba_fo_start_time = time()
        pba_fo = proximal_bundle_method(M, f_spd, ∂f_spd, p0; pba_kwargs_phase2...,
            retraction_method=FirstOrderSPDRetraction(), vector_transport_method=ProjectionTransport())
        pba_fo_end_time = time()
        pba_fo_result = get_solver_result(pba_fo)
        pba_fo_record_raw = get_record(pba_fo)
        pba_fo_record = vcat([initial_entry], pba_fo_record_raw)

        # Create time-based record for PBA-FO
        pba_fo_times = [0.0]
        pba_fo_time_step = (pba_fo_end_time - pba_fo_start_time) / length(pba_fo_record_raw)
        for i in 1:length(pba_fo_record_raw)
            push!(pba_fo_times, i * pba_fo_time_step)
        end
        pba_fo_time_record = [(pba_fo_times[i], pba_fo_record[i][2]) for i in 1:length(pba_fo_record)]

        println("Phase 2: SGM-FO")
        sgm_fo_start_time = time()
        sgm_fo = subgradient_method(M, f_spd, ∂f_spd, p0; sgm_kwargs_phase2...,
            retraction_method=FirstOrderSPDRetraction())
        sgm_fo_end_time = time()
        sgm_fo_result = get_solver_result(sgm_fo)
        sgm_fo_record_raw = get_record(sgm_fo)
        sgm_fo_record = vcat([initial_entry], sgm_fo_record_raw)

        # Create time-based record for SGM-FO
        sgm_fo_times = [0.0]
        sgm_fo_time_step = (sgm_fo_end_time - sgm_fo_start_time) / length(sgm_fo_record_raw)
        for i in 1:length(sgm_fo_record_raw)
            push!(sgm_fo_times, i * sgm_fo_time_step)
        end
        sgm_fo_time_record = [(sgm_fo_times[i], sgm_fo_record[i][2]) for i in 1:length(sgm_fo_record)]

        println("Phase 2 complete. SPD dimension $n")

        # --- Save objective gaps to CSV and free solver state ---
        dim_prefix = "spd_$(n)x$(n)"

        # Save iteration-based records
        save_record_csv(data_folder, dim_prefix, "rpb", rpb_record)
        save_record_csv(data_folder, dim_prefix, "rpb_fo", rpb_fo_record)
        save_record_csv(data_folder, dim_prefix, "rcbm", rcbm_record)
        save_record_csv(data_folder, dim_prefix, "rcbm_fo", rcbm_fo_record)
        save_record_csv(data_folder, dim_prefix, "pba", pba_record)
        save_record_csv(data_folder, dim_prefix, "pba_fo", pba_fo_record)
        save_record_csv(data_folder, dim_prefix, "sgm", sgm_record)
        save_record_csv(data_folder, dim_prefix, "sgm_fo", sgm_fo_record)

        # Save wall clock times
        wallclock_df = DataFrame(
            method = ["RPB", "RPB-FO", "RCBM", "RCBM-FO", "PBA", "PBA-FO", "SGM", "SGM-FO"],
            wallclock_seconds = [
                rpb_end_time - rpb_start_time,
                rpb_fo_end_time - rpb_fo_start_time,
                rcbm_end_time - rcbm_start_time,
                rcbm_fo_end_time - rcbm_fo_start_time,
                pba_end_time - pba_start_time,
                pba_fo_end_time - pba_fo_start_time,
                sgm_end_time - sgm_start_time,
                sgm_fo_end_time - sgm_fo_start_time,
            ],
        )
        CSV.write(joinpath(data_folder, "$(dim_prefix)_wallclock.csv"), wallclock_df)

        # Save experiment parameters
        params_df = DataFrame(
            key = ["true_min_estimate", "N", "n", "seed", "atol", "maxiter"],
            value = [true_min_estimate, N, n, seed_argument, atol, maxiter],
        )
        CSV.write(joinpath(data_folder, "$(dim_prefix)_params.csv"), params_df)

        println("Data saved to: $data_folder (prefix: $dim_prefix)")

        # Free Phase 1 and Phase 2 solver state
        rpb_solver = nothing; rpb_fo_solver = nothing
        rpb_solver_phase1 = nothing; rpb_fo_solver_phase1 = nothing
        rcbm = nothing; rcbm_fo = nothing
        pba = nothing; pba_fo = nothing; pba_phase1 = nothing
        sgm = nothing; sgm_fo = nothing; sgm_phase1 = nothing
        GC.gc()

    else
        # Original single-phase approach
        println("Running all methods on SPD dimension $n ...")

        # --- Exponential retraction methods ---
        println("RCBM")
        rcbm = convex_bundle_method(M, f_spd, ∂f_spd, p0; rcbm_kwargs(diameter_spd, domf_spd, k_max_spd, k_min_spd)...)
        rcbm_result = get_solver_result(rcbm)
        rcbm_record_raw = get_record(rcbm)
        rcbm_record = vcat([initial_entry], rcbm_record_raw)

        println("PBA")
        pba = proximal_bundle_method(M, f_spd, ∂f_spd, p0; pba_kwargs...)
        pba_result = get_solver_result(pba)
        pba_record_raw = get_record(pba)
        pba_record = vcat([initial_entry], pba_record_raw)

        println("RPB (Exponential)")
        initial_obj = f_spd(M, p0)
        initial_subgrad = ∂f_spd(M, p0)

        rpb_solver = RProximalBundle(
            M, retraction_exp, transport_exp,
            (x) -> f_spd(M, x), (x) -> ∂f_spd(M, x),
            p0, initial_obj, initial_subgrad;
            max_iter=maxiter, tolerance=atol,
            proximal_parameter=1.0, trust_parameter=0.1,
            adaptive_proximal=true, know_minimizer=false, relative_error=false
        )
        run!(rpb_solver)

        rpb_iterations = collect(0:length(rpb_solver.raw_objective_history)-1)
        rpb_objectives = rpb_solver.raw_objective_history
        rpb_record = [(iter, obj) for (iter, obj) in zip(rpb_iterations, rpb_objectives)]

        println("SGM")
        sgm = subgradient_method(M, f_spd, ∂f_spd, p0; sgm_kwargs...)
        sgm_result = get_solver_result(sgm)
        sgm_record_raw = get_record(sgm)
        sgm_record = vcat([initial_entry], sgm_record_raw)

        # --- First-order retraction methods ---
        println("RPB (First-Order + Projection)")
        rpb_fo_solver = RProximalBundle(
            M, retraction_first_order, transport_projection,
            (x) -> f_spd(M, x), (x) -> ∂f_spd(M, x),
            p0, initial_obj, initial_subgrad;
            retraction_error = 1.0, transport_error = 1.0,
            max_iter=maxiter, tolerance=atol,
            proximal_parameter=1.0, trust_parameter=0.1,
            adaptive_proximal=true, know_minimizer=false, relative_error=false
        )
        run!(rpb_fo_solver)

        rpb_fo_iterations = collect(0:length(rpb_fo_solver.raw_objective_history)-1)
        rpb_fo_objectives = rpb_fo_solver.raw_objective_history
        rpb_fo_record = [(iter, obj) for (iter, obj) in zip(rpb_fo_iterations, rpb_fo_objectives)]

        println("RCBM-FO")
        rcbm_fo = convex_bundle_method(M, f_spd, ∂f_spd, p0; rcbm_kwargs(diameter_spd, domf_spd, k_max_spd, k_min_spd)...,
            retraction_method=FirstOrderSPDRetraction(), vector_transport_method=ProjectionTransport())
        rcbm_fo_result = get_solver_result(rcbm_fo)
        rcbm_fo_record_raw = get_record(rcbm_fo)
        rcbm_fo_record = vcat([initial_entry], rcbm_fo_record_raw)

        println("PBA-FO")
        pba_fo = proximal_bundle_method(M, f_spd, ∂f_spd, p0; pba_kwargs...,
            retraction_method=FirstOrderSPDRetraction(), vector_transport_method=ProjectionTransport())
        pba_fo_result = get_solver_result(pba_fo)
        pba_fo_record_raw = get_record(pba_fo)
        pba_fo_record = vcat([initial_entry], pba_fo_record_raw)

        println("SGM-FO")
        sgm_fo = subgradient_method(M, f_spd, ∂f_spd, p0; sgm_kwargs...,
            retraction_method=FirstOrderSPDRetraction())
        sgm_fo_result = get_solver_result(sgm_fo)
        sgm_fo_record_raw = get_record(sgm_fo)
        sgm_fo_record = vcat([initial_entry], sgm_fo_record_raw)

        println("Completed SPD dimension $n")
    end

    if benchmarking
        rcbm_bm = @benchmark convex_bundle_method($M, $f_spd, $∂f_spd, $p0; rcbm_bm_kwargs($diameter_spd, $domf_spd, $k_max_spd, $k_min_spd)...)
        pba_bm = @benchmark proximal_bundle_method($M, $f_spd, $∂f_spd, $p0; $pba_bm_kwargs...)

        # Benchmark RPB solver (exponential)
        rpb_bm = @benchmark begin
            initial_obj_bm = f_spd($M, $p0)
            initial_subgrad_bm = ∂f_spd($M, $p0)
            rpb_solver_bm = RProximalBundle(
                $M, (x, v) -> exp($M, x, v), (x, y, v) -> vector_transport_to($M, x, v, y, ParallelTransport()),
                (x) -> f_spd($M, x), (x) -> ∂f_spd($M, x),
                $p0, initial_obj_bm, initial_subgrad_bm;
                max_iter=$maxiter, tolerance=$atol,
                proximal_parameter=1.0, trust_parameter=0.1,
                adaptive_proximal=true, know_minimizer=false, relative_error=false
            )
            run!(rpb_solver_bm)
        end

        # Benchmark RPB solver (first-order + projection)
        rpb_fo_bm = @benchmark begin
            initial_obj_bm = f_spd($M, $p0)
            initial_subgrad_bm = ∂f_spd($M, $p0)
            rpb_fo_solver_bm = RProximalBundle(
                $M, $retraction_first_order, $transport_projection,
                (x) -> f_spd($M, x), (x) -> ∂f_spd($M, x),
                $p0, initial_obj_bm, initial_subgrad_bm;
                max_iter=$maxiter, tolerance=$atol,
                retraction_error = 1.0, transport_error = 1.0,
                proximal_parameter=1.0, trust_parameter=0.1,
                adaptive_proximal=true, know_minimizer=false, relative_error=false
            )
            run!(rpb_fo_solver_bm)
        end

        sgm_bm = @benchmark subgradient_method($M, $f_spd, $∂f_spd, $p0; $sgm_bm_kwargs...)

        times = [
            median(rpb_bm).time * 1e-9,
            median(rpb_fo_bm).time * 1e-9,
            median(rcbm_bm).time * 1e-9,
            median(pba_bm).time * 1e-9,
            median(sgm_bm).time * 1e-9,
        ]

        B1_SPD, B2_SPD = export_dataframes(
            M,
            records_all,
            times,
            results_folder,
            experiment_name,
            subexperiment_name,
            col_names_1,
            col_names_2,
        )

        append!(A1_SPD, B1_SPD)
        append!(A2_SPD, B2_SPD)
        (export_table) && (write_dataframes(B1_SPD, B2_SPD, results_folder, experiment_name, subexperiment_name))
    end

end