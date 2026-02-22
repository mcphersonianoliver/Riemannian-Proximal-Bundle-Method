using CSV, DataFrames
using Plots, Plots.PlotMeasures, LaTeXStrings

pgfplotsx()

# --- Load experiment data ---
experiment_name = "denoising_TV_hyperbolic"
data_folder = joinpath(@__DIR__, "Denoising TV Hyperbolic Data")
results_folder = joinpath(@__DIR__, "Denoising TV Hyperbolic Results")
isdir(results_folder) || mkpath(results_folder)

# Load objective histories
function load_record(data_folder, experiment_name, method)
    df = CSV.read(joinpath(data_folder, "$(experiment_name)_$(method)_objectives.csv"), DataFrame)
    return collect(zip(df.iteration, df.objective))
end

rpb_record = load_record(data_folder, experiment_name, "rpb")
rcbm_record = load_record(data_folder, experiment_name, "rcbm")
pba_record = load_record(data_folder, experiment_name, "pba")
sgm_record = load_record(data_folder, experiment_name, "sgm")

# Load wall clock times
wallclock_df = CSV.read(joinpath(data_folder, "$(experiment_name)_wallclock.csv"), DataFrame)
wallclock_map = Dict(zip(wallclock_df.method, wallclock_df.wallclock_seconds))
rpb_total_time = wallclock_map["RPB"]
rcbm_total_time = wallclock_map["RCBM"]
pba_total_time = wallclock_map["PBA"]
sgm_total_time = wallclock_map["SGM"]

# Load experiment parameters
params_df = CSV.read(joinpath(data_folder, "$(experiment_name)_params.csv"), DataFrame)
params_map = Dict(zip(params_df.key, params_df.value))
true_min_estimate = params_map["true_min_estimate"]

println("Data loaded from: $data_folder")
println("  RPB: $(length(rpb_record)) iterations, $(rpb_total_time)s")
println("  RCBM: $(length(rcbm_record)) iterations, $(rcbm_total_time)s")
println("  PBA: $(length(pba_record)) iterations, $(pba_total_time)s")
println("  SGM: $(length(sgm_record)) iterations, $(sgm_total_time)s")
println("  True min estimate: $true_min_estimate")

# --- Plotting function ---
function plot_objective_gap_convergence(records, method_names, true_min_estimate;
                                   xlabel=nothing,
                                   ylabel=nothing,
                                   filename=nothing,
                                   plot_current=false,
                                   xscale_log=false,
                                   show_reference=true,
                                   show_legend=true,
                                   wallclock=false,
                                   wallclock_times=nothing,
                                   ylims=(0.00001, Inf),
                                   max_x=nothing)
    if ylabel === nothing
        ylabel = plot_current ? L"\textrm{Current Objective Gap}" : L"\textrm{Minimum Objective Gap}"
    end
    if xlabel === nothing
        xlabel = wallclock ? L"\textrm{Time (seconds)}" : L"\textrm{Oracle Calls}"
    end

    p = plot(
        xlabel=xlabel,
        ylabel=ylabel,
        yscale=:log10,
        xscale=xscale_log ? :log10 : :identity,
        ylims=ylims,
        legend=show_legend ? :topright : false,
        size=(600, 400),
        guidefontsize=18,
        tickfontsize=14,
        legendfontsize=14,
        legendtitlefontsize=14,
        background_color_legend=:white,
        foreground_color_legend=:black,
        grid=true,
        gridcolor=:lightgray,
        gridwidth=0.5,
        gridalpha=0.3,
        linewidth=2.5,
        margin=1mm,
        extra_kwargs=Dict(:subplot => Dict("width" => raw"12cm", "height" => raw"8cm")),
    )

    color_map = Dict(
        L"\textrm{RPB}" => "#785ef0",
        L"\textrm{RPB (Ours)}" => "#785ef0",
        L"\textrm{PBA}" => "#dc267f",
        L"\textrm{RCBM}" => "#fe6100",
        L"\textrm{SGM}" => "#ffb000",
    )
    fallback_colors = ["#785ef0", "#dc267f", "#fe6100", "#ffb000", :brown]

    max_iterations_across_all = maximum(length(record) for record in records if !isempty(record); init=0)
    max_iter_idx_global = max(1, Int(floor(1 * max_iterations_across_all)))

    first_method_data = nothing

    for (i, (record, name)) in enumerate(zip(records, method_names))
        if !isempty(record)
            if wallclock && wallclock_times !== nothing && i <= length(wallclock_times)
                total_time = wallclock_times[i]
                original_num_iters = length(record)
                time_per_iter = original_num_iters > 1 ? total_time / (original_num_iters - 1) : 0.0
                x_values = [r[1] * time_per_iter for r in record]
                objective_gaps = [max(r[2] - true_min_estimate, 1e-16) for r in record]
            else
                x_values = [r[1] + 1 for r in record]
                objective_gaps = [max(r[2] - true_min_estimate, 1e-16) for r in record]
            end

            if plot_current
                gaps_to_plot = objective_gaps
            else
                gaps_to_plot = [minimum(objective_gaps[1:j]) for j in 1:length(objective_gaps)]
            end

            actual_limit = min(max_iter_idx_global, length(x_values))
            x_values_limited = x_values[1:actual_limit]
            gaps_limited = gaps_to_plot[1:actual_limit]

            valid_indices = (x_values_limited .> 0) .& (gaps_limited .> 0)
            valid_x_values = x_values_limited[valid_indices]
            valid_gaps = gaps_limited[valid_indices]

            if i == 1 && !isempty(valid_x_values)
                first_method_data = (valid_x_values, valid_gaps)
            end

            # Downsample to at most 2000 points per line
            n_pts = length(valid_x_values)
            if n_pts > 2000
                stride = cld(n_pts, 2000)
                valid_x_values = valid_x_values[1:stride:end]
                valid_gaps = valid_gaps[1:stride:end]
            end

            if !isempty(valid_x_values)
                plot_color = get(color_map, name, fallback_colors[mod1(i, length(fallback_colors))])
                plot!(p, valid_x_values, valid_gaps,
                    label=name,
                    color=plot_color,
                    linewidth=2.5,
                    markershape=:circle,
                    markersize=0,
                    markerstrokewidth=0)
            end
        end
    end

    # O(1/√k) reference line
    if show_reference && first_method_data !== nothing && !wallclock
        first_x_values, first_gaps = first_method_data
        x_min = maximum([1, minimum(first_x_values)])
        x_max = maximum(first_x_values)
        x_range = range(x_min, x_max, length=100)

        C = first_gaps[1] * sqrt(first_x_values[1])
        reference_line = C .* (x_range .^ (-0.5))

        valid_ref_indices = (x_range .>= x_min) .& (x_range .<= x_max) .& (reference_line .> 0)
        x_ref = x_range[valid_ref_indices]
        y_ref = reference_line[valid_ref_indices]

        if !isempty(x_ref)
            plot!(p, x_ref, y_ref,
                label=L"O(1/\sqrt{k})",
                color=:gray,
                linestyle=:dot,
                linewidth=2.5)
        end
    end

    if max_x !== nothing
        current_xlims = Plots.xlims(p)
        xlims!(p, (current_xlims[1], max_x))
    end

    if filename !== nothing
        savefig(p, filename)
        println("Plot saved to: $filename")
    end

    return p
end

# --- Generate all plots ---

# SGM vs RPB
records = [rpb_record, sgm_record]
method_names = [L"\textrm{RPB (Ours)}", L"\textrm{SGM}"]

plot1 = plot_objective_gap_convergence(
    records, method_names, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-sgm-vs-rpb-loglog.pdf"),
    xscale_log=true,
    show_reference=true,
    show_legend=false,
)

plot2 = plot_objective_gap_convergence(
    records, method_names, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-sgm-vs-rpb-loglog-noref.pdf"),
    xscale_log=true,
    show_reference=false,
    show_legend=false,
)

plot1_current = plot_objective_gap_convergence(
    records, method_names, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-sgm-vs-rpb-current-loglog.pdf"),
    xscale_log=true,
    plot_current=true,
    show_reference=false,
    show_legend=false,
)

plot2_current = plot_objective_gap_convergence(
    records, method_names, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-sgm-vs-rpb-current-loglog-noref.pdf"),
    xscale_log=true,
    plot_current=true,
    show_reference=false,
    show_legend=false,
)

# All Bundle Methods
records_bundles = [rpb_record, rcbm_record, pba_record]
method_names_bundles = [L"\textrm{RPB (Ours)}", L"\textrm{RCBM}", L"\textrm{PBA}"]

plot3 = plot_objective_gap_convergence(
    records_bundles, method_names_bundles, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-all-bundles-wallclock.pdf"),
    wallclock=true,
    wallclock_times=[rpb_total_time, rcbm_total_time, pba_total_time],
    show_reference=false,
    ylabel="",
)

bundles_fastest_time = minimum([rpb_total_time, rcbm_total_time, pba_total_time])
plot3_short = plot_objective_gap_convergence(
    records_bundles, method_names_bundles, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-all-bundles-wallclock-short.pdf"),
    wallclock=true,
    wallclock_times=[rpb_total_time, rcbm_total_time, pba_total_time],
    show_reference=false,
    max_x=5 * bundles_fastest_time,
    ylabel="",
)

plot4 = plot_objective_gap_convergence(
    records_bundles, method_names_bundles, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-all-bundles-loglog.pdf"),
    xscale_log=true,
    show_reference=false,
    show_legend=false,
)

# All Methods
records_all = [rpb_record, rcbm_record, pba_record, sgm_record]
method_names_all = [L"\textrm{RPB (Ours)}", L"\textrm{RCBM}", L"\textrm{PBA}", L"\textrm{SGM}"]

plot5 = plot_objective_gap_convergence(
    records_all, method_names_all, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-all-methods-semilogx.pdf"),
    show_reference=false,
    show_legend=false,
)

plot6 = plot_objective_gap_convergence(
    records_all, method_names_all, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-all-methods-loglog.pdf"),
    xscale_log=true,
    show_reference=false,
    show_legend=false,
)

plot7 = plot_objective_gap_convergence(
    records_all, method_names_all, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-all-methods-wallclock.pdf"),
    wallclock=true,
    wallclock_times=[rpb_total_time, rcbm_total_time, pba_total_time, sgm_total_time],
    show_reference=false,
    ylabel="",
)

all_fastest_time = minimum([rpb_total_time, rcbm_total_time, pba_total_time, sgm_total_time])
plot7_short = plot_objective_gap_convergence(
    records_all, method_names_all, true_min_estimate;
    filename=joinpath(results_folder, experiment_name * "-all-methods-wallclock-short.pdf"),
    wallclock=true,
    wallclock_times=[rpb_total_time, rcbm_total_time, pba_total_time, sgm_total_time],
    show_reference=false,
    max_x=5 * all_fastest_time,
    ylabel="",
)

println("\nAll plots generated.")
