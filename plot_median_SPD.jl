using CSV, DataFrames
using Plots, Plots.PlotMeasures, LaTeXStrings

pgfplotsx()

# --- Configuration ---
N = 20  # number of data points used in experiment
spd_dims = [3, 5, 15, 30, 55]
atol = 1e-12

data_folder = joinpath(@__DIR__, "data", "RCBM Median $N Points")
results_folder = joinpath(@__DIR__, "plots", "RCBM Median $N Points")
isdir(results_folder) || mkpath(results_folder)

# --- Load helpers ---
function load_record(data_folder, prefix, method)
    filepath = joinpath(data_folder, "$(prefix)_$(method)_objectives.csv")
    if !isfile(filepath)
        println("  Warning: $filepath not found, skipping")
        return Tuple{Int,Float64}[]
    end
    df = CSV.read(filepath, DataFrame)
    return collect(zip(df.iteration, df.objective))
end

function load_wallclock(data_folder, prefix)
    filepath = joinpath(data_folder, "$(prefix)_wallclock.csv")
    if !isfile(filepath)
        return Dict{String,Float64}()
    end
    df = CSV.read(filepath, DataFrame)
    return Dict(zip(df.method, df.wallclock_seconds))
end

function load_params(data_folder, prefix)
    filepath = joinpath(data_folder, "$(prefix)_params.csv")
    if !isfile(filepath)
        return Dict{String,Float64}()
    end
    df = CSV.read(filepath, DataFrame)
    return Dict(zip(df.key, df.value))
end

# --- Plotting function ---
function plot_objective_gap_convergence(records, method_names, true_min_estimate;
                                       xlabel=nothing,
                                       ylabel=nothing,
                                       filename=nothing,
                                       show_legend=true,
                                       wallclock=false,
                                       wallclock_times=nothing,
                                       offset_iterations=false,
                                       ylims_lower=nothing,
                                       max_x=nothing)
    if ylabel === nothing
        ylabel = L"\textrm{Minimum Objective Gap}"
    end
    if xlabel === nothing
        xlabel = wallclock ? L"\textrm{Time (seconds)}" : L"\textrm{Oracle Calls}"
    end

    p = plot(
        xlabel=xlabel,
        ylabel=ylabel,
        yscale=:log10,
        legend=show_legend ? :topright : false,
        size=(600, 400),
        guidefontsize=18,
        tickfontsize=14,
        legendfontsize=14,
        background_color_legend=:white,
        foreground_color_legend=:black,
        grid=true,
        linewidth=2.5,
        margin=1mm,
        extra_kwargs=Dict(:subplot => Dict("width" => raw"12cm", "height" => raw"8cm")),
    )

    color_map = Dict(
        L"\textrm{RPB (Ours)}" => "#785ef0",
        L"\textrm{RPB-FO (Ours)}" => "#785ef0",
        L"\textrm{RCBM}" => "#fe6100",
        L"\textrm{RCBM-FO}" => "#fe6100",
        L"\textrm{PBA}" => "#dc267f",
        L"\textrm{PBA-FO}" => "#dc267f",
        L"\textrm{SGM}" => "#ffb000",
        L"\textrm{SGM-FO}" => "#ffb000",
    )
    fallback_colors = ["#785ef0", "#dc267f", "#fe6100", "#ffb000", "#333333"]

    line_style_map = Dict(
        L"\textrm{RPB (Ours)}" => :solid,
        L"\textrm{RPB-FO (Ours)}" => :dash,
        L"\textrm{RCBM}" => :solid,
        L"\textrm{RCBM-FO}" => :dash,
        L"\textrm{PBA}" => :solid,
        L"\textrm{PBA-FO}" => :dash,
        L"\textrm{SGM}" => :solid,
        L"\textrm{SGM-FO}" => :dash,
    )
    fallback_line_styles = [:solid, :dash, :solid, :dash, :solid, :dash, :solid, :dash]

    if wallclock && wallclock_times !== nothing
        time_records = records
        max_time_across_all = maximum(
            maximum(r[1] for r in tr) for tr in time_records if !isempty(tr);
            init=0.0
        )
        time_cutoff_global = max_time_across_all

        for (i, (time_record, name)) in enumerate(zip(time_records, method_names))
            if !isempty(time_record)
                times = [r[1] for r in time_record]
                objective_gaps = [max(r[2] - true_min_estimate, 1e-16) for r in time_record]
                min_gaps_so_far = [minimum(objective_gaps[1:j]) for j in 1:length(objective_gaps)]

                time_indices = times .<= time_cutoff_global
                times_limited = times[time_indices]
                min_gaps_limited = min_gaps_so_far[time_indices]

                valid_indices = (times_limited .>= 0) .& (min_gaps_limited .> 0)
                valid_times = times_limited[valid_indices]
                valid_gaps = min_gaps_limited[valid_indices]

                n_pts = length(valid_times)
                if n_pts > 2000
                    stride = cld(n_pts, 2000)
                    valid_times = valid_times[1:stride:end]
                    valid_gaps = valid_gaps[1:stride:end]
                end

                if !isempty(valid_times)
                    plot_color = get(color_map, name, fallback_colors[mod1(i, length(fallback_colors))])
                    plot_linestyle = get(line_style_map, name, fallback_line_styles[mod1(i, length(fallback_line_styles))])
                    plot_label = contains(string(name), "-FO") ? "" : name
                    plot!(p, valid_times, valid_gaps,
                          label=plot_label,
                          color=plot_color,
                          linestyle=plot_linestyle,
                          linewidth=2.5)
                end
            end
        end
    else
        max_iterations_across_all = maximum(length(r) for r in records if !isempty(r); init=0)
        max_iter_idx_global = max(1, Int(ceil(max_iterations_across_all)))

        for (i, (record, name)) in enumerate(zip(records, method_names))
            if !isempty(record)
                if offset_iterations
                    iterations = [r[1] + 1 for r in record]
                    valid_threshold = 0
                else
                    iterations = [r[1] for r in record]
                    valid_threshold = -1
                end

                objective_gaps = [max(r[2] - true_min_estimate, 1e-16) for r in record]
                min_gaps_so_far = [minimum(objective_gaps[1:j]) for j in 1:length(objective_gaps)]

                actual_limit = min(max_iter_idx_global, length(iterations))
                iterations_limited = iterations[1:actual_limit]
                min_gaps_limited = min_gaps_so_far[1:actual_limit]

                valid_indices = (iterations_limited .> valid_threshold) .& (min_gaps_limited .> 0)
                valid_iterations = iterations_limited[valid_indices]
                valid_gaps = min_gaps_limited[valid_indices]

                n_pts = length(valid_iterations)
                if n_pts > 2000
                    stride = cld(n_pts, 2000)
                    valid_iterations = valid_iterations[1:stride:end]
                    valid_gaps = valid_gaps[1:stride:end]
                end

                if !isempty(valid_iterations)
                    plot_color = get(color_map, name, fallback_colors[mod1(i, length(fallback_colors))])
                    plot_linestyle = get(line_style_map, name, fallback_line_styles[mod1(i, length(fallback_line_styles))])
                    plot_label = contains(string(name), "-FO") ? "" : name
                    plot!(p, valid_iterations, valid_gaps,
                        label=plot_label,
                        color=plot_color,
                        linestyle=plot_linestyle,
                        linewidth=2.5)
                end
            end
        end
    end

    if ylims_lower !== nothing
        current_ylims = Plots.ylims(p)
        ylims!(p, (ylims_lower, current_ylims[2]))
    end
    x_start = (!wallclock && offset_iterations) ? 1 : 0
    if max_x !== nothing
        xlims!(p, (x_start, max_x))
    else
        current_xlims = Plots.xlims(p)
        xlims!(p, (x_start, current_xlims[2]))
    end

    if filename !== nothing
        savefig(p, filename)
        println("Plot saved to: $filename")
    end

    return p
end

# --- Generate plots for each SPD dimension ---
method_keys = ["rpb", "rpb_fo", "rcbm", "rcbm_fo", "pba", "pba_fo", "sgm", "sgm_fo"]
wallclock_keys = ["RPB", "RPB-FO", "RCBM", "RCBM-FO", "PBA", "PBA-FO", "SGM", "SGM-FO"]
method_names_all = [
    L"\textrm{RPB (Ours)}", L"\textrm{RPB-FO (Ours)}",
    L"\textrm{RCBM}", L"\textrm{RCBM-FO}",
    L"\textrm{PBA}", L"\textrm{PBA-FO}",
    L"\textrm{SGM}", L"\textrm{SGM-FO}",
]

for n in spd_dims
    dim_prefix = "spd_$(n)x$(n)"
    println("\n--- Loading data for SPD $n×$n ---")

    # Load records
    records_all = [load_record(data_folder, dim_prefix, m) for m in method_keys]
    params = load_params(data_folder, dim_prefix)
    wc = load_wallclock(data_folder, dim_prefix)

    if isempty(params)
        println("  No params found for $dim_prefix, skipping")
        continue
    end

    true_min_estimate = params["true_min_estimate"]
    println("  True min estimate: $true_min_estimate")

    # Build wall-clock time records from iteration records + total times
    time_records_all = []
    for (i, mk) in enumerate(method_keys)
        record = records_all[i]
        wk = wallclock_keys[i]
        total_time = get(wc, wk, 0.0)
        if !isempty(record) && total_time > 0
            n_iters = length(record)
            time_step = n_iters > 1 ? total_time / (n_iters - 1) : 0.0
            push!(time_records_all, [(j * time_step, record[j+1][2]) for j in 0:n_iters-1])
        else
            push!(time_records_all, Tuple{Float64,Float64}[])
        end
    end

    # Subsets
    nopba_indices = [i for i in 1:length(method_names_all) if !contains(string(method_names_all[i]), "PBA")]
    bundles_indices = [i for i in 1:length(method_names_all) if !contains(string(method_names_all[i]), "SGM")]

    records_nopba = records_all[nopba_indices]
    time_records_nopba = time_records_all[nopba_indices]
    method_names_nopba = method_names_all[nopba_indices]

    records_bundles = records_all[bundles_indices]
    time_records_bundles = time_records_all[bundles_indices]
    method_names_bundles = method_names_all[bundles_indices]

    # Compute 5× shortest method cutoffs
    nonempty_records_all = [r for r in records_all if !isempty(r)]
    nonempty_records_nopba = [r for r in records_nopba if !isempty(r)]
    nonempty_records_bundles = [r for r in records_bundles if !isempty(r)]

    iter_cutoff_all = isempty(nonempty_records_all) ? 100 : 5 * minimum(length(r) for r in nonempty_records_all)
    iter_cutoff_nopba = isempty(nonempty_records_nopba) ? 100 : 5 * minimum(length(r) for r in nonempty_records_nopba)
    iter_cutoff_bundles = isempty(nonempty_records_bundles) ? 100 : 5 * minimum(length(r) for r in nonempty_records_bundles)

    nonempty_tr_all = [tr for tr in time_records_all if !isempty(tr)]
    nonempty_tr_nopba = [tr for tr in time_records_nopba if !isempty(tr)]
    nonempty_tr_bundles = [tr for tr in time_records_bundles if !isempty(tr)]

    time_cutoff_all = isempty(nonempty_tr_all) ? 1.0 : 5 * minimum(maximum(r[1] for r in tr) for tr in nonempty_tr_all)
    time_cutoff_nopba = isempty(nonempty_tr_nopba) ? 1.0 : 5 * minimum(maximum(r[1] for r in tr) for tr in nonempty_tr_nopba)
    time_cutoff_bundles = isempty(nonempty_tr_bundles) ? 1.0 : 5 * minimum(maximum(r[1] for r in tr) for tr in nonempty_tr_bundles)

    # --- Generate plots ---
    for (subfolder, rec, trec, names, iter_cut, time_cut) in [
        ("all_methods", records_all,     time_records_all,     method_names_all,     iter_cutoff_all,     time_cutoff_all),
        ("no_pba",      records_nopba,   time_records_nopba,   method_names_nopba,   iter_cutoff_nopba,   time_cutoff_nopba),
        ("bundles_only", records_bundles, time_records_bundles, method_names_bundles, iter_cutoff_bundles, time_cutoff_bundles),
    ]
        for (duration, iter_max, time_max) in [
            ("long",  nothing,   nothing),
            ("short", iter_cut,  time_cut),
        ]
            dir = joinpath(results_folder, duration, subfolder)
            isdir(dir) || mkpath(dir)
            fname = "$(subfolder)_$(duration)_spd_$(n)x$(n).pdf"

            # Oracle calls (offset iterations)
            plot_objective_gap_convergence(rec, names, true_min_estimate;
                filename=joinpath(dir, "convergence_gap_$fname"),
                offset_iterations=true, ylims_lower=atol, show_legend=false,
                max_x=iter_max)

            # Wall-clock time
            plot_objective_gap_convergence(trec, names, true_min_estimate;
                filename=joinpath(dir, "wallclock_gap_$fname"),
                wallclock=true, ylims_lower=atol, ylabel="",
                max_x=time_max)
        end
    end

    println("Plots generated for SPD $n×$n")
end

println("\nAll plots generated.")
