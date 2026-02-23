# Riemannian Proximal Bundle Method

Official implementation of **"Convergence Rates for Riemannian Proximal Bundle Methods"**.

## Requirements

- **Julia:** Julia 1.9+
- **Packages:** All dependencies are listed in `Project.toml`. Install them from the repository root:
  ```julia
  using Pkg; Pkg.instantiate()
  ```
- **Jupyter (optional):** To run the interactive notebooks, install IJulia:
  ```julia
  using Pkg; Pkg.add("IJulia")
  ```

## Repository Structure

```
├── src/
│   └── RiemannianProximalBundle.jl   # Core RPB solver
├── experiments/
│   ├── median_SPD.ipynb              # Interactive notebook: SPD median experiment
│   └── denoising_hyperbolic.ipynb    # Interactive notebook: TV denoising experiment
├── data/
│   ├── RCBM Median <N> Points/       # CSV output from experiment_median_SPD.jl
│   └── Denoising TV Hyperbolic/      # CSV output from experiment_denoising_hyperbolic.jl
├── plots/
│   ├── RCBM Median <N> Points/       # PDF/SVG output from plot_median_SPD.jl
│   └── Denoising TV Hyperbolic/      # PDF output from plot_denoising_hyperbolic.jl
├── experiment_median_SPD.jl          # Run SPD median experiment, saves CSVs to data/
├── experiment_denoising_hyperbolic.jl# Run TV denoising experiment, saves CSVs to data/
├── plot_median_SPD.jl                # Load CSVs from data/ and produce plots in plots/
└── plot_denoising_hyperbolic.jl      # Load CSVs from data/ and produce plots in plots/
```

Experiments and plotting are intentionally separated: run an experiment once (potentially expensive), then re-run the plotting script freely.

## The RPB Solver (`src/RiemannianProximalBundle.jl`)

The core algorithm is the `RProximalBundle` struct. Construct it and call `run!`:

```julia
include("src/RiemannianProximalBundle.jl")

solver = RProximalBundle(
    manifold, retraction_map, transport_map,
    objective_function, subgradient_function,
    initial_point, initial_objective, initial_subgradient;
    # keyword arguments below
)
run!(solver)
```

**Positional arguments:**

| Argument | Description |
|----------|-------------|
| `manifold` | The Riemannian manifold (any object accepted by the retraction/transport maps) |
| `retraction_map` | `(p, v) -> q`: retract point `p` along tangent vector `v` |
| `transport_map` | `(p, q, v) -> w`: transport tangent vector `v` from `p` to `q` |
| `objective_function` | `(p) -> f(p)`: returns a scalar |
| `subgradient_function` | `(p) -> g`: returns a tangent vector at `p` |
| `initial_point` | Starting iterate `x_0` |
| `initial_objective` | `f(x_0)` |
| `initial_subgradient` | `g_0 ∈ ∂f(x_0)` |

**Keyword arguments:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `proximal_parameter` | `0.02` | Initial proximal parameter $\rho_0$ |
| `trust_parameter` | `0.1` | Descent condition threshold $\beta$ |
| `max_iter` | `5000` | Maximum iterations |
| `tolerance` | `1e-12` | Stopping tolerance (objective gap or subgradient norm) |
| `know_minimizer` | `true` | If `true`, stops when `f(x) - f* < tolerance`; if `false`, uses a different criterion |
| `true_min_obj` | `0.0` | Known or estimated minimum value $f^*$; only used when `know_minimizer=true` |
| `relative_error` | `true` | Track relative gap `(f(x) - f*) / (f(x_0) - f*)` instead of absolute |
| `retraction_error` | `0.0` | Retraction error constant (for first-order retractions) |
| `transport_error` | `0.0` | Transport error constant (for approximate transports) |
| `sectional_curvature` | `-1.0` | Lower bound on sectional curvature of the manifold |
| `back_tracking_factor` | `2.0` | Factor by which $\rho$ is multiplied during proximal backtracking |
| `max_rho` | `1e8` | Upper bound on proximal parameter |
| `debugging` | `false` | Enable extra descent-step ratio logging |
| `memory_lite` | `false` | Skip storing iterate histories (proximal centers, candidate directions, etc.); only objective, proximal parameter, and step-index arrays are kept. Use for long runs to avoid memory issues. |

**Accessing results after `run!`:**

| Field | Always stored | Description |
|-------|:---:|-------------|
| `solver.raw_objective_history` | yes | `f(x_k)` at every iteration |
| `solver.objective_history` | yes | Objective gap `f(x_k) - f*` at every iteration |
| `solver.proximal_parameter_history` | yes | $\rho_k$ at every iteration |
| `solver.indices_of_descent_steps` | yes | Iteration indices where a descent step occurred |
| `solver.indices_of_null_steps` | yes | Iteration indices where a null step occurred |
| `solver.iteration` | yes | Vector `[0, 1, 2, ..., T]` |
| `solver.current_proximal_center` | yes | Final iterate |
| `solver.proximal_center_history` | no* | All proximal centers $x_k$ |
| `solver.objective_increase_flags` | no* | `true` at iterations where $f(x_{k+1}) > f(x_k)$ |

\* Not stored when `memory_lite=true`.

## Experiments

### Riemannian Median on SPD Matrices

Computes the Fréchet median of $N$ random Symmetric Positive Definite matrices on $\mathcal{S}_{++}^n$ and compares RPB (exponential and first-order retraction variants) against RCBM, PBA, and SGM.

The experiment runs in two phases: Phase 1 estimates the true minimum by running all methods for extended iterations; Phase 2 runs each method until the objective gap falls below `PHASE2_GAP_TOL`.

Key parameters (configurable at the top of `experiment_median_SPD.jl` or in `experiments/median_SPD.ipynb`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `N` | `500` | Number of data points |
| `spd_dims` | `[3, 5, 15, 30, 55]` | SPD matrix dimensions to benchmark |
| `atol` | `1e-12` | Convergence tolerance |
| `maxiter` | `1000` | Maximum iterations per method |
| `seed_argument` | `57` | RNG seed |
| `PHASE1_MULTIPLIER` | `2` | Phase 1 runs for `PHASE1_MULTIPLIER × maxiter` iterations to estimate the true minimum |
| `PHASE2_GAP_TOL` | `1e-12` | Phase 2 stops when objective gap drops below this |

Run the experiment (saves CSVs to `data/RCBM Median <N> Points/`):
```bash
julia experiment_median_SPD.jl
```

Generate plots from saved CSVs (reads `N` from the top of the file; adjust to match the experiment):
```bash
julia plot_median_SPD.jl
```

Or use the interactive notebook: `experiments/median_SPD.ipynb`

---

### TV Denoising on the Hyperbolic Manifold

Solves a total variation denoising problem on the power manifold $(\mathbb{H}^2)^n$ and compares RPB against RCBM, PBA, and SGM.

A synthetic piecewise-geodesic signal is generated on $\mathbb{H}^2$, corrupted with Riemannian Gaussian noise, and recovered by minimizing:
$$f(p) = \frac{1}{n}\left(\frac{1}{2} d^2(p, \tilde{p}) + \alpha \cdot \mathrm{TV}(p)\right)$$

Phase 1 estimates the true minimum via CPPA; Phase 2 runs each method until the objective gap is below `PHASE2_GAP_TOL`.

Key parameters (configurable at the top of `experiment_denoising_hyperbolic.jl` or in `experiments/denoising_hyperbolic.ipynb`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_points` | `496` | Signal length |
| `sigma` | `0.3` | Noise level ($\sigma$ of Riemannian Gaussian) |
| `alpha` | `0.5` | TV regularization strength |
| `atol` | `1e-8` | Inner convergence tolerance |
| `max_iter` | `100,000` | Maximum iterations |
| `PHASE1_MULTIPLIER` | `1` | Phase 1 multiplier for true-minimum estimation |
| `PHASE2_GAP_TOL` | `1e-8` | Phase 2 stops when objective gap drops below this |

Run the experiment (saves CSVs to `data/Denoising TV Hyperbolic/`):
```bash
julia experiment_denoising_hyperbolic.jl
```

Generate plots from saved CSVs:
```bash
julia plot_denoising_hyperbolic.jl
```

Or use the interactive notebook: `experiments/denoising_hyperbolic.ipynb`

#### QP Solver Diagnostics (debugging)

The experiment overrides the internal QP subsolvers of RCBM and PBA (via RipQP) to enable optional diagnostics. This was used to diagnose a stalling issue caused by the newest bundle element being pruned at every iteration.

To enable, set at the top of `experiment_denoising_hyperbolic.jl`:
```julia
const QP_DIAGNOSTICS = true
```

When enabled, after each RCBM and PBA run the following summary is printed:
- QP solve status breakdown (first-order, acceptable, infeasible, etc.)
- Primal/dual feasibility statistics
- Gram matrix condition number statistics
- Solution health (NaN/Inf/negative weights, simplex constraint violation)
- Bundle size statistics
- **Newest-element pruning rate** — the key diagnostic: if `λ[end] ≤ eps()` at most iterations, the solver is discarding the most recently added cut and may stall

## References

[1] Insert Later