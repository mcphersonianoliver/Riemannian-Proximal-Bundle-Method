# %% Experiment 1: PSD Matrices
import autograd.numpy as anp
from pymanopt.manifolds import SymmetricPositiveDefinite
from src.RiemannianProximalBundle import RProximalBundle

# %% Setting up functions
# Riemannian median cost function
def cost_set_up(point, data):
    # riemannian median cost function given points

    return sum([manifold.dist(point, x) for x in data]) / len(data)

def cost(point):
    # compute the cost function
    return cost_set_up(point, data)

# Riemannian subgradient operator
def subgradient_set_up(point, data):
    # compute the Riemannian subgradient of the cost function
    grad = anp.zeros_like(point)
    for x in data:
        log_point = manifold.log(point, x)
        grad += - (log_point)/ manifold.norm(point, log_point)
    return grad / len(data)

def subgradient(point):
    # compute the subgradient
    return subgradient_set_up(point, data)
# %%
# set the manifold a priori
dim = 10
manifold = SymmetricPositiveDefinite(dim)

# Generate n truly random points for median computation
n = 200  # Number of data points
data = []
for _ in range(n):
    random_point = manifold.random_point()
    data.append(random_point)

# Note: Since we're using truly random points, there's no known "true median"
# The algorithm will find the Riemannian median of these random points

# %% Initialize points and set up the optimizer
# First, let's sample some random points to understand the typical objective range
print("Sampling random points to understand objective value distribution...")
sample_objectives = []
for i in range(20):
    sample_point = manifold.random_point()
    sample_obj = cost(sample_point)
    sample_objectives.append(sample_obj)

min_obj = min(sample_objectives)
max_obj = max(sample_objectives)
avg_obj = sum(sample_objectives) / len(sample_objectives)

print(f"Sample statistics from 20 random points:")
print(f"  Min objective: {min_obj:.4f}")
print(f"  Max objective: {max_obj:.4f}")
print(f"  Average objective: {avg_obj:.4f}")

# Adjust target based on observed range
if max_obj < 5.0:
    target_objective = max_obj * 1.2  # Target 20% above the max we've seen
    print(f"Adjusting target to {target_objective:.4f} (20% above observed max)")
else:
    target_objective = 5.0

# Keep drawing random points until initial objective > target or max attempts reached
max_attempts = 300
attempt = 0
initial_point = None
initial_objective = 0
all_objectives = []

print(f"\nSearching for initial point with objective > {target_objective:.4f}...")

while attempt < max_attempts and initial_objective <= target_objective:
    candidate_point = manifold.random_point()
    candidate_objective = cost(candidate_point)
    all_objectives.append(candidate_objective)

    if candidate_objective > initial_objective:
        initial_point = candidate_point
        initial_objective = candidate_objective

    attempt += 1

    # Print progress every 50 attempts
    if attempt % 50 == 0:
        recent_max = max(all_objectives[-50:]) if len(all_objectives) >= 50 else max(all_objectives)
        print(f"Attempt {attempt}: Best overall = {initial_objective:.4f}, Recent max = {recent_max:.4f}")

initial_subgradient = subgradient(initial_point)

print(f"\nFinal results:")
print(f"Initial objective value: {initial_objective:.4f} (after {attempt} attempts)")
print(f"Range of all {attempt} attempts: [{min(all_objectives):.4f}, {max(all_objectives):.4f}]")
if initial_objective > target_objective:
    print(f"✓ Successfully found initial point with objective > {target_objective:.4f}")
else:
    print(f"⚠ Reached maximum attempts without finding objective > {target_objective:.4f}")
    print(f"  Consider using the best point found: {initial_objective:.4f}")
true_objective = None  # No known true minimum for random points


# Set up the optimizer
optimizer = RProximalBundle(
    manifold=manifold,
    retraction_map = manifold.exp,
    transport_map = manifold.transport,
    objective_function=cost,
    subgradient = subgradient,
    true_min_obj=0,  # Use 0 as base line
    initial_point=initial_point,
    initial_objective=initial_objective,
    initial_subgradient=initial_subgradient,
    adaptive_proximal = True,
    trust_parameter=0.2,
    transport_error =1,
    retraction_error=1,
    know_minimizer=False,  # We don't know the true minimizer for random points
)
# %%

# %% Phase 1: Estimate true minimum with long run
print("\n" + "="*70)
print("PHASE 1: ESTIMATING TRUE MINIMUM (300 iterations)")
print("="*70 + "\n")

# Run optimizer for 300 iterations to estimate true minimum
optimizer_phase1 = RProximalBundle(
    manifold=manifold,
    retraction_map=manifold.exp,
    transport_map=manifold.transport,
    objective_function=cost,
    subgradient=subgradient,
    true_min_obj=0,  # Use 0 as baseline for first run
    initial_point=initial_point,
    initial_objective=initial_objective,
    initial_subgradient=initial_subgradient,
    adaptive_proximal=True,
    trust_parameter=0.2,
    transport_error=1,
    retraction_error=1,
    know_minimizer=False,
    max_iter=300  # Long run to estimate minimum
)

optimizer_phase1.run()

# Get the estimated minimum
estimated_minimum = optimizer_phase1.raw_objective_history[-1]
best_point_found = optimizer_phase1.current_proximal_center

print(f"\nPhase 1 Results:")
print(f"Estimated minimum objective: {estimated_minimum:.6f}")
print(f"Initial objective was: {initial_objective:.6f}")
print(f"Improvement: {initial_objective - estimated_minimum:.6f}")

# %%
print("\nPlotting Phase 1 convergence (objective values):")
optimizer_phase1.plot_objective_versus_iter()

# %% Phase 2: Rerun with different initialization using estimated minimum
print("\n" + "="*70)
print("PHASE 2: CONVERGENCE ANALYSIS WITH ESTIMATED MINIMUM")
print("="*70 + "\n")

# Find a new initial point with at least as good objective as the first run
print("Finding new initial point with objective >= first run...")
max_attempts_phase2 = 500
attempt = 0
new_initial_point = None
new_initial_objective = 0

while attempt < max_attempts_phase2 and new_initial_objective < initial_objective:
    candidate_point = manifold.random_point()
    candidate_objective = cost(candidate_point)

    if candidate_objective >= initial_objective:
        new_initial_point = candidate_point
        new_initial_objective = candidate_objective
        break
    elif candidate_objective > new_initial_objective:
        new_initial_point = candidate_point
        new_initial_objective = candidate_objective

    attempt += 1

    if attempt % 100 == 0:
        print(f"Attempt {attempt}: Best so far = {new_initial_objective:.4f}, Target >= {initial_objective:.4f}")

if new_initial_objective >= initial_objective:
    print(f"✓ Found suitable initial point: {new_initial_objective:.6f} (>= {initial_objective:.6f})")
else:
    print(f"⚠ Using best point found: {new_initial_objective:.6f} (target was >= {initial_objective:.6f})")

new_initial_subgradient = subgradient(new_initial_point)

# Set up second optimizer with estimated minimum as true_min_obj
optimizer_phase2 = RProximalBundle(
    manifold=manifold,
    retraction_map=manifold.exp,
    transport_map=manifold.transport,
    objective_function=cost,
    subgradient=subgradient,
    true_min_obj=estimated_minimum,  # Use estimated minimum from Phase 1
    initial_point=new_initial_point,
    initial_objective=new_initial_objective,
    initial_subgradient=new_initial_subgradient,
    adaptive_proximal=True,
    trust_parameter=0.2,
    transport_error=1,
    retraction_error=1,
    know_minimizer=True,  # We have an estimate of the minimizer
    max_iter=200  # Shorter run for analysis
)

print(f"\nRunning Phase 2 with estimated minimum {estimated_minimum:.6f}")
print(f"Initial gap: {new_initial_objective - estimated_minimum:.6f}")

optimizer_phase2.run()

# %%
print("\nPlotting Phase 2 convergence (objective gaps with estimated minimum):")
optimizer_phase2.plot_objective_versus_iter()

# %%
print("\nPlotting Phase 2 log-log convergence to check for linear convergence:")
optimizer_phase2.plot_objective_versus_iter(log_log=True)

# %%
print("\n" + "="*70)
print("COMPARISON SUMMARY")
print("="*70)
print(f"Phase 1 (objective values):")
print(f"  Initial: {initial_objective:.6f}")
print(f"  Final:   {estimated_minimum:.6f}")
print(f"  Total improvement: {initial_objective - estimated_minimum:.6f}")
print(f"\nPhase 2 (objective gaps):")
print(f"  Initial gap: {new_initial_objective - estimated_minimum:.6f}")
print(f"  Final gap:   {optimizer_phase2.objective_history[-1]:.6f}")
print(f"  Gap reduction: {(new_initial_objective - estimated_minimum) - optimizer_phase2.objective_history[-1]:.6f}")
print("="*70)
# %%
