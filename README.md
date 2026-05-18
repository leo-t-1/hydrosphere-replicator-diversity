# Hydrosphere Replicator Diversity Simulation

> **Companion documents:**
> [SCIENTIFIC_RATIONALE.md](SCIENTIFIC_RATIONALE.md) explains *why* the
> question matters and how it differs from existing ecology results.
> [paper.pdf](paper.pdf) is the writeup.

This repository studies one question:

**If early replicators can adapt to local conditions, does connectivity help diversity by spreading good variants, or hurt diversity by homogenising niches?**

The answer depends critically on the transport model:

- **Under global teleportation**: Diversity is highest at D = 0 in every heterogeneity slice. At `h = 0.8`, `theta_std` drops from `0.168` to `0.063` (62.5% decrease, Cohen's d > 5, p < 0.001).
- **Under local diffusion** (physically realistic): The suppression effect nearly disappears (~1.1x ratio vs ~2.7x), consistent with Czárán et al. (2015).
- In the named hydrosphere scenarios, **Shallow Ponds** are about **3.5x** higher than **Deep Ocean** in functional diversity.
- A **colonization-limited regime** (local diffusion, low abiogenesis) tests whether the intermediate dispersal optimum from metacommunity ecology has a prebiotic analogue.

This is a **model-based ecological result**, not direct chemistry.

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib reportlab scipy

# Full experiment: scenarios + sweep + sensitivity + figures
python run_experiment.py

# Quick test
python run_experiment.py --quick

# Paper PDF from saved results
python generate_paper.py

# Standalone JavaScript simulator
open simulator.html

# True Python-backed interactive frontend
python3 backend_simulator_server.py
# then open http://127.0.0.1:8765/
```

## What The Model Contains

The world is a periodic 2D lattice. Each cell is either:

- empty, or
- occupied by one replicator.

There is no multi-occupancy. One cell holds at most one replicator.

### What Is A Replicator?

A **replicator** is one occupied lattice site with heritable traits. In this model, each occupied cell stores:

- `theta`: its niche optimum, on `[0, 1]`
- `rep_rate`: its intrinsic replication rate `r`
- `species ID`: a lineage label used for ancestry tracking

So a replicator is not just an abstract population count. It is an individual occupant of one cell with state.

### What Is A Species Here?

`species ID` in this code is really a **lineage identifier**, not a biological species in the modern taxonomic sense.

The logic is:

- Each founder gets a unique ID.
- If a replicator reproduces **without mutation**, the child keeps the parent’s ID.
- If reproduction happens **with mutation**, the child gets a new ID.
- If abiogenesis creates a new replicator, it also gets a new ID.

So:

- “species” in this model means **descent lineage**
- it does **not** mean a stable phenotype class
- two different IDs can still end up with similar `theta` and `rep_rate`

That is why the project does **not** use lineage diversity as the main scientific result. Lineages can split without creating genuinely different ecological strategies.

## Core Variables

### `theta`

`theta` (`θ`) is the replicator’s **preferred environment**.

- `theta = 0.1` means it does best in low-`E` environments
- `theta = 0.8` means it does best in high-`E` environments

It is the main ecological trait in the model.

### `beta`

`beta` (`β`) is the **selection strength** or **niche width** parameter.

It appears in the niche-matching term:

```text
exp(-beta * (E - theta)^2)
```

Interpretation:

- low `beta` = broad niche, mismatch hurts less
- high `beta` = narrow niche, mismatch hurts more

Important: `beta` is a **global run parameter**, not an individual or species-specific trait in the default model.

### `rep_rate`

`rep_rate` (`r`) is the intrinsic replication tendency of one replicator.

This **does** vary across individuals because it can mutate. It is inherited by offspring, with possible mutation.

### `E`

`E` is the local environment value of a cell, also on `[0, 1]`.

It comes from the chosen environment map:

- `uniform`
- `gradient`
- `patches`
- `hotspots`

### `R`

`R` is the local resource level of a cell. Replication is resource-limited.

### `D`

`D` is the mixing rate. It is the fraction of living replicators that attempt transport each tick.

In the experiment and the Python-backed frontend, mixing is fixed to **global teleport mixing**.

### `h`

`h` is heterogeneity. It controls how varied the environment map is.

- `h = 0` means nearly uniform conditions
- `h = 1` means maximally varied conditions in the chosen pattern

## Fitness Mathematics

The replication fitness term is:

```text
fitness = r * exp(-beta * (E - theta)^2) * min(1, R)
```

This means:

- high `r` helps
- good environmental matching (`E ≈ theta`) helps
- high local resources help

If `E = theta`, then the niche penalty is `exp(0) = 1`.

If `E` and `theta` differ, fitness falls off like a Gaussian.

With the default `beta = 20`:

- mismatch `|E - theta| = 0.2` gives `exp(-20 * 0.04) ≈ 0.45`
- mismatch `|E - theta| = 0.3` gives `exp(-20 * 0.09) ≈ 0.17`

## What Actually Differs Between Replicators?

Within one simulation run:

- `theta` differs across replicators
- `rep_rate` differs across replicators
- lineage ID differs across replicators

Within one simulation run, these are **global** and shared:

- `beta`
- base death rate
- tradeoff coefficient
- mutation rate
- mutation step sizes
- resource cost and regeneration rate
- mixing rate `D`
- heterogeneity `h`
- environment pattern

So the model is not giving each species its own `beta`. The main heritable phenotype is:

- niche optimum `theta`
- replication rate `r`

## Life Cycle Per Tick

The tick order is fixed:

1. Death
2. Mixing
3. Replication
4. Resource regeneration
5. Abiogenesis

### 1. Death

Occupied cells die with probability:

```text
d_eff = base_d + tradeoff * r^2
```

So faster replicators pay a death penalty.

### 2. Mixing

With probability `D`, a living replicator attempts transport.

In the main experiment:

- transport is **global**
- the replicator teleports to a **random empty cell anywhere on the grid**

This moves the whole replicator:

- same lineage ID
- same `theta`
- same `rep_rate`

### 3. Replication

This is the part most people want to know exactly.

A living replicator can attempt reproduction only if:

- it survived death,
- a uniform random number is less than its fitness,
- it has enough local resources to pay the replication cost.

Then it checks its **four von Neumann neighbors**:

- up
- down
- left
- right

Replication happens **only if at least one of those neighboring cells is empty**.

So yes: **offspring are placed only into an empty adjacent cell**. A replicator cannot directly overwrite an occupied neighbor, and it cannot place offspring at arbitrary distance.

The replication logic is:

```text
if alive and random < fitness and resources >= cost:
    choose one empty neighboring cell
    if mutation:
        child gets mutated theta
        child gets mutated rep_rate
        child gets new lineage ID
    else:
        child inherits parent's theta
        child inherits parent's rep_rate
        child keeps parent's lineage ID
```

Mathematically, if mutation occurs:

```text
theta_child = clip(theta_parent + Normal(0, sigma_theta), 0, 1)
r_child     = clip(r_parent     + Normal(0, sigma_r), 0.005, 0.95)
```

The parent then pays the resource cost.

### 4. Resource Regeneration

Only empty cells regenerate resources:

```text
R_empty <- min(1, R_empty + resource_regen)
```

### 5. Abiogenesis

Rare spontaneous entrants can appear in empty cells:

- only in empty cells
- only if a random draw is below `spawn_rate`
- only if the cell has enough resources

These entrants get:

- a new lineage ID
- `theta ~ Uniform(0, 1)` by default
- `rep_rate = base_r`

Abiogenesis is therefore **uniform**, not environment-matched by construction.

There is now an optional alternative:

- `abiogenesis_mode = "fixed_theta"`
- `abiogenesis_theta = some value in [0, 1]`

In that mode, every abiogenesis entrant starts with the same preferred environment.

## Diversity Metrics

The project uses two main diversity metrics and one diagnostic metric.

### 1. Functional Diversity: `theta_std`

This is the standard deviation of all occupied cells’ `theta` values:

```text
theta_std = sqrt((1/N) * sum_i (theta_i - mean(theta))^2)
```

where `N` is the number of occupied cells.

Interpretation:

- low `theta_std` = most replicators have similar ecological preferences
- high `theta_std` = the population spans a wider range of niche strategies

This is called **functional diversity** because it measures spread in ecological function or strategy, not just ancestry labels.

### 2. Ecological Diversity: `H'_eco`

This is Shannon entropy computed on **20 bins of `theta`**, not on lineage IDs.

Procedure:

1. Divide `[0, 1]` into 20 `theta` bins.
2. Count how many occupied cells fall in each bin.
3. Convert counts to proportions `p_k`.
4. Compute:

```text
H' = -sum_k p_k ln(p_k)
```

This is standard **Shannon entropy**.

Interpretation:

- `H' = 0` if all occupied cells are in one bin
- `H'` gets larger if more bins are occupied
- `H'` gets larger if occupancy is more even across bins

Because this model uses natural log, the units are **nats**.

With 20 equally occupied bins, the maximum possible value is:

```text
H'_max = ln(20) ≈ 2.996
```

This is called **ecological diversity** here because it measures how much of niche space is occupied and how evenly it is occupied.

### 3. Lineage Diversity: `shannon_lin`

The code also computes Shannon entropy over lineage IDs:

```text
H'_lin = -sum_j q_j ln(q_j)
```

where `q_j` is the fraction of occupied cells belonging to lineage `j`.

This is useful diagnostically, but it is **not** the main scientific result, because lineage splitting can happen without large ecological change.

## Functional Diversity vs Ecological Diversity

They are related, but not identical.

- `theta_std` cares about **spread**
- `H'_eco` cares about **occupancy and evenness across niche bins**

Examples:

- If the population is concentrated near one `theta`, both are low.
- If the population occupies many bins fairly evenly, both are high.
- If the population occupies two far-apart extremes only, `theta_std` can be high while `H'_eco` is only moderate.

That is why the project tracks both.

## Hydrosphere Scenarios

| Scenario | `h` | `D` | Pattern | Interpretation |
|---|---:|---:|---|---|
| Deep Ocean | 0.10 | 0.50 | `uniform` | low heterogeneity, high mixing |
| Shallow Ponds | 0.80 | 0.02 | `patches` | high heterogeneity, very low mixing |
| Tidal Zone | 0.70 | 0.15 | `patches` | high heterogeneity, moderate mixing |
| Hydrothermal Vents | 0.60 | 0.08 | `hotspots` | moderate heterogeneity, low-moderate mixing |

## What The Current Results Claim

The current defensible claim is:

- heterogeneity increases diversity,
- mixing suppresses diversity,
- heterogeneity helps most when mixing is weak,
- in this transport regime, the optimum remains at the **lowest sampled mixing value**.

This is more specific than “heterogeneity matters.” The main result is:

**Global transport suppresses local specialisation faster than it spreads useful variants.**

## Frontends

There are two frontends.

### `simulator.html`

This is a standalone JavaScript implementation of the same rule set.

- good for intuition
- does not call Python
- does not reproduce NumPy’s exact random stream

### `simulator_backend.html` + `backend_simulator_server.py`

This is the true Python-backed frontend.

- the browser does not simulate locally
- Python owns the session state
- the page only renders lattice snapshots, metrics, and history returned by the backend

The live run controls include:

- `BATCH STEPS`: how many Python ticks happen per live update
- `LIVE PACE`: delay in milliseconds between live updates
- `ABIOGENESIS MODE`: switch between neutral uniform entrants and fixed-theta entrants
- `ABIO THETA`: choose the fixed starting `theta` used when fixed-theta abiogenesis is enabled

## Outputs

Important generated files:

- [simulation.py](simulation.py)
- [run_experiment.py](run_experiment.py)
- [experiment_summary.py](experiment_summary.py)
- [results/experiment_summary.json](results/experiment_summary.json)
- [generate_paper.py](generate_paper.py)
- [paper.pdf](paper.pdf)
- [make_poster_figures_reviewed.py](make_poster_figures_reviewed.py)
- [poster_figures_reviewed_v1/](poster_figures_reviewed_v1/)

## Bottom Line

If you want the shortest possible interpretation:

- a replicator is one occupied cell with heritable `theta` and `rep_rate`
- a species is really a lineage ID
- replication only places offspring into an **empty neighboring cell**
- `theta_std` measures trait spread
- `H'` measures niche occupancy/evenness
- `beta` controls how harsh mismatch is
- low mixing + high heterogeneity gives the highest diversity in this model

## License

- Code: MIT (see [LICENSE](LICENSE)).
- Paper PDF, figures, and SCIENTIFIC_RATIONALE.md: CC BY 4.0.
