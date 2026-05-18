"""
Replicator Diversity Simulation on a 2D Lattice
================================================

Spatial agent-based model studying how mixing regime and environmental
heterogeneity jointly affect replicator diversification on a periodic
2D lattice.

Key features:
  - Niche-based fitness: fitness = r * exp(-β(E-θ)²) * min(1, R)
  - Global or local mixing (transport vs diffusion)
  - Two diversity metrics: functional (θ_std) and ecological (θ-binned H')
  - Configurable abiogenesis: uniform θ or fixed-θ entrants, with unique species identifiers

Hydrosphere mapping:
  - mixing_rate D ∈ [0,1]: 0 = isolated pools, 1 = well-mixed ocean
  - heterogeneity h ∈ [0,1]: 0 = uniform environment, 1 = max variation

References:
  - Eigen (1971): Error threshold and replicator dynamics
  - Takeuchi & Hogeweg (2012): Spatial structure in replicator systems
  - Czárán et al. (2002): Surface-bound replicator models
  - MacArthur & Wilson (1967): Island biogeography theory
"""

import numpy as np
from dataclasses import dataclass, asdict
import json, csv, os, time


# Number of theta bins for ecological Shannon H'
N_THETA_BINS = 20


@dataclass
class SimConfig:
    """All parameters for a single simulation run."""

    # Grid
    grid_size: int = 50

    # Biology
    base_r: float = 0.15            # Base replication rate
    base_d: float = 0.04            # Base death rate
    mutation_rate: float = 0.02     # Prob offspring gets new theta/r
    mutation_sigma_theta: float = 0.10  # Std dev of theta mutation
    mutation_sigma_r: float = 0.015     # Std dev of replication rate mutation
    tradeoff: float = 0.08         # Death cost: d_eff = base_d + tradeoff * r²

    # Niche fitness
    niche_width: float = 20.0  # β: selection strength
    # fitness = r * exp(-β*(E-θ)²) * min(1, R)
    # β=20: 0.2 from optimum → 45% fitness; 0.3 → 17%

    # Environment
    heterogeneity: float = 0.5  # h ∈ [0,1]
    pattern: str = "patches"    # uniform | gradient | patches | hotspots

    # Resources
    resource_regen: float = 0.03
    resource_cost: float = 0.12

    # Mixing (KEY VARIABLE)
    mixing_rate: float = 0.1     # D: fraction transported per tick
    mixing_mode: str = "global"  # "global" (teleport) or "local" (diffusion)

    # Abiogenesis
    spawn_rate: float = 0.0005
    abiogenesis_mode: str = "uniform"  # "uniform" or "fixed_theta"
    abiogenesis_theta: float = 0.5     # Used when abiogenesis_mode == "fixed_theta"

    # Run settings
    n_ticks: int = 500
    seed: int = 42
    n_initial: int = 20

    @property
    def n_cells(self):
        return self.grid_size ** 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


class Simulation:
    """Lattice-based replicator diversity simulation.

    Each tick:
      1. Death:        P_death = base_d + tradeoff * r²
      2. Mixing:       fraction D teleported globally or diffused locally
      3. Replication:  fitness-dependent, offspring inherit θ±mutation
      4. Resources:    empty cells regenerate
      5. Abiogenesis:  rare spontaneous replicator appearance
    """

    def __init__(self, config: SimConfig):
        self.cfg = config
        self.N = config.grid_size
        self.S = self.N * self.N  # total cells
        self.rng = np.random.default_rng(config.seed)

        # Neighbor indices (von Neumann, periodic)
        N = self.N
        self._nbrs = np.zeros((self.S, 4), dtype=np.int32)
        for i in range(self.S):
            x, y = i % N, i // N
            self._nbrs[i, 0] = ((y - 1) % N) * N + x
            self._nbrs[i, 1] = ((y + 1) % N) * N + x
            self._nbrs[i, 2] = y * N + (x - 1) % N
            self._nbrs[i, 3] = y * N + (x + 1) % N

        # State arrays
        self.grid = np.full(self.S, -1, dtype=np.int32)        # species ID (-1=empty)
        self.theta = np.zeros(self.S, dtype=np.float64)         # env optimum
        self.rep_rate = np.zeros(self.S, dtype=np.float64)      # replication rate
        self.resources = np.ones(self.S, dtype=np.float64)      # [0,1]

        # Environment map
        self.env_map = self._build_environment()

        # Species tracking
        self.next_id = 0
        self.tick = 0
        self.history = []

        # Seed population
        self._seed_population()

    def _build_environment(self):
        """Generate environmental quality map E(x,y) ∈ [0,1]."""
        N = self.N
        S = self.S
        h = self.cfg.heterogeneity
        p = self.cfg.pattern

        if p == "uniform" or h == 0:
            return np.full(S, 0.5, dtype=np.float64)

        if p == "gradient":
            env = np.zeros(S, dtype=np.float64)
            for y in range(N):
                for x in range(N):
                    env[y * N + x] = (1 - h) * 0.5 + h * (x / max(1, N - 1))
            return env

        if p == "patches":
            env = self.rng.random(S)
            # Box filter smoothing (4 passes, radius 2)
            for _ in range(4):
                tmp = np.zeros(S)
                for y in range(N):
                    for x in range(N):
                        sm, cnt = 0.0, 0
                        for dy in range(-2, 3):
                            for dx in range(-2, 3):
                                sm += env[((y + dy) % N) * N + ((x + dx) % N)]
                                cnt += 1
                        tmp[y * N + x] = sm / cnt
                env = tmp
            mn, mx = env.min(), env.max()
            return (1 - h) * 0.5 + h * (env - mn) / (mx - mn + 1e-10)

        if p == "hotspots":
            env = np.full(S, 0.2, dtype=np.float64)
            n_spots = 4 + self.rng.integers(6)
            for _ in range(n_spots):
                cx, cy = self.rng.integers(N), self.rng.integers(N)
                rad = 5 + self.rng.random() * 10
                for y in range(N):
                    for x in range(N):
                        dx = min(abs(x - cx), N - abs(x - cx))
                        dy = min(abs(y - cy), N - abs(y - cy))
                        dist = np.sqrt(dx**2 + dy**2)
                        if dist < rad:
                            env[y * N + x] += (1 - dist / rad) * h * 0.6
            return np.clip(env, 0.0, 1.0)

        return np.full(S, 0.5, dtype=np.float64)

    def _seed_population(self):
        """Place initial replicators. Each gets a unique species ID.
        Theta is drawn uniformly in [0,1] so diversity must emerge from selection/mutation."""
        n = min(self.cfg.n_initial, self.S)
        indices = self.rng.choice(self.S, size=n, replace=False)
        for idx in indices:
            self.grid[idx] = self.next_id
            self.next_id += 1
            self.theta[idx] = self.rng.random()  # uniform; no assumption of local adaptation
            self.rep_rate[idx] = self.cfg.base_r

    def step(self):
        """Execute one simulation tick."""
        S = self.S
        cfg = self.cfg

        # Pre-generate random numbers
        r_death = self.rng.random(S)
        r_mix = self.rng.random(S)
        r_rep = self.rng.random(S)
        r_mut = self.rng.random(S)
        r_abio = self.rng.random(S)

        occupied = self.grid >= 0

        # ═══ 1. DEATH ═══
        d_prob = np.where(occupied,
                          cfg.base_d + cfg.tradeoff * self.rep_rate ** 2, 0.0)
        dies = occupied & (r_death < d_prob)
        self.grid[dies] = -1
        alive = occupied & ~dies

        # ═══ 2. MIXING ═══
        if cfg.mixing_rate > 0:
            wants_move = alive & (r_mix < cfg.mixing_rate)
            movers = np.where(wants_move)[0]

            if len(movers) > 0:
                self.rng.shuffle(movers)

                if cfg.mixing_mode == "global":
                    # GLOBAL: teleport to any empty cell on the grid
                    for i in movers:
                        if self.grid[i] == -1:
                            continue  # already vacated by another swap
                        empty = np.where(self.grid == -1)[0]
                        if len(empty) == 0:
                            break
                        t = empty[self.rng.integers(len(empty))]
                        self.grid[t] = self.grid[i]
                        self.theta[t] = self.theta[i]
                        self.rep_rate[t] = self.rep_rate[i]
                        self.grid[i] = -1
                        alive[i] = False
                        alive[t] = True
                else:
                    # LOCAL: diffuse to adjacent empty cell
                    for i in movers:
                        nbrs = self._nbrs[i]
                        empty_mask = self.grid[nbrs] == -1
                        if np.any(empty_mask):
                            empty_nbrs = nbrs[empty_mask]
                            t = empty_nbrs[self.rng.integers(len(empty_nbrs))]
                            self.grid[t] = self.grid[i]
                            self.theta[t] = self.theta[i]
                            self.rep_rate[t] = self.rep_rate[i]
                            self.grid[i] = -1
                            alive[i] = False
                            alive[t] = True

        # ═══ 3. REPLICATION ═══
        niche = np.exp(-cfg.niche_width * (self.env_map - self.theta) ** 2)
        fitness = self.rep_rate * niche * np.minimum(1.0, self.resources)
        can_rep = alive & (r_rep < fitness) & (self.resources >= cfg.resource_cost)

        reproducers = np.where(can_rep)[0]
        self.rng.shuffle(reproducers)
        for i in reproducers:
            nbrs = self._nbrs[i]
            empty_mask = self.grid[nbrs] == -1
            if np.any(empty_mask):
                empty_nbrs = nbrs[empty_mask]
                t = empty_nbrs[self.rng.integers(len(empty_nbrs))]

                if r_mut[i] < cfg.mutation_rate:
                    new_theta = np.clip(
                        self.theta[i] + self.rng.normal(0, cfg.mutation_sigma_theta),
                        0.0, 1.0)
                    new_r = np.clip(
                        self.rep_rate[i] + self.rng.normal(0, cfg.mutation_sigma_r),
                        0.005, 0.95)
                    child_id = self.next_id
                    self.next_id += 1
                else:
                    new_theta = self.theta[i]
                    new_r = self.rep_rate[i]
                    child_id = self.grid[i]

                self.grid[t] = child_id
                self.theta[t] = new_theta
                self.rep_rate[t] = new_r
                self.resources[i] -= cfg.resource_cost

        # ═══ 4. RESOURCE REGENERATION ═══
        empty = self.grid == -1
        self.resources[empty] = np.minimum(
            1.0, self.resources[empty] + cfg.resource_regen)

        # ═══ 5. ABIOGENESIS ═══
        new_life = empty & (r_abio < cfg.spawn_rate) & (self.resources >= cfg.resource_cost)
        abio_cells = np.where(new_life)[0]
        for cell in abio_cells:
            self.grid[cell] = self.next_id
            self.next_id += 1
            if cfg.abiogenesis_mode == "fixed_theta":
                self.theta[cell] = float(np.clip(cfg.abiogenesis_theta, 0.0, 1.0))
            else:
                self.theta[cell] = self.rng.random()  # neutral default: diversity not assumed
            self.rep_rate[cell] = cfg.base_r
            self.resources[cell] -= cfg.resource_cost

        self.tick += 1
        self._record_metrics()

    def _record_metrics(self):
        """Compute diversity metrics for this tick.

        Two Shannon H' variants are tracked:
          - shannon_eco:  θ-binned ecological diversity (primary metric)
          - shannon_lin:  lineage-based diversity (for comparison / diagnostics)
        """
        occupied = self.grid >= 0
        pop = int(np.sum(occupied))

        if pop == 0:
            self.history.append({
                'tick': self.tick,
                'population': 0,
                'occupancy': 0.0,
                'n_lineages': 0,
                'shannon_eco': 0.0,
                'shannon_lin': 0.0,
                'mean_theta': 0.0,
                'theta_std': 0.0,
                'theta_range': 0.0,
                'mean_r': 0.0,
                'mean_fitness': 0.0,
                'mean_resource': float(np.mean(self.resources)),
            })
            return

        th = self.theta[occupied]

        # ── Ecological Shannon H' (θ-binned) ──
        bin_idx = np.clip((th * N_THETA_BINS).astype(int), 0, N_THETA_BINS - 1)
        bins = np.bincount(bin_idx, minlength=N_THETA_BINS).astype(float)
        props_eco = bins[bins > 0] / pop
        shannon_eco = float(-np.sum(props_eco * np.log(props_eco)))

        # ── Lineage Shannon H' (species-ID based) ──
        ids = self.grid[occupied]
        _, counts = np.unique(ids, return_counts=True)
        props_lin = counts / pop
        shannon_lin = float(-np.sum(props_lin * np.log(props_lin + 1e-30)))

        # ── Trait statistics ──
        rs = self.rep_rate[occupied]
        niche = np.exp(-self.cfg.niche_width * (self.env_map[occupied] - th) ** 2)
        mean_fit = float(np.mean(rs * niche * np.minimum(1.0, self.resources[occupied])))

        self.history.append({
            'tick': self.tick,
            'population': pop,
            'occupancy': pop / self.S,
            'n_lineages': int(len(counts)),
            'shannon_eco': shannon_eco,
            'shannon_lin': shannon_lin,
            'mean_theta': float(np.mean(th)),
            'theta_std': float(np.std(th)),
            'theta_range': float(np.ptp(th)) if pop > 1 else 0.0,
            'mean_r': float(np.mean(rs)),
            'mean_fitness': mean_fit,
            'mean_resource': float(np.mean(self.resources)),
        })

    def ensure_metrics_recorded(self):
        """Record metrics for the current state if history is missing/stale."""
        if not self.history or self.history[-1]['tick'] != self.tick:
            self._record_metrics()

    def get_current_metrics(self):
        """Return metrics for the current state without advancing the simulation."""
        self.ensure_metrics_recorded()
        return dict(self.history[-1])

    def get_history_series(self, max_points=300):
        """Return compact time-series slices for frontend charting."""
        self.ensure_metrics_recorded()
        recent = self.history[-max_points:]
        return {
            'ticks': [int(row['tick']) for row in recent],
            'theta_std': [float(row['theta_std']) for row in recent],
            'shannon_eco': [float(row['shannon_eco']) for row in recent],
            'population': [int(row['population']) for row in recent],
        }

    def run(self, verbose=False):
        """Run the full simulation."""
        t0 = time.time()
        for i in range(self.cfg.n_ticks):
            self.step()
            if verbose and (i + 1) % 100 == 0:
                h = self.history[-1]
                print(f"  tick {h['tick']:4d} | pop={h['population']:5d} "
                      f"H'_eco={h['shannon_eco']:.2f} θ_std={h['theta_std']:.3f} "
                      f"[{time.time()-t0:.1f}s]")
        return self.history

    def get_equilibrium_metrics(self, last_n=100):
        """Average metrics over last N ticks for equilibrium estimates."""
        if not self.history:
            return {}
        n = min(last_n, len(self.history))
        recent = self.history[-n:]
        result = {}
        keys = [k for k in recent[0] if k != 'tick']
        for key in keys:
            vals = [r[key] for r in recent]
            result[key + '_mean'] = float(np.mean(vals))
            result[key + '_std'] = float(np.std(vals))
        return result

    def save_history(self, path):
        """Save time series to CSV."""
        if not self.history:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)

    def get_grid_snapshot(self):
        """Return current grid state for visualization."""
        return {
            'grid': self.grid.copy(),
            'theta': self.theta.copy(),
            'rep_rate': self.rep_rate.copy(),
            'resources': self.resources.copy(),
            'env_map': self.env_map.copy(),
            'N': self.N,
        }


# ── Convenience functions ──

def run_single(config: SimConfig, verbose=False):
    """Run one simulation, return (history, equilibrium_metrics, snapshot)."""
    sim = Simulation(config)
    history = sim.run(verbose=verbose)
    eq = sim.get_equilibrium_metrics()
    snap = sim.get_grid_snapshot()
    return history, eq, snap


def run_sweep_point(mixing, heterogeneity, seeds, base_config=None,
                    grid_size=40, n_ticks=600, pattern="patches"):
    """Run multiple seeds for one (D, h) point.

    Returns dict with proper across-seed statistics:
      {metric}_mean  — mean across seeds
      {metric}_std   — std across seeds
    """
    if base_config is None:
        base_config = SimConfig()

    # Collect equilibrium θ_std and H'_eco from each seed
    seed_metrics = []
    for seed in seeds:
        cfg = SimConfig(
            grid_size=grid_size,
            base_r=base_config.base_r,
            base_d=base_config.base_d,
            mutation_rate=base_config.mutation_rate,
            mutation_sigma_theta=base_config.mutation_sigma_theta,
            mutation_sigma_r=base_config.mutation_sigma_r,
            tradeoff=base_config.tradeoff,
            niche_width=base_config.niche_width,
            heterogeneity=heterogeneity,
            pattern=pattern,
            resource_regen=base_config.resource_regen,
            resource_cost=base_config.resource_cost,
            mixing_rate=mixing,
            mixing_mode=base_config.mixing_mode,
            spawn_rate=base_config.spawn_rate,
            abiogenesis_mode=base_config.abiogenesis_mode,
            abiogenesis_theta=base_config.abiogenesis_theta,
            n_ticks=n_ticks,
            seed=seed,
            n_initial=base_config.n_initial,
        )
        sim = Simulation(cfg)
        sim.run()
        eq = sim.get_equilibrium_metrics(last_n=100)
        seed_metrics.append(eq)

    # Compute across-seed statistics properly
    avg = {'mixing_rate': mixing, 'heterogeneity': heterogeneity}
    if seed_metrics:
        # For each metric, take the _mean from each seed (the within-run
        # temporal average), then compute mean and std across seeds.
        for key in seed_metrics[0]:
            if key.endswith('_mean'):
                base = key[:-5]
                vals = [sm[key] for sm in seed_metrics]
                avg[base + '_mean'] = float(np.mean(vals))
                avg[base + '_std'] = float(np.std(vals))

    return avg
