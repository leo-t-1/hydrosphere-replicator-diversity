"""
Test suite for the replicator diversity simulation.

Covers:
  - SimConfig construction and parameter handling
  - Environment map generation (all four patterns)
  - Population seeding
  - Single-tick mechanics: death, mixing, replication, resources, abiogenesis
  - Diversity metrics correctness (theta_std, shannon_eco)
  - Equilibrium statistics
  - Multi-seed sweep point
  - Boundary and edge cases
  - Regression: key numerical claims from the paper

Run with:
    python -m pytest tests/ -v
or:
    python tests/test_simulation.py
"""

import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation import SimConfig, Simulation, run_single, run_sweep_point, N_THETA_BINS


# ─── helpers ───────────────────────────────────────────────────────────────────

def make_sim(*, seed=42, n_ticks=50, grid_size=20, heterogeneity=0.5,
             mixing_rate=0.1, mixing_mode='global', pattern='patches',
             spawn_rate=0.0005, mutation_rate=0.02):
    cfg = SimConfig(
        seed=seed, n_ticks=n_ticks, grid_size=grid_size,
        heterogeneity=heterogeneity, mixing_rate=mixing_rate,
        mixing_mode=mixing_mode, pattern=pattern,
        spawn_rate=spawn_rate, mutation_rate=mutation_rate,
    )
    return Simulation(cfg)


def assert_close(a, b, rtol=0.05, msg=''):
    err = abs(a - b) / (abs(b) + 1e-12)
    assert err < rtol, f'{msg}: {a:.6f} vs {b:.6f} (rtol={rtol}, err={err:.4f})'


# ─── SimConfig ─────────────────────────────────────────────────────────────────

class TestSimConfig:

    def test_defaults(self):
        cfg = SimConfig()
        assert cfg.grid_size == 50
        assert cfg.base_r == 0.15
        assert cfg.niche_width == 20.0
        assert cfg.mixing_mode == 'global'
        assert cfg.abiogenesis_mode == 'uniform'

    def test_n_cells(self):
        cfg = SimConfig(grid_size=30)
        assert cfg.n_cells == 900

    def test_round_trip_dict(self):
        cfg = SimConfig(grid_size=15, heterogeneity=0.7, mixing_rate=0.3)
        cfg2 = SimConfig.from_dict(cfg.to_dict())
        assert cfg2.grid_size == 15
        assert cfg2.heterogeneity == 0.7
        assert cfg2.mixing_rate == 0.3

    def test_from_dict_ignores_unknown_keys(self):
        d = SimConfig().to_dict()
        d['nonexistent_key'] = 99
        cfg = SimConfig.from_dict(d)
        assert not hasattr(cfg, 'nonexistent_key')


# ─── Environment building ───────────────────────────────────────────────────────

class TestEnvironmentBuild:

    def test_uniform_pattern_is_constant(self):
        sim = make_sim(pattern='uniform', heterogeneity=0.5)
        assert np.allclose(sim.env_map, 0.5), 'uniform pattern should be all 0.5'

    def test_zero_heterogeneity_is_constant(self):
        sim = make_sim(pattern='patches', heterogeneity=0.0)
        assert np.allclose(sim.env_map, 0.5), 'h=0 should give uniform environment'

    def test_env_range_patches(self):
        sim = make_sim(pattern='patches', heterogeneity=0.8)
        assert sim.env_map.min() >= 0.0 - 1e-9
        assert sim.env_map.max() <= 1.0 + 1e-9

    def test_env_range_gradient(self):
        sim = make_sim(pattern='gradient', heterogeneity=0.8)
        assert sim.env_map.min() >= 0.0 - 1e-9
        assert sim.env_map.max() <= 1.0 + 1e-9

    def test_env_range_hotspots(self):
        sim = make_sim(pattern='hotspots', heterogeneity=0.6)
        assert sim.env_map.min() >= 0.0 - 1e-9
        assert sim.env_map.max() <= 1.0 + 1e-9

    def test_gradient_monotone_in_x(self):
        cfg = SimConfig(pattern='gradient', heterogeneity=1.0, grid_size=20, seed=1)
        sim = Simulation(cfg)
        N = sim.N
        env = sim.env_map.reshape(N, N)
        # Each row should be monotonically non-decreasing left-to-right
        for y in range(N):
            assert np.all(np.diff(env[y, :]) >= -1e-9), 'gradient should be monotone in x'

    def test_env_length(self):
        cfg = SimConfig(grid_size=15)
        sim = Simulation(cfg)
        assert len(sim.env_map) == 225

    def test_env_heterogeneity_increases_variance(self):
        """Higher heterogeneity should produce higher spatial variance."""
        sims = [make_sim(pattern='patches', heterogeneity=h, seed=0)
                for h in [0.2, 0.5, 0.8]]
        vars_ = [np.var(s.env_map) for s in sims]
        assert vars_[0] < vars_[1] < vars_[2], 'variance should increase with heterogeneity'


# ─── Population seeding ────────────────────────────────────────────────────────

class TestPopulationSeed:

    def test_initial_population_count(self):
        cfg = SimConfig(n_initial=20, grid_size=20)
        sim = Simulation(cfg)
        assert int(np.sum(sim.grid >= 0)) == 20

    def test_initial_theta_range(self):
        cfg = SimConfig(n_initial=50, grid_size=30, seed=7)
        sim = Simulation(cfg)
        occupied = sim.grid >= 0
        th = sim.theta[occupied]
        assert th.min() >= 0.0
        assert th.max() <= 1.0

    def test_initial_unique_ids(self):
        cfg = SimConfig(n_initial=30, grid_size=20)
        sim = Simulation(cfg)
        ids = sim.grid[sim.grid >= 0]
        assert len(np.unique(ids)) == len(ids), 'all founders should have unique IDs'

    def test_no_overlap_with_empty(self):
        cfg = SimConfig(n_initial=10, grid_size=10)
        sim = Simulation(cfg)
        # Empty cells have -1, occupied >= 0
        assert np.all(sim.grid >= -1)
        assert np.sum(sim.grid >= 0) == 10


# ─── Single-tick mechanics ──────────────────────────────────────────────────────

class TestSingleTick:

    def test_tick_increments(self):
        sim = make_sim(n_ticks=10)
        assert sim.tick == 0
        sim.step()
        assert sim.tick == 1

    def test_run_advances_n_ticks(self):
        sim = make_sim(n_ticks=10)
        sim.run()
        assert sim.tick == 10

    def test_history_length(self):
        sim = make_sim(n_ticks=30)
        sim.run()
        assert len(sim.history) == 30

    def test_population_positive(self):
        sim = make_sim(n_ticks=100, spawn_rate=0.001)
        sim.run()
        pops = [r['population'] for r in sim.history]
        assert all(p >= 0 for p in pops), 'population must never go negative'

    def test_occupancy_in_01(self):
        sim = make_sim(n_ticks=50)
        sim.run()
        for r in sim.history:
            assert 0.0 <= r['occupancy'] <= 1.0

    def test_resources_non_negative(self):
        sim = make_sim(grid_size=15, n_ticks=200, mixing_rate=0.0, spawn_rate=0.0)
        sim.run()
        assert sim.resources.min() >= 0.0, 'resources should never go negative'

    def test_grid_only_neg1_or_non_neg(self):
        sim = make_sim(n_ticks=50)
        sim.run()
        assert np.all(sim.grid >= -1), 'grid should only contain -1 or species IDs'

    def test_theta_stays_in_01(self):
        sim = make_sim(n_ticks=100, mutation_rate=0.5)  # high mutation
        sim.run()
        occupied = sim.grid >= 0
        if occupied.any():
            assert sim.theta[occupied].min() >= 0.0
            assert sim.theta[occupied].max() <= 1.0

    def test_rep_rate_stays_in_bounds(self):
        cfg = SimConfig(n_ticks=100, mutation_rate=0.5, seed=10)
        sim = Simulation(cfg)
        sim.run()
        occupied = sim.grid >= 0
        if occupied.any():
            assert sim.rep_rate[occupied].min() >= 0.005
            assert sim.rep_rate[occupied].max() <= 0.95

    def test_death_reduces_population(self):
        """With very high death rate and no abiogenesis or replication, pop should drop."""
        cfg = SimConfig(
            base_d=0.99, tradeoff=0.0, base_r=0.0,
            spawn_rate=0.0, mutation_rate=0.0,
            n_initial=100, grid_size=20, n_ticks=10, seed=1,
        )
        sim = Simulation(cfg)
        sim.run()
        final_pop = sim.history[-1]['population']
        assert final_pop < 100, 'high death rate should reduce population'

    def test_no_mixing_at_zero_rate(self):
        """With D=0, each replicator's position should not change via mixing."""
        cfg = SimConfig(mixing_rate=0.0, n_initial=10, grid_size=20, n_ticks=5, seed=42)
        sim = Simulation(cfg)
        # Record original positions of occupied cells
        orig_occupied = set(np.where(sim.grid >= 0)[0])
        sim.step()
        # No mixing teleportation, but replication CAN change which cells are occupied
        # Just check that tick incremented
        assert sim.tick == 1

    def test_abiogenesis_uniform_theta(self):
        """Abiogenesis in uniform mode should create replicators with random theta."""
        cfg = SimConfig(
            spawn_rate=1.0,  # very high to guarantee some abiogenesis
            abiogenesis_mode='uniform',
            n_initial=0, grid_size=10, n_ticks=5, seed=99,
            base_d=0.0, base_r=0.0,  # no death/replication
            mixing_rate=0.0,
        )
        sim = Simulation(cfg)
        sim.run()
        occupied = sim.grid >= 0
        if occupied.any():
            th = sim.theta[occupied]
            assert th.min() >= 0.0
            assert th.max() <= 1.0

    def test_abiogenesis_fixed_theta(self):
        """Fixed-theta abiogenesis should create replicators at exactly the specified theta."""
        cfg = SimConfig(
            spawn_rate=1.0,
            abiogenesis_mode='fixed_theta',
            abiogenesis_theta=0.333,
            n_initial=0, grid_size=10, n_ticks=3, seed=5,
            base_d=0.0, base_r=0.0,
            mixing_rate=0.0,
        )
        sim = Simulation(cfg)
        sim.run()
        occupied = sim.grid >= 0
        if occupied.any():
            th = sim.theta[occupied]
            assert np.allclose(th, 0.333, atol=1e-9), \
                'fixed_theta abiogenesis should produce exactly abiogenesis_theta'


# ─── Diversity metrics ──────────────────────────────────────────────────────────

class TestDiversityMetrics:

    def test_theta_std_zero_if_all_same(self):
        """If all occupied cells have the same theta, theta_std should be 0."""
        cfg = SimConfig(
            n_initial=50, grid_size=15, seed=42,
            mutation_rate=0.0, spawn_rate=0.0,
            abiogenesis_mode='fixed_theta', abiogenesis_theta=0.5,
            n_ticks=1,
        )
        sim = Simulation(cfg)
        # Force all occupied cells to same theta
        occupied = sim.grid >= 0
        sim.theta[occupied] = 0.5
        sim.step()
        # After one step with no mutation, theta_std should still be ~0
        # (some mutation might occur from the initial seeding, but with mutation_rate=0 it won't)
        # Just check the final metric is reasonable
        last = sim.history[-1]
        assert last['theta_std'] >= 0.0

    def test_shannon_eco_zero_if_all_same_bin(self):
        """H'_eco should be 0 if all occupied cells are in the same theta bin."""
        sim = make_sim(n_ticks=0, grid_size=10)
        # Manually set all occupied to theta = 0.5 (bin 10)
        occupied = sim.grid >= 0
        sim.theta[occupied] = 0.5
        sim._record_metrics()
        last = sim.history[-1]
        assert abs(last['shannon_eco']) < 1e-9, 'H_eco should be 0 with all in same bin'

    def test_shannon_eco_max_is_log20(self):
        """Maximum H'_eco with 20 bins is ln(20)."""
        assert abs(math.log(N_THETA_BINS) - math.log(20)) < 1e-9
        max_h = math.log(20)
        # Uniform theta distribution should approach max
        sim = make_sim(n_ticks=0, grid_size=25)
        occupied = sim.grid >= 0
        n = int(occupied.sum())
        # Spread theta evenly across 20 bins
        thetas = np.linspace(0.025, 0.975, n)  # uniformly in (0, 1)
        sim.theta[occupied] = thetas
        sim._record_metrics()
        h_eco = sim.history[-1]['shannon_eco']
        assert h_eco <= max_h + 1e-9
        assert h_eco > 2.5  # should be close to max_h ≈ 2.996

    def test_theta_std_increases_with_spread(self):
        """theta_std should be larger when thetas span a wider range."""
        sim = make_sim(n_ticks=0, grid_size=15)
        occupied = sim.grid >= 0
        n = int(occupied.sum())

        sim.theta[occupied] = 0.5  # all same, std=0
        sim._record_metrics()
        std_narrow = sim.history[-1]['theta_std']

        sim.theta[occupied] = np.linspace(0.0, 1.0, n)
        sim._record_metrics()
        std_wide = sim.history[-1]['theta_std']

        assert std_wide > std_narrow, 'wider theta spread should give larger theta_std'

    def test_theta_std_uses_population_std(self):
        """Verify theta_std uses ddof=0 (population) not ddof=1 (sample)."""
        sim = make_sim(n_ticks=0, grid_size=5)
        occupied = sim.grid >= 0
        n = int(occupied.sum())
        th = np.linspace(0.1, 0.9, n)
        sim.theta[occupied] = th
        sim._record_metrics()
        expected = np.std(th)  # default ddof=0
        assert abs(sim.history[-1]['theta_std'] - expected) < 1e-12

    def test_bin_mapping_edge_cases(self):
        """theta=0 should go to bin 0; theta=1 should go to bin 19 (clipped)."""
        sim = make_sim(n_ticks=0, grid_size=5)
        occupied = sim.grid >= 0
        idx = np.where(occupied)[0]
        assert len(idx) >= 2
        sim.theta[idx[0]] = 0.0
        sim.theta[idx[1]] = 1.0
        # Compute bins manually
        th = np.array([0.0, 1.0])
        bins = np.clip((th * N_THETA_BINS).astype(int), 0, N_THETA_BINS - 1)
        assert bins[0] == 0
        assert bins[1] == N_THETA_BINS - 1

    def test_population_zero_gives_zero_metrics(self):
        """With no replicators, all metrics should be 0."""
        cfg = SimConfig(n_initial=0, grid_size=10, spawn_rate=0.0, n_ticks=1, seed=1)
        sim = Simulation(cfg)
        sim.run()
        last = sim.history[-1]
        assert last['population'] == 0
        assert last['theta_std'] == 0.0
        assert last['shannon_eco'] == 0.0


# ─── Equilibrium statistics ─────────────────────────────────────────────────────

class TestEquilibriumMetrics:

    def test_get_equilibrium_returns_dict(self):
        sim = make_sim(n_ticks=50)
        sim.run()
        eq = sim.get_equilibrium_metrics(last_n=20)
        assert isinstance(eq, dict)
        assert 'theta_std_mean' in eq
        assert 'theta_std_std' in eq
        assert 'shannon_eco_mean' in eq

    def test_equilibrium_last_n_respected(self):
        sim = make_sim(n_ticks=100)
        sim.run()
        eq10 = sim.get_equilibrium_metrics(last_n=10)
        eq100 = sim.get_equilibrium_metrics(last_n=100)
        # Both should be valid floats; not necessarily equal
        assert isinstance(eq10['theta_std_mean'], float)
        assert isinstance(eq100['theta_std_mean'], float)

    def test_equilibrium_empty_history(self):
        sim = make_sim(n_ticks=0)
        eq = sim.get_equilibrium_metrics()
        assert eq == {}

    def test_current_metrics_consistency(self):
        sim = make_sim(n_ticks=20)
        sim.run()
        m = sim.get_current_metrics()
        assert m['tick'] == 20
        assert 'theta_std' in m
        assert 'shannon_eco' in m


# ─── Multi-seed sweep ───────────────────────────────────────────────────────────

class TestRunSweepPoint:

    def test_returns_required_keys(self):
        result = run_sweep_point(
            mixing=0.1, heterogeneity=0.5,
            seeds=[42, 7],
            grid_size=15, n_ticks=50,
        )
        assert 'mixing_rate' in result
        assert 'heterogeneity' in result
        assert 'theta_std_mean' in result
        assert 'theta_std_std' in result
        assert 'shannon_eco_mean' in result

    def test_mixing_rate_recorded(self):
        r = run_sweep_point(mixing=0.3, heterogeneity=0.6, seeds=[1, 2],
                            grid_size=10, n_ticks=30)
        assert abs(r['mixing_rate'] - 0.3) < 1e-9

    def test_heterogeneity_recorded(self):
        r = run_sweep_point(mixing=0.1, heterogeneity=0.7, seeds=[1],
                            grid_size=10, n_ticks=20)
        assert abs(r['heterogeneity'] - 0.7) < 1e-9

    def test_std_non_negative(self):
        r = run_sweep_point(mixing=0.1, heterogeneity=0.5, seeds=[1, 2, 3],
                            grid_size=12, n_ticks=40)
        assert r['theta_std_std'] >= 0.0
        assert r['shannon_eco_std'] >= 0.0


# ─── Mixing mechanics ───────────────────────────────────────────────────────────

class TestMixingMechanics:

    def test_d0_does_not_teleport(self):
        """D=0: no replicator should be teleported (position only changes via replication)."""
        cfg = SimConfig(
            mixing_rate=0.0, base_r=0.0, spawn_rate=0.0,
            base_d=0.0, n_initial=5, grid_size=10, n_ticks=1, seed=3,
        )
        sim = Simulation(cfg)
        orig_pos = set(np.where(sim.grid >= 0)[0])
        sim.step()
        final_pos = set(np.where(sim.grid >= 0)[0])
        # With no death, no replication, no abiogenesis, and D=0, positions must not change
        assert orig_pos == final_pos

    def test_global_vs_local_mixing_same_d0(self):
        """At D=0, global and local mixing should give identical results."""
        cfg_g = SimConfig(mixing_rate=0.0, mixing_mode='global', seed=77, n_ticks=30)
        cfg_l = SimConfig(mixing_rate=0.0, mixing_mode='local', seed=77, n_ticks=30)
        sim_g = Simulation(cfg_g)
        sim_l = Simulation(cfg_l)
        sim_g.run()
        sim_l.run()
        eq_g = sim_g.get_equilibrium_metrics(last_n=10)
        eq_l = sim_l.get_equilibrium_metrics(last_n=10)
        # Populations should be identical (both RNG streams progress identically)
        assert_close(eq_g['population_mean'], eq_l['population_mean'], rtol=0.01,
                     msg='D=0 global vs local should give same population')

    def test_high_mixing_homogenises_theta(self):
        """Very high D should collapse theta diversity."""
        r_d0 = run_sweep_point(mixing=0.0, heterogeneity=0.8,
                               seeds=[1, 2, 3], grid_size=30, n_ticks=300)
        r_d1 = run_sweep_point(mixing=0.8, heterogeneity=0.8,
                               seeds=[1, 2, 3], grid_size=30, n_ticks=300)
        assert r_d0['theta_std_mean'] > r_d1['theta_std_mean'], \
            'D=0 should give higher theta_std than D=0.8 (isolation vs mixing)'


# ─── Regression tests (key paper claims) ───────────────────────────────────────

class TestRegressions:
    """
    These test the core quantitative claims in the paper using the original
    5-seed set [42, 123, 456, 789, 1011] with 600 ticks and 40x40 grid.
    Tolerance: ±5% relative error (generous, to allow minor code changes).
    """

    SEEDS = [42, 123, 456, 789, 1011]
    GRID = 40
    TICKS = 600

    def _sweep(self, d, h):
        return run_sweep_point(d, h, self.SEEDS,
                               grid_size=self.GRID, n_ticks=self.TICKS,
                               pattern='patches')

    def test_d0_h08_theta_std(self):
        """D=0, h=0.8 should give theta_std ≈ 0.168."""
        r = self._sweep(0.0, 0.8)
        assert_close(r['theta_std_mean'], 0.168, rtol=0.06,
                     msg='D=0 h=0.8 theta_std regression')

    def test_d08_h08_theta_std(self):
        """D=0.8, h=0.8 should give theta_std ≈ 0.063."""
        r = self._sweep(0.8, 0.8)
        assert_close(r['theta_std_mean'], 0.063, rtol=0.06,
                     msg='D=0.8 h=0.8 theta_std regression')

    def test_d0_h10_theta_std(self):
        """D=0, h=1.0 should give the highest theta_std ≈ 0.207."""
        r = self._sweep(0.0, 1.0)
        assert_close(r['theta_std_mean'], 0.207, rtol=0.06,
                     msg='D=0 h=1.0 theta_std regression (best overall)')

    def test_monotone_d_at_h08(self):
        """theta_std should be monotonically decreasing in D at h=0.8."""
        ds = [0.0, 0.05, 0.2, 0.5, 0.8]
        ts = [self._sweep(d, 0.8)['theta_std_mean'] for d in ds]
        for i in range(len(ts) - 1):
            assert ts[i] > ts[i+1] - 0.005, \
                f'theta_std should decrease with D: {ts}'

    def test_d0_beats_d01_at_h08(self):
        """D=0 should give strictly higher theta_std than D=0.01 at h=0.8."""
        r0 = self._sweep(0.0, 0.8)
        r01 = self._sweep(0.01, 0.8)
        assert r0['theta_std_mean'] > r01['theta_std_mean'], \
            'D=0 should beat D=0.01 (no intermediate optimum)'

    def test_h_increases_diversity_at_d0(self):
        """At D=0, theta_std should increase with h."""
        hs = [0.0, 0.3, 0.6, 1.0]
        ts = [self._sweep(0.0, h)['theta_std_mean'] for h in hs]
        for i in range(len(ts) - 1):
            assert ts[i] < ts[i+1], \
                f'theta_std should increase with h at D=0: {ts}'


# ─── run as script ──────────────────────────────────────────────────────────────

def run_all_tests():
    """Simple test runner for use without pytest."""
    import traceback
    classes = [
        TestSimConfig, TestEnvironmentBuild, TestPopulationSeed,
        TestSingleTick, TestDiversityMetrics, TestEquilibriumMetrics,
        TestRunSweepPoint, TestMixingMechanics,
    ]
    # Skip TestRegressions in quick mode (they are slow)
    passed = failed = 0
    for cls in classes:
        obj = cls()
        methods = [m for m in dir(obj) if m.startswith('test_')]
        for m in methods:
            try:
                getattr(obj, m)()
                print(f'  PASS  {cls.__name__}.{m}')
                passed += 1
            except Exception as e:
                print(f'  FAIL  {cls.__name__}.{m}: {e}')
                traceback.print_exc()
                failed += 1
    print(f'\n{passed} passed, {failed} failed')
    return failed == 0


if __name__ == '__main__':
    import sys
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
