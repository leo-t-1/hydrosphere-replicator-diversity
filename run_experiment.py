#!/usr/bin/env python3
"""
Full hydrosphere replicator experiment.

Phases:
  1. Four hydrosphere scenario simulations (5 seeds each, time series)
  2. Mixing × heterogeneity parameter sweep (10×10, 5 seeds)
  3. Niche-width sensitivity analysis (β ∈ {10, 20, 40})
  4. Publication-quality figures
  5. Results summary

Usage:
    python run_experiment.py              # Full experiment
    python run_experiment.py --quick      # Quick test
"""

import os, sys, time, json, csv, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation import SimConfig, Simulation, run_single, run_sweep_point
from experiment_summary import build_experiment_summary, write_experiment_summary

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE, 'results')
FIGURES = os.path.join(BASE, 'figures')

# ── Colours ──
BLUE   = '#2563eb'
GREEN  = '#059669'
AMBER  = '#d97706'
RED    = '#dc2626'
CYAN   = '#0891b2'
DARK   = '#1e293b'
GRAY   = '#6b7280'

# ── Hydrosphere scenarios ──
SCENARIOS = {
    'deep_ocean': dict(
        label='Deep Ocean', color=BLUE,
        heterogeneity=0.1, mixing_rate=0.5, pattern='uniform',
        desc='Low heterogeneity, high mixing'),
    'pond_network': dict(
        label='Shallow Ponds', color=GREEN,
        heterogeneity=0.8, mixing_rate=0.02, pattern='patches',
        desc='High heterogeneity, low mixing'),
    'tidal_zone': dict(
        label='Tidal Zone', color=AMBER,
        heterogeneity=0.7, mixing_rate=0.15, pattern='patches',
        desc='High heterogeneity, moderate mixing'),
    'hydrothermal_vent': dict(
        label='Hydrothermal Vents', color=RED,
        heterogeneity=0.6, mixing_rate=0.08, pattern='hotspots',
        desc='Moderate heterogeneity, low-moderate mixing'),
}

# ── Sweep grid ──
SWEEP_D = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.80]
SWEEP_H = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
SEEDS   = [42, 123, 456, 789, 1011]

# Quick mode
QUICK_D = [0.0, 0.05, 0.15, 0.30, 0.80]
QUICK_H = [0.0, 0.3, 0.5, 0.8, 1.0]
QUICK_SEEDS = [42, 123]


# ============================================================
# PHASE 1: Scenario simulations
# ============================================================

def run_scenarios(grid_size=40, n_ticks=800, seeds=None):
    """Run 4 hydrosphere scenarios with multiple seeds."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1011]

    print("\n" + "=" * 60)
    print("PHASE 1: Hydrosphere Scenario Simulations")
    print(f"  Grid: {grid_size}×{grid_size}, {n_ticks} ticks, {len(seeds)} seeds")
    print("=" * 60)

    results = {}
    for name, spec in SCENARIOS.items():
        print(f"\n  ── {spec['label']} (h={spec['heterogeneity']}, D={spec['mixing_rate']}) ──")

        eq_list = []
        primary_history = None
        primary_snapshot = None

        for seed in seeds:
            cfg = SimConfig(
                grid_size=grid_size,
                heterogeneity=spec['heterogeneity'],
                mixing_rate=spec['mixing_rate'],
                mixing_mode='global',
                pattern=spec['pattern'],
                n_ticks=n_ticks,
                seed=seed,
            )
            h, eq, snap = run_single(cfg, verbose=(seed == seeds[0]))
            eq_list.append(eq)
            if seed == seeds[0]:
                primary_history = h
                primary_snapshot = snap

        # Save time series for primary seed
        csv_path = os.path.join(RESULTS, f'scenario_{name}.csv')
        _save_history(primary_history, csv_path)

        # Compute proper across-seed statistics
        eq_summary = {}
        for key in eq_list[0]:
            if key.endswith('_mean'):
                base = key[:-5]
                vals = [e[key] for e in eq_list]
                eq_summary[base + '_mean'] = float(np.mean(vals))
                eq_summary[base + '_std'] = float(np.std(vals))

        results[name] = {
            'history': primary_history,
            'equilibrium': eq_summary,
            'snapshot': primary_snapshot,
        }

        ts = eq_summary.get('theta_std_mean', 0)
        he = eq_summary.get('shannon_eco_mean', 0)
        print(f"  θ_std={ts:.3f}±{eq_summary.get('theta_std_std',0):.3f}  "
              f"H'_eco={he:.2f}±{eq_summary.get('shannon_eco_std',0):.2f}")

    return results


# ============================================================
# PHASE 2: Parameter sweep
# ============================================================

def run_sweep(d_vals, h_vals, seeds, grid_size=40, n_ticks=600):
    """10×10 mixing × heterogeneity sweep."""
    print("\n" + "=" * 60)
    print("PHASE 2: Parameter Sweep")
    n_total = len(d_vals) * len(h_vals)
    print(f"  {len(d_vals)}×{len(h_vals)} = {n_total} points, "
          f"{len(seeds)} seeds, {n_total * len(seeds)} total runs")
    print("=" * 60)

    results = []
    t0 = time.time()

    for idx, (hi, h) in enumerate([(i, hv) for i, hv in enumerate(h_vals)]):
        for di, d in enumerate(d_vals):
            n = idx * len(d_vals) + di + 1
            avg = run_sweep_point(
                mixing=d, heterogeneity=h, seeds=seeds,
                grid_size=grid_size, n_ticks=n_ticks, pattern='patches')
            results.append(avg)

            elapsed = time.time() - t0
            eta = elapsed / n * (n_total - n)
            print(f"\r  [{n}/{n_total}] h={h:.1f} D={d:.2f} "
                  f"θ_std={avg.get('theta_std_mean',0):.3f} "
                  f"H'_eco={avg.get('shannon_eco_mean',0):.2f} "
                  f"[{elapsed:.0f}s ETA {eta:.0f}s]",
                  end='', flush=True)

    print(f"\n  Sweep done in {time.time()-t0:.0f}s")

    csv_path = os.path.join(RESULTS, 'sweep_results.csv')
    if results:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    return results


# ============================================================
# PHASE 3: Sensitivity analysis
# ============================================================

def run_sensitivity(seeds, grid_size=40, n_ticks=600):
    """Niche-width sensitivity: β ∈ {10, 20, 40} at h=0.8."""
    print("\n" + "=" * 60)
    print("PHASE 3: Niche-Width Sensitivity Analysis")
    print("=" * 60)

    betas = [10.0, 20.0, 40.0]
    d_vals = SWEEP_D
    results = {}
    t0 = time.time()

    for beta in betas:
        results[beta] = []
        base = SimConfig(niche_width=beta, mixing_mode='global')
        for d in d_vals:
            avg = run_sweep_point(
                mixing=d, heterogeneity=0.8, seeds=seeds,
                base_config=base, grid_size=grid_size,
                n_ticks=n_ticks, pattern='patches')
            avg['niche_width'] = beta
            results[beta].append(avg)
        elapsed = time.time() - t0
        print(f"  β={beta:.0f} done [{elapsed:.0f}s]")

    # Save
    all_rows = []
    for beta in betas:
        all_rows.extend(results[beta])
    csv_path = os.path.join(RESULTS, 'sensitivity_niche_width.csv')
    if all_rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            w.writeheader()
            w.writerows(all_rows)

    return results


# ============================================================
# PHASE 6: Local diffusion sweep
# ============================================================

def run_local_sweep(d_vals, h_vals, seeds, grid_size=40, n_ticks=800):
    """Full parameter sweep with physically realistic local diffusion."""
    print("\n" + "=" * 60)
    print("PHASE 6: Local Diffusion Parameter Sweep")
    n_total = len(d_vals) * len(h_vals)
    print(f"  mixing_mode=local, {len(d_vals)}x{len(h_vals)} = {n_total} points, "
          f"{len(seeds)} seeds, {n_ticks} ticks")
    print("=" * 60)

    base_config = SimConfig(mixing_mode='local')
    results = []
    t0 = time.time()

    for idx, (hi, h) in enumerate([(i, hv) for i, hv in enumerate(h_vals)]):
        for di, d in enumerate(d_vals):
            n = idx * len(d_vals) + di + 1
            avg = run_sweep_point(
                mixing=d, heterogeneity=h, seeds=seeds,
                base_config=base_config,
                grid_size=grid_size, n_ticks=n_ticks, pattern='patches')
            results.append(avg)

            elapsed = time.time() - t0
            eta = elapsed / n * (n_total - n)
            print(f"\r  [{n}/{n_total}] h={h:.1f} D={d:.2f} "
                  f"theta_std={avg.get('theta_std_mean',0):.3f} "
                  f"H'_eco={avg.get('shannon_eco_mean',0):.2f} "
                  f"[{elapsed:.0f}s ETA {eta:.0f}s]",
                  end='', flush=True)

    print(f"\n  Local sweep done in {time.time()-t0:.0f}s")

    csv_path = os.path.join(RESULTS, 'local_sweep_results.csv')
    if results:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    return results


# ============================================================
# PHASE 7: Colonization-limited sweep
# ============================================================

def run_colonization_sweep(d_vals, h_vals, seeds, grid_size=40, n_ticks=800):
    """Local diffusion with reduced abiogenesis (colonization-limited regime).

    With spawn_rate near zero, empty patches can only be rescued by
    neighbouring dispersers.  This creates the colonization-homogenisation
    tradeoff that can produce an intermediate dispersal optimum.
    """
    print("\n" + "=" * 60)
    print("PHASE 7: Colonization-Limited Sweep (local, low abiogenesis)")
    n_total = len(d_vals) * len(h_vals)
    print(f"  mixing_mode=local, spawn_rate=0.0001")
    print(f"  {len(d_vals)}x{len(h_vals)} = {n_total} points, {len(seeds)} seeds")
    print("=" * 60)

    base_config = SimConfig(mixing_mode='local', spawn_rate=0.0001, n_initial=50)
    results = []
    t0 = time.time()

    for idx, (hi, h) in enumerate([(i, hv) for i, hv in enumerate(h_vals)]):
        for di, d in enumerate(d_vals):
            n = idx * len(d_vals) + di + 1
            avg = run_sweep_point(
                mixing=d, heterogeneity=h, seeds=seeds,
                base_config=base_config,
                grid_size=grid_size, n_ticks=n_ticks, pattern='patches')
            results.append(avg)

            elapsed = time.time() - t0
            eta = elapsed / n * (n_total - n)
            print(f"\r  [{n}/{n_total}] h={h:.1f} D={d:.2f} "
                  f"theta_std={avg.get('theta_std_mean',0):.3f} "
                  f"[{elapsed:.0f}s ETA {eta:.0f}s]",
                  end='', flush=True)

    print(f"\n  Colonization sweep done in {time.time()-t0:.0f}s")

    csv_path = os.path.join(RESULTS, 'colonization_sweep_results.csv')
    if results:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=results[0].keys())
            w.writeheader()
            w.writerows(results)

    return results


# ============================================================
# Statistical helpers
# ============================================================

def compute_pairwise_stats(sweep_results, h_target, d_ref=0.0, n_seeds=5):
    """Welch t-test for each D vs D=d_ref at given h.

    Returns list of dicts: mixing_rate, theta_std_mean/std, t_stat,
    p_value, cohen_d.
    """
    from scipy import stats as sp_stats

    ref_row = None
    target_rows = []
    for r in sweep_results:
        if abs(r['heterogeneity'] - h_target) < 0.01:
            target_rows.append(r)
            if abs(r['mixing_rate'] - d_ref) < 0.005:
                ref_row = r

    if ref_row is None:
        return []

    target_rows.sort(key=lambda r: r['mixing_rate'])
    ref_mean = ref_row['theta_std_mean']
    ref_std = max(ref_row['theta_std_std'], 1e-10)

    out = []
    for row in target_rows:
        d = row['mixing_rate']
        m = row['theta_std_mean']
        s = max(row['theta_std_std'], 1e-10)

        se = np.sqrt(ref_std**2 / n_seeds + s**2 / n_seeds)
        t_stat = (ref_mean - m) / se if se > 0 else 0
        num = (ref_std**2 / n_seeds + s**2 / n_seeds)**2
        den = ((ref_std**2 / n_seeds)**2 / max(1, n_seeds - 1)
               + (s**2 / n_seeds)**2 / max(1, n_seeds - 1))
        df = num / den if den > 0 else n_seeds - 1
        p_val = float(2 * (1 - sp_stats.t.cdf(abs(t_stat), df))) if df > 0 else 1.0

        pooled_std = np.sqrt((ref_std**2 + s**2) / 2)
        cohen_d = (ref_mean - m) / pooled_std if pooled_std > 0 else 0

        out.append({
            'mixing_rate': d, 'theta_std_mean': m, 'theta_std_std': s,
            't_stat': float(t_stat), 'p_value': p_val, 'cohen_d': float(cohen_d),
        })

    return out


def _sig_stars(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ============================================================
# PHASE 4: Figures
# ============================================================

def setup_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'figure.dpi': 150, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.12,
    })


def fig_model_schematic():
    """Fig 1: Model overview — fitness function, grid, metrics."""
    fig = plt.figure(figsize=(10, 3.8))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.42)

    # Panel A: Gaussian fitness
    ax = fig.add_subplot(gs[0])
    E = np.linspace(0, 1, 200)
    for b, lab, col, lw in [(10, 'β=10', '#94a3b8', 1.3),
                             (20, 'β=20', CYAN, 2.2),
                             (40, 'β=40', DARK, 1.3)]:
        ax.plot(E, np.exp(-b * (E - 0.5)**2), color=col, lw=lw, label=lab)
    ax.axvline(0.5, color=CYAN, lw=1, ls='--', alpha=0.5)
    ax.set_xlabel('Environment E')
    ax.set_ylabel('Niche match')
    ax.set_title('A  Fitness function', fontweight='bold', loc='left')
    ax.annotate('θ = 0.5', xy=(0.5, 1.0), xytext=(0.68, 0.82),
                arrowprops=dict(arrowstyle='->', color=CYAN), fontsize=8, color=CYAN)
    ax.text(0.04, 0.22, 'f = r · exp(−β(E−θ)²) · R', fontsize=7.5,
            transform=ax.transAxes, color=DARK,
            bbox=dict(boxstyle='round,pad=0.3', fc='#f0f9ff', ec='#bae6fd'))
    ax.legend(fontsize=7.5, framealpha=0.7)
    ax.set_ylim(-0.05, 1.12)

    # Panel B: Grid cartoon
    theta_cmap = LinearSegmentedColormap.from_list('t',
        [(0, '#1e50dc'), (0.2, '#00bec8'), (0.5, '#32dc96'),
         (0.8, '#dcb41e'), (1, '#dc3232')])
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 5); ax2.set_ylim(0, 5); ax2.set_aspect('equal'); ax2.axis('off')
    ax2.set_title('B  Spatial grid', fontweight='bold', loc='left')
    env_g = np.array([[.2,.3,.5,.7,.8],[.2,.4,.5,.7,.9],[.3,.4,.5,.6,.8],
                      [.4,.5,.6,.7,.7],[.5,.5,.5,.6,.6]])
    th_g = np.array([[.2,np.nan,.5,np.nan,.8],[np.nan,.4,np.nan,.7,np.nan],
                     [.3,np.nan,.5,np.nan,.8],[np.nan,.5,np.nan,.7,np.nan],
                     [.5,np.nan,.5,np.nan,.6]])
    for y in range(5):
        for x in range(5):
            bg = plt.cm.Blues(0.2 + 0.5*env_g[4-y, x])
            ax2.add_patch(plt.Rectangle((x+.05, y+.05), .9, .9, fc=bg, ec='white', lw=.5))
            th = th_g[4-y, x]
            if not np.isnan(th):
                ax2.add_patch(plt.Circle((x+.5, y+.5), .3, color=theta_cmap(th), zorder=3))
                ax2.text(x+.5, y+.5, f'{th:.1f}', ha='center', va='center',
                         fontsize=6, color='white', fontweight='bold', zorder=4)
    ax2.text(2.5, -.35, 'environment gradient', ha='center', fontsize=7.5,
             color=GRAY, style='italic')

    # Panel C: Two metrics
    ax3 = fig.add_subplot(gs[2])
    ax3.set_title('C  Two diversity metrics', fontweight='bold', loc='left')
    x = np.linspace(0, 1, 200)
    ocean = 0.8*np.exp(-80*(x-.5)**2) + 0.05
    pond = 0.4*np.exp(-60*(x-.2)**2) + 0.4*np.exp(-60*(x-.5)**2) + 0.3*np.exp(-60*(x-.8)**2) + 0.05
    ax3.fill_between(x, ocean, alpha=.25, color=BLUE); ax3.plot(x, ocean, color=BLUE, lw=1.5, label='Ocean')
    ax3.fill_between(x, pond, alpha=.25, color=GREEN); ax3.plot(x, pond, color=GREEN, lw=1.5, label='Ponds')
    ax3.text(.52, .46, 'low θ_std', fontsize=7.5, color=BLUE)
    ax3.text(.28, .32, 'high θ_std', fontsize=7.5, color=GREEN)
    ax3.set_xlabel('θ (niche optimum)')
    ax3.set_ylabel('Frequency')
    ax3.legend(fontsize=8, loc='upper right')
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1.0)

    fig.suptitle('Model: Niche-Based Replicator Diversity on a 2D Lattice',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.savefig(os.path.join(FIGURES, 'fig1_model_schematic.png'))
    plt.close(fig)
    print('  fig1_model_schematic.png')


def fig_heatmaps(sweep_results, d_vals, h_vals):
    """Fig 2: Parameter-space heatmaps for θ_std and H'_eco."""
    nd, nh = len(d_vals), len(h_vals)

    def make_mat(key):
        mat = np.zeros((nh, nd))
        for r in sweep_results:
            di = d_vals.index(r['mixing_rate'])
            hi = h_vals.index(r['heterogeneity'])
            mat[hi, di] = r.get(key, 0)
        return mat

    mat_ts = make_mat('theta_std_mean')
    mat_he = make_mat('shannon_eco_mean')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor('white')

    sc_pts = {'Deep Ocean': (0.50, 0.1), 'Ponds': (0.02, 0.8),
              'Tidal': (0.15, 0.7), 'Vents': (0.08, 0.6)}
    sc_c = [BLUE, GREEN, AMBER, RED]

    for ax, mat, title, cmap, cbl in [
        (axes[0], mat_ts, 'Functional Diversity (θ_std)', 'YlOrRd', 'θ_std'),
        (axes[1], mat_he, "Ecological Diversity (H'_eco)", 'viridis', "H'_eco"),
    ]:
        im = ax.imshow(mat, origin='lower', aspect='auto',
                       extent=[-.5, nd-.5, -.5, nh-.5],
                       cmap=cmap, interpolation='bilinear')
        ax.set_xticks(range(nd))
        ax.set_xticklabels([f'{v:.2f}' for v in d_vals], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(nh))
        ax.set_yticklabels([f'{v:.1f}' for v in h_vals], fontsize=8)
        ax.set_xlabel('Mixing Rate D')
        ax.set_ylabel('Heterogeneity h')
        ax.set_title(title, fontweight='bold', pad=8)
        div = make_axes_locatable(ax)
        cax = div.append_axes('right', size='4%', pad=0.06)
        plt.colorbar(im, cax=cax).set_label(cbl, fontsize=8)

        for (lab, (d, h)), col in zip(sc_pts.items(), sc_c):
            di = min(range(nd), key=lambda i: abs(d_vals[i] - d))
            hi = min(range(nh), key=lambda i: abs(h_vals[i] - h))
            ax.plot(di, hi, 'o', ms=9, mfc='white', mec=col, mew=2, zorder=5)
            ax.annotate(lab, (di, hi), xytext=(di+.4, hi+.3), fontsize=7.5,
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=.15', fc=col, ec='none', alpha=.85))

    axes[0].text(.02, .97, 'Isolation + heterogeneity\n= highest theta_std',
                 transform=axes[0].transAxes, fontsize=8, va='top',
                 bbox=dict(boxstyle='round', fc='#fff7ed', ec='#fdba74', alpha=.9))

    fig.suptitle('Parameter Space: Mixing and Heterogeneity Shape Diversity',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig2_heatmaps.png'))
    plt.close(fig)
    print('  fig2_heatmaps.png')


def fig_scenario_bars(scenario_results):
    """Fig 3: Bar chart comparison of 4 scenarios with error bars."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.patch.set_facecolor('white')

    names = ['deep_ocean', 'pond_network', 'tidal_zone', 'hydrothermal_vent']
    labels = ['Deep\nOcean', 'Shallow\nPonds', 'Tidal\nZone', 'Hydro-\nthermal']
    colors = [BLUE, GREEN, AMBER, RED]
    x = np.arange(len(names))

    for ax, metric, ylabel, title in [
        (axes[0], 'theta_std', 'θ_std', 'Functional Diversity'),
        (axes[1], 'shannon_eco', "H'_eco", 'Ecological Diversity'),
    ]:
        means = [scenario_results[n]['equilibrium'].get(f'{metric}_mean', 0) for n in names]
        stds  = [scenario_results[n]['equilibrium'].get(f'{metric}_std', 0) for n in names]
        bars = ax.bar(x, means, color=colors, alpha=.85, ec='white', lw=.5, width=.55, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt='none', color='#374151',
                    capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
        for xi, (m, s) in enumerate(zip(means, stds)):
            ax.text(xi, m + s + .003, f'{m:.3f}', ha='center', va='bottom',
                    fontsize=8.5, fontweight='bold', color=colors[xi])
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(ylabel); ax.set_title(title, fontweight='bold')
        ax.set_ylim(0, max(means)*1.35 if means else 1)

    # Annotate winner
    axes[0].annotate('★ Highest', xy=(1, means[1] if 'means' in dir() else .15),
                     xytext=(2.2, max(means)*1.1 if means else .18),
                     arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.3),
                     fontsize=8, color=GREEN, fontweight='bold')

    fig.suptitle('Hydrosphere Comparison: Shallow Ponds Win on Functional Diversity',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig3_scenario_bars.png'))
    plt.close(fig)
    print('  fig3_scenario_bars.png')


def fig_mechanism(sweep_results, d_vals, h_vals):
    """Fig 4: Mechanism — peak diversity sits at the lowest mixing values."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor('white')

    # A: θ_std vs D
    ax = axes[0]
    h_show = [0.2, 0.4, 0.6, 0.8, 1.0]
    pal = ['#cbd5e1', '#94a3b8', AMBER, GREEN, CYAN]
    for h_val, col in zip(h_show, pal):
        pts = sorted([(r['mixing_rate'], r['theta_std_mean'], r['theta_std_std'])
                       for r in sweep_results if abs(r['heterogeneity'] - h_val) < .01])
        if not pts:
            continue
        ds, ts, er = zip(*pts)
        lw = 2.3 if h_val >= 0.8 else 1.4
        al = 1.0 if h_val >= 0.6 else .65
        ax.plot(ds, ts, 'o-', color=col, lw=lw, ms=4, alpha=al, label=f'h={h_val:.1f}')
        if h_val == 0.8:
            ax.fill_between(ds, [t-e for t,e in zip(ts,er)],
                            [t+e for t,e in zip(ts,er)], color=col, alpha=.15)
    ax.set_xlabel('Mixing Rate D')
    ax.set_ylabel('θ_std')
    ax.set_title('A  Mixing reduces functional diversity', fontweight='bold', loc='left')
    ax.legend(title='h', fontsize=8, title_fontsize=8)
    ax.text(.53, .91, 'Peak theta_std always sits\nat the lowest mixing end',
            transform=ax.transAxes, fontsize=8, color=DARK,
            bbox=dict(boxstyle='round', fc='#fef3c7', ec='#fcd34d'))

    # B: θ_std vs h
    ax2 = axes[1]
    d_show = [0.0, 0.02, 0.15, 0.50, 0.80]
    d_lab = ['D=0', 'D=0.02', 'D=0.15', 'D=0.50', 'D=0.80']
    d_pal = [CYAN, GREEN, AMBER, BLUE, RED]
    for d_val, col, lab in zip(d_show, d_pal, d_lab):
        pts = sorted([(r['heterogeneity'], r['theta_std_mean'], r['theta_std_std'])
                       for r in sweep_results if abs(r['mixing_rate'] - d_val) < .005])
        if not pts:
            continue
        hs, ts, er = zip(*pts)
        lw = 2.3 if d_val <= 0.02 else 1.4
        ax2.plot(hs, ts, 's-', color=col, lw=lw, ms=4, label=lab)
        if d_val == 0.0:
            ax2.fill_between(hs, [t-e for t,e in zip(ts,er)],
                             [t+e for t,e in zip(ts,er)], color=col, alpha=.15)
    ax2.set_xlabel('Heterogeneity h')
    ax2.set_ylabel('θ_std')
    ax2.set_title('B  Heterogeneity drives diversity', fontweight='bold', loc='left')
    ax2.legend(fontsize=7.8)

    fig.suptitle('Mechanism: Isolation + Heterogeneity Maximises Functional Diversity',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig4_mechanism.png'))
    plt.close(fig)
    print('  fig4_mechanism.png')


def fig_timeseries(scenario_results):
    """Fig 5: Time series for 4 scenarios."""
    fig = plt.figure(figsize=(11, 4.5))
    fig.patch.set_facecolor('white')
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=.38, width_ratios=[2, 2, 1.2])

    sc_col = {n: s['color'] for n, s in SCENARIOS.items()}
    sc_lab = {n: s['label'] for n, s in SCENARIOS.items()}

    # A: θ_std
    ax = fig.add_subplot(gs[0])
    for n in SCENARIOS:
        h = scenario_results[n]['history']
        t = [r['tick'] for r in h]
        ts = [r['theta_std'] for r in h]
        sm = np.convolve(ts, np.ones(15)/15, mode='valid')
        ax.plot(t[7:-7], sm, color=sc_col[n], lw=2, label=sc_lab[n])
    ax.set_xlabel('Tick'); ax.set_ylabel('θ_std')
    ax.set_title('A  Functional diversity', fontweight='bold', loc='left')
    ax.legend(fontsize=7.5)

    # B: H'_eco
    ax2 = fig.add_subplot(gs[1])
    for n in SCENARIOS:
        h = scenario_results[n]['history']
        t = [r['tick'] for r in h]
        he = [r['shannon_eco'] for r in h]
        sm = np.convolve(he, np.ones(15)/15, mode='valid')
        ax2.plot(t[7:-7], sm, color=sc_col[n], lw=2, label=sc_lab[n])
    ax2.set_xlabel('Tick'); ax2.set_ylabel("H'_eco")
    ax2.set_title("B  Ecological H'", fontweight='bold', loc='left')
    ax2.legend(fontsize=7.5)

    # C: Diversity space
    ax3 = fig.add_subplot(gs[2])
    for n in SCENARIOS:
        h = scenario_results[n]['history']
        last = h[-100:]
        ts = np.mean([r['theta_std'] for r in last])
        he = np.mean([r['shannon_eco'] for r in last])
        ax3.scatter(ts, he, s=160, color=sc_col[n], ec='white', lw=1.2, zorder=4)
        ax3.annotate(sc_lab[n], (ts, he), xytext=(ts+.004, he+.04),
                     fontsize=7.5, color=sc_col[n], fontweight='bold')
    ax3.set_xlabel('θ_std'); ax3.set_ylabel("H'_eco")
    ax3.set_title('C  Diversity space', fontweight='bold', loc='left')

    fig.suptitle('Dynamics and Equilibrium across Hydrosphere Types',
                 fontsize=11, fontweight='bold')
    plt.savefig(os.path.join(FIGURES, 'fig5_timeseries.png'))
    plt.close(fig)
    print('  fig5_timeseries.png')


def fig_sensitivity(sensitivity_results):
    """Fig 6: Niche-width sensitivity (β = 10, 20, 40)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    fig.patch.set_facecolor('white')

    pal = {10.0: '#94a3b8', 20.0: GREEN, 40.0: DARK}
    styles = {10.0: '--', 20.0: '-', 40.0: ':'}

    # A: θ_std vs D
    ax = axes[0]
    for beta, rows in sensitivity_results.items():
        pts = sorted([(r['mixing_rate'], r['theta_std_mean'], r['theta_std_std']) for r in rows])
        ds, ts, er = zip(*pts)
        lw = 2.5 if beta == 20 else 1.6
        ax.plot(ds, ts, 'o', color=pal[beta], lw=lw, ms=4,
                ls=styles[beta], label=f'β={int(beta)}')
        ax.fill_between(ds, [t-e for t,e in zip(ts,er)],
                        [t+e for t,e in zip(ts,er)], color=pal[beta], alpha=.1)
    ax.set_xlabel('Mixing Rate D')
    ax.set_ylabel('θ_std')
    ax.set_title('A  θ_std vs mixing (h=0.8)', fontweight='bold', loc='left')
    ax.legend(fontsize=9)
    ax.text(.5, .92, 'Lowest mixing remains optimal\nacross all beta values',
            transform=ax.transAxes, fontsize=8, ha='center', color=DARK,
            bbox=dict(boxstyle='round', fc='#f0fdf4', ec='#86efac'))

    # B: H'_eco vs D
    ax2 = axes[1]
    for beta, rows in sensitivity_results.items():
        pts = sorted([(r['mixing_rate'], r['shannon_eco_mean'], r['shannon_eco_std']) for r in rows])
        ds, he, er = zip(*pts)
        lw = 2.5 if beta == 20 else 1.6
        ax2.plot(ds, he, 's', color=pal[beta], lw=lw, ms=4,
                 ls=styles[beta], label=f'β={int(beta)}')
        ax2.fill_between(ds, [t-e for t,e in zip(he,er)],
                         [t+e for t,e in zip(he,er)], color=pal[beta], alpha=.1)
    ax2.set_xlabel('Mixing Rate D')
    ax2.set_ylabel("H'_eco")
    ax2.set_title("B  H'_eco vs mixing (h=0.8)", fontweight='bold', loc='left')
    ax2.legend(fontsize=9)

    fig.suptitle('Sensitivity Analysis: Results Robust to Niche Width (β)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig6_sensitivity.png'))
    plt.close(fig)
    print('  fig6_sensitivity.png')


def fig_environment_and_grids(scenario_results):
    """Fig 7: Environment maps + population theta maps."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    theta_cmap = LinearSegmentedColormap.from_list('t',
        [(0, '#1e50dc'), (.2, '#00bec8'), (.5, '#32dc96'), (.8, '#dcb41e'), (1, '#dc3232')])

    for j, (name, spec) in enumerate(SCENARIOS.items()):
        snap = scenario_results[name]['snapshot']
        N = snap['N']

        # Top: theta map
        theta = snap['theta'].reshape(N, N)
        grid = snap['grid'].reshape(N, N)
        masked = np.ma.array(theta, mask=(grid < 0))
        axes[0, j].imshow(masked, cmap=theta_cmap, vmin=0, vmax=1, origin='lower')
        axes[0, j].set_title(f"{spec['label']}\n(θ map)", fontsize=9)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])

        # Bottom: environment
        env = snap['env_map'].reshape(N, N)
        axes[1, j].imshow(env, cmap='RdYlBu_r', vmin=0, vmax=1, origin='lower')
        occupied = (grid >= 0).astype(float) * .3
        axes[1, j].imshow(occupied, cmap='Greens', alpha=.4, origin='lower')
        axes[1, j].set_title('Environment', fontsize=9)
        axes[1, j].set_xticks([]); axes[1, j].set_yticks([])

    fig.suptitle('Niche Adaptation: Replicator θ (top) vs Environment E (bottom)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig7_grids.png'))
    plt.close(fig)
    print('  fig7_grids.png')


def fig_key_result(sweep_results):
    """Fig 8: Key result summary — θ_std vs D at h=0.8."""
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor('white')

    pts = sorted([(r['mixing_rate'], r['theta_std_mean'], r['theta_std_std'])
                   for r in sweep_results if abs(r['heterogeneity'] - 0.8) < .01])
    if not pts:
        plt.close(fig); return
    ds, ts, er = zip(*pts)

    ax.fill_between(ds, [t-e for t,e in zip(ts,er)],
                    [t+e for t,e in zip(ts,er)], color=GREEN, alpha=.18, label='±1σ')
    ax.plot(ds, ts, 'o-', color=GREEN, lw=2.8, ms=7,
            mfc='white', mew=2.2, label='Mean θ_std (h=0.8)')

    sc = {'Ponds': (.02, GREEN), 'Tidal': (.15, AMBER), 'Deep Ocean': (.50, BLUE)}
    for lab, (d, col) in sc.items():
        ts_i = np.interp(d, list(ds), list(ts))
        ax.axvline(d, color=col, lw=1.2, ls=':', alpha=.7)
        ax.scatter([d], [ts_i], s=100, color=col, zorder=5, ec='white', lw=1.5)
        ax.annotate(lab, (d, ts_i), xytext=(d+.02, ts_i+.008),
                    fontsize=9, color=col, fontweight='bold')

    ax.set_xlabel('Mixing Rate D (0=isolated, 1=well-mixed)')
    ax.set_ylabel('Functional Diversity θ_std')
    ax.set_title('Core Result: Isolation Maximises Replicator Diversity (h=0.8)',
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.text(.03, .18, 'Shallow ponds (low D) retain\nhighest functional diversity.\n'
            'No Goldilocks optimum.', transform=ax.transAxes, fontsize=9.5, color=DARK,
            bbox=dict(boxstyle='round,pad=.4', fc='#f0fdf4', ec='#86efac', lw=1.2))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig8_key_result.png'))
    plt.close(fig)
    print('  fig8_key_result.png')


def fig_transport_comparison(global_results, local_results, d_vals):
    """Fig 9: Global vs local transport at h=0.8 with effect sizes."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor('white')

    # Panel A: theta_std vs D for both transport modes
    ax = axes[0]
    for results, label, color, marker in [
        (global_results, 'Global teleportation', RED, 'o'),
        (local_results, 'Local diffusion', CYAN, 's'),
    ]:
        pts = sorted([(r['mixing_rate'], r['theta_std_mean'], r['theta_std_std'])
                       for r in results if abs(r['heterogeneity'] - 0.8) < 0.01])
        if not pts:
            continue
        ds, ts, er = zip(*pts)
        ax.plot(ds, ts, f'{marker}-', color=color, lw=2.2, ms=6,
                mfc='white', mew=2, label=label)
        ax.fill_between(ds, [t-e for t,e in zip(ts,er)],
                        [t+e for t,e in zip(ts,er)], color=color, alpha=0.12)

    ax.set_xlabel('Mixing Rate D')
    ax.set_ylabel('theta_std')
    ax.set_title('A  Transport model comparison (h=0.8)', fontweight='bold', loc='left')
    ax.legend(fontsize=9)

    g_pts = {r['mixing_rate']: r for r in global_results if abs(r['heterogeneity'] - 0.8) < 0.01}
    l_pts = {r['mixing_rate']: r for r in local_results if abs(r['heterogeneity'] - 0.8) < 0.01}
    if 0.0 in g_pts and 0.8 in g_pts and 0.0 in l_pts and 0.8 in l_pts:
        g_ratio = g_pts[0.0]['theta_std_mean'] / max(g_pts[0.8]['theta_std_mean'], 1e-10)
        l_ratio = l_pts[0.0]['theta_std_mean'] / max(l_pts[0.8]['theta_std_mean'], 1e-10)
        ax.text(0.40, 0.50, f'Global: {g_ratio:.2f}x decline\nLocal: {l_ratio:.2f}x decline',
                transform=ax.transAxes, fontsize=9, color=DARK,
                bbox=dict(boxstyle='round', fc='#fef3c7', ec='#fcd34d'))

    # Panel B: Cohen's d bar chart
    ax2 = axes[1]
    global_stats = compute_pairwise_stats(global_results, 0.8)
    local_stats = compute_pairwise_stats(local_results, 0.8)

    if global_stats:
        ds_g = [s['mixing_rate'] for s in global_stats if s['mixing_rate'] > 0]
        cd_g = [s['cohen_d'] for s in global_stats if s['mixing_rate'] > 0]
        x_g = np.arange(len(ds_g))
        ax2.bar(x_g - 0.18, cd_g, width=0.35, color=RED, alpha=0.7,
                label="Global (Cohen's d)")
    if local_stats:
        ds_l = [s['mixing_rate'] for s in local_stats if s['mixing_rate'] > 0]
        cd_l = [s['cohen_d'] for s in local_stats if s['mixing_rate'] > 0]
        x_l = np.arange(len(ds_l))
        ax2.bar(x_l + 0.18, cd_l, width=0.35, color=CYAN, alpha=0.7,
                label="Local (Cohen's d)")
        # Add significance stars
        for i, s in enumerate([s for s in local_stats if s['mixing_rate'] > 0]):
            ax2.text(i + 0.18, s['cohen_d'] + 0.1, _sig_stars(s['p_value']),
                     ha='center', fontsize=7.5, fontweight='bold', color=CYAN)

    if global_stats:
        for i, s in enumerate([s for s in global_stats if s['mixing_rate'] > 0]):
            ax2.text(i - 0.18, s['cohen_d'] + 0.1, _sig_stars(s['p_value']),
                     ha='center', fontsize=7.5, fontweight='bold', color=RED)
        ax2.set_xticks(np.arange(len(ds_g)))
        ax2.set_xticklabels([f'{d:.2f}' for d in ds_g], fontsize=8)

    ax2.axhline(0.8, color=GRAY, ls='--', lw=1, alpha=0.5)
    ax2.text(len(ds_g) - 0.5 if global_stats else 0.5, 0.85,
             'large effect', fontsize=7.5, color=GRAY)
    ax2.set_xlabel('Mixing Rate D')
    ax2.set_ylabel("Cohen's d (vs D=0)")
    ax2.set_title('B  Effect sizes (vs D=0)', fontweight='bold', loc='left')
    ax2.legend(fontsize=8)

    fig.suptitle('Transport Model Matters: Global Teleportation vs Local Diffusion',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig9_transport_comparison.png'))
    plt.close(fig)
    print('  fig9_transport_comparison.png')


def fig_colonization_rescue(colonization_results, local_results, d_vals, h_vals):
    """Fig 10: Colonization-rescue mechanism."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor('white')

    # Panel A: theta_std vs D at h=0.8
    ax = axes[0]
    for results, label, color, marker in [
        (local_results, 'Local (default abiogenesis)', CYAN, 's'),
        (colonization_results, 'Local (low abiogenesis)', GREEN, 'D'),
    ]:
        pts = sorted([(r['mixing_rate'], r['theta_std_mean'], r['theta_std_std'])
                       for r in results if abs(r['heterogeneity'] - 0.8) < 0.01])
        if not pts:
            continue
        ds, ts, er = zip(*pts)
        ax.plot(ds, ts, f'{marker}-', color=color, lw=2.2, ms=6,
                mfc='white', mew=2, label=label)
        ax.fill_between(ds, [t-e for t,e in zip(ts,er)],
                        [t+e for t,e in zip(ts,er)], color=color, alpha=0.12)

    # Check for interior peak (Goldilocks)
    col_pts = sorted([(r['mixing_rate'], r['theta_std_mean'])
                       for r in colonization_results if abs(r['heterogeneity'] - 0.8) < 0.01])
    goldilocks_found = False
    if col_pts:
        peak = max(col_pts, key=lambda x: x[1])
        if peak[0] > 0.005:
            goldilocks_found = True
            ax.axvline(peak[0], color=GREEN, ls=':', lw=1.5, alpha=0.7)
            ax.annotate(f'Peak at D={peak[0]:.2f}', xy=(peak[0], peak[1]),
                        xytext=(peak[0]+0.08, peak[1]+0.008),
                        arrowprops=dict(arrowstyle='->', color=GREEN),
                        fontsize=9, color=GREEN, fontweight='bold')

    if not goldilocks_found:
        ax.text(0.03, 0.18, 'No intermediate optimum:\nisolation still wins',
                transform=ax.transAxes, fontsize=8.5, color=DARK,
                bbox=dict(boxstyle='round', fc='#fef3c7', ec='#fcd34d'))

    ax.set_xlabel('Mixing Rate D')
    ax.set_ylabel('theta_std')
    ax.set_title('A  Colonization-limited regime (h=0.8)', fontweight='bold', loc='left')
    ax.legend(fontsize=8)

    # Panel B: Heatmap of colonization sweep
    ax2 = axes[1]
    nd, nh = len(d_vals), len(h_vals)
    mat = np.zeros((nh, nd))
    for r in colonization_results:
        try:
            di = d_vals.index(r['mixing_rate'])
            hi = h_vals.index(r['heterogeneity'])
            mat[hi, di] = r.get('theta_std_mean', 0)
        except ValueError:
            continue

    im = ax2.imshow(mat, origin='lower', aspect='auto',
                    extent=[-.5, nd-.5, -.5, nh-.5],
                    cmap='YlOrRd', interpolation='bilinear')
    ax2.set_xticks(range(nd))
    ax2.set_xticklabels([f'{v:.2f}' for v in d_vals], rotation=45, ha='right', fontsize=8)
    ax2.set_yticks(range(nh))
    ax2.set_yticklabels([f'{v:.1f}' for v in h_vals], fontsize=8)
    ax2.set_xlabel('Mixing Rate D')
    ax2.set_ylabel('Heterogeneity h')
    ax2.set_title('B  Colonization-limited theta_std', fontweight='bold', loc='left')
    div = make_axes_locatable(ax2)
    cax = div.append_axes('right', size='4%', pad=0.06)
    plt.colorbar(im, cax=cax).set_label('theta_std', fontsize=8)

    # Find overall best in colonization sweep
    if colonization_results:
        best = max(colonization_results, key=lambda r: r.get('theta_std_mean', 0))
        try:
            bdi = d_vals.index(best['mixing_rate'])
            bhi = h_vals.index(best['heterogeneity'])
            ax2.plot(bdi, bhi, '*', ms=14, mfc='white', mec=GREEN, mew=2, zorder=5)
            ax2.annotate(f"Best: D={best['mixing_rate']:.2f}",
                        (bdi, bhi), xytext=(bdi+0.5, bhi+0.3),
                        fontsize=8, color=GREEN, fontweight='bold')
        except ValueError:
            pass

    fig.suptitle('Colonization-Limited Regime: Testing for Intermediate Dispersal Optimum',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig10_colonization_rescue.png'))
    plt.close(fig)
    print('  fig10_colonization_rescue.png')


def fig_key_result_with_stats(sweep_results, local_results, n_seeds=5):
    """Fig 11: Updated key result with both transport modes and statistical tests."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white')

    g_pts = sorted([(r['mixing_rate'], r['theta_std_mean'], r['theta_std_std'])
                     for r in sweep_results if abs(r['heterogeneity'] - 0.8) < 0.01])
    l_pts = sorted([(r['mixing_rate'], r['theta_std_mean'], r['theta_std_std'])
                     for r in local_results if abs(r['heterogeneity'] - 0.8) < 0.01])

    if g_pts:
        ds, ts, er = zip(*g_pts)
        ax.fill_between(ds, [t-e for t,e in zip(ts,er)],
                        [t+e for t,e in zip(ts,er)], color=RED, alpha=0.1)
        ax.plot(ds, ts, 'o-', color=RED, lw=2.5, ms=7,
                mfc='white', mew=2, label='Global teleportation')

    if l_pts:
        ds, ts, er = zip(*l_pts)
        ax.fill_between(ds, [t-e for t,e in zip(ts,er)],
                        [t+e for t,e in zip(ts,er)], color=CYAN, alpha=0.1)
        ax.plot(ds, ts, 's-', color=CYAN, lw=2.5, ms=7,
                mfc='white', mew=2, label='Local diffusion')

    # Add statistical annotations for global
    global_stats = compute_pairwise_stats(sweep_results, 0.8)
    for s in global_stats:
        if s['mixing_rate'] in [0.05, 0.15, 0.50, 0.80]:
            stars = _sig_stars(s['p_value'])
            ax.annotate(f"{stars}\nd={s['cohen_d']:.1f}",
                        xy=(s['mixing_rate'], s['theta_std_mean']),
                        xytext=(s['mixing_rate'], s['theta_std_mean'] - 0.022),
                        fontsize=7, ha='center', color=RED, fontweight='bold')

    ax.set_xlabel('Mixing Rate D (0 = isolated, 1 = well-mixed)')
    ax.set_ylabel('Functional Diversity theta_std')
    ax.set_title('Key Result: Transport Model Determines Whether\n'
                 'Connectivity Suppresses Diversity (h = 0.8)',
                 fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')

    if g_pts and l_pts:
        g_ratio = g_pts[0][1] / max(g_pts[-1][1], 1e-10)
        l_ratio = l_pts[0][1] / max(l_pts[-1][1], 1e-10)
        stats_text = (f'Global: {g_ratio:.2f}x suppression (D=0 vs D=0.8)\n'
                      f'Local: {l_ratio:.2f}x suppression\n'
                      f'* p<0.05  ** p<0.01  *** p<0.001')
        ax.text(0.03, 0.38, stats_text, transform=ax.transAxes, fontsize=8.5,
                va='top', color=DARK,
                bbox=dict(boxstyle='round,pad=0.4', fc='#f8fafc', ec='#94a3b8', lw=1))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES, 'fig11_key_result_stats.png'))
    plt.close(fig)
    print('  fig11_key_result_stats.png')


# ============================================================
# UTILITIES
# ============================================================

def _save_history(history, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if not history:
        return
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)


def print_summary(scenario_results, sweep_results, d_vals, h_vals):
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\n  Scenario Comparison:")
    print(f"  {'Scenario':<20} {'θ_std':>10} {'H_eco':>10} {'Pop':>8}")
    print("  " + "-" * 52)
    for n in SCENARIOS:
        eq = scenario_results[n]['equilibrium']
        ts = eq.get('theta_std_mean', 0)
        he = eq.get('shannon_eco_mean', 0)
        po = eq.get('population_mean', 0)
        print(f"  {SCENARIOS[n]['label']:<20} {ts:10.4f} {he:10.3f} {po:8.0f}")

    if sweep_results:
        best = max(sweep_results, key=lambda r: r.get('theta_std_mean', 0))
        print(f"\n  Best θ_std: D={best['mixing_rate']:.2f} h={best['heterogeneity']:.1f}"
              f" → θ_std={best['theta_std_mean']:.4f}")

        # Goldilocks check at h=0.8
        h08 = sorted([(r['mixing_rate'], r['theta_std_mean'])
                       for r in sweep_results if abs(r['heterogeneity'] - .8) < .01])
        if h08:
            peak_d = max(h08, key=lambda x: x[1])
            print(f"  At h=0.8: peak θ_std at D={peak_d[0]:.2f} (θ_std={peak_d[1]:.4f})")
            if peak_d[0] <= 0.02:
                print("  → NO Goldilocks effect: isolation wins")
            else:
                print(f"  → Possible Goldilocks at D={peak_d[0]:.2f}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Hydrosphere Replicator Experiment')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--sweep-only', action='store_true')
    parser.add_argument('--scenarios-only', action='store_true')
    args = parser.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    setup_style()

    print("=" * 60)
    print("HYDROSPHERE REPLICATOR DIVERSITY EXPERIMENT")
    print("=" * 60)

    if args.quick:
        print("\n[QUICK MODE]")
        sc_grid, sc_ticks = 30, 400
        sw_grid, sw_ticks = 30, 300
        d_vals, h_vals, seeds = QUICK_D, QUICK_H, QUICK_SEEDS
    else:
        sc_grid, sc_ticks = 40, 800
        sw_grid, sw_ticks = 40, 600
        d_vals, h_vals, seeds = SWEEP_D, SWEEP_H, SEEDS

    t_start = time.time()
    scenario_results = {}
    sweep_results = []
    sensitivity_results = {}
    local_results = []
    colonization_results = []

    # Phase 1
    if not args.sweep_only:
        scenario_results = run_scenarios(sc_grid, sc_ticks, seeds)

    # Phase 2
    if not args.scenarios_only:
        sweep_results = run_sweep(d_vals, h_vals, seeds, sw_grid, sw_ticks)

    # Phase 3: Sensitivity
    if not args.scenarios_only and not args.quick:
        sensitivity_results = run_sensitivity(seeds, sw_grid, sw_ticks)

    # Phase 6: Local diffusion sweep
    if not args.scenarios_only:
        local_results = run_local_sweep(d_vals, h_vals, seeds, sw_grid,
                                        n_ticks=sw_ticks + 200)

    # Phase 7: Colonization-limited sweep
    if not args.scenarios_only and not args.quick:
        colonization_results = run_colonization_sweep(
            d_vals, h_vals, seeds, sw_grid, n_ticks=sw_ticks + 200)

    # Phase 4: Figures
    print("\n" + "=" * 60)
    print("PHASE 4: Generating Figures")
    print("=" * 60)

    fig_model_schematic()
    if sweep_results:
        fig_heatmaps(sweep_results, d_vals, h_vals)
        fig_mechanism(sweep_results, d_vals, h_vals)
        fig_key_result(sweep_results)
    if scenario_results:
        fig_scenario_bars(scenario_results)
        fig_timeseries(scenario_results)
        fig_environment_and_grids(scenario_results)
    if sensitivity_results:
        fig_sensitivity(sensitivity_results)
    if sweep_results and local_results:
        fig_transport_comparison(sweep_results, local_results, d_vals)
        fig_key_result_with_stats(sweep_results, local_results)
    if colonization_results and local_results:
        fig_colonization_rescue(colonization_results, local_results, d_vals, h_vals)

    # Phase 5: Summary
    if scenario_results:
        print_summary(scenario_results, sweep_results, d_vals, h_vals)

    # Print local vs global comparison
    if sweep_results and local_results:
        print("\n  Transport Model Comparison (h=0.8):")
        for label, res in [('Global', sweep_results), ('Local', local_results)]:
            pts = {r['mixing_rate']: r for r in res if abs(r['heterogeneity'] - 0.8) < 0.01}
            if 0.0 in pts and 0.8 in pts:
                d0 = pts[0.0]['theta_std_mean']
                d8 = pts[0.8]['theta_std_mean']
                print(f"    {label:8s}: D=0 theta_std={d0:.4f}, D=0.8={d8:.4f}, "
                      f"ratio={d0/max(d8,1e-10):.2f}x")

    # Print colonization results
    if colonization_results:
        col_h08 = sorted([(r['mixing_rate'], r['theta_std_mean'])
                           for r in colonization_results
                           if abs(r['heterogeneity'] - 0.8) < 0.01])
        if col_h08:
            peak = max(col_h08, key=lambda x: x[1])
            print(f"\n  Colonization-limited (h=0.8): peak theta_std at D={peak[0]:.2f} "
                  f"(theta_std={peak[1]:.4f})")
            if peak[0] > 0.005:
                print("    -> GOLDILOCKS OPTIMUM at intermediate dispersal!")
            else:
                print("    -> No Goldilocks: isolation still wins")

    total = time.time() - t_start
    print(f"\nTotal time: {total:.0f}s ({total/60:.1f}min)")

    # Save metadata
    meta = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_time_s': total,
        'grid_sizes': {'scenario': sc_grid, 'sweep': sw_grid},
        'ticks': {'scenario': sc_ticks, 'sweep': sw_ticks},
        'seeds': seeds,
        'sweep_d': d_vals,
        'sweep_h': h_vals,
        'mixing_mode': 'global',
        'local_sweep_included': bool(local_results),
        'colonization_sweep_included': bool(colonization_results),
        'fixes_applied': [
            'theta-binned ecological H (not lineage-based)',
            'global mixing (teleport, not local diffusion)',
            'local diffusion full sweep added',
            'colonization-limited sweep added',
            'statistical tests (Welch t-test, Cohen d) added',
            'uniform abiogenesis (not E-dependent)',
            'unique IDs for all founders',
            'proper across-seed statistics',
        ]
    }
    with open(os.path.join(RESULTS, 'experiment_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    summary = build_experiment_summary(meta, scenario_results, sweep_results, sensitivity_results)
    write_experiment_summary(BASE, summary)


if __name__ == '__main__':
    main()
