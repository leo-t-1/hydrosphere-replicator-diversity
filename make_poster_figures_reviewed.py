#!/usr/bin/env python3
"""
Reviewed poster figure pack generated from the current verified results.

This pack prioritizes:
  - low annotation density,
  - external legends when practical,
  - larger margins / spacing,
  - wording aligned with the current model and summary.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
OUT = BASE / "poster_figures_reviewed_v1"
OUT.mkdir(exist_ok=True)

BLUE = "#2563eb"
GREEN = "#059669"
AMBER = "#d97706"
RED = "#dc2626"
CYAN = "#0891b2"
DARK = "#1e293b"
SLATE = "#64748b"
LIGHT = "#94a3b8"
GRID = "#dbe4ee"

SCENARIOS = {
    "deep_ocean": {"label": "Deep Ocean", "color": BLUE, "D": 0.50, "h": 0.10},
    "pond_network": {"label": "Shallow Ponds", "color": GREEN, "D": 0.02, "h": 0.80},
    "tidal_zone": {"label": "Tidal Zone", "color": AMBER, "D": 0.15, "h": 0.70},
    "hydrothermal_vent": {"label": "Hydrothermal Vents", "color": RED, "D": 0.08, "h": 0.60},
}


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.20,
    "figure.dpi": 180,
})


def load_rows(path: Path):
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            parsed = {}
            for key, value in row.items():
                try:
                    parsed[key] = float(value)
                except (TypeError, ValueError):
                    parsed[key] = value
            rows.append(parsed)
    return rows


summary = json.loads((RESULTS / "experiment_summary.json").read_text())
sweep = load_rows(RESULTS / "sweep_results.csv")
sensitivity = load_rows(RESULTS / "sensitivity_niche_width.csv")
scenario_histories = {name: load_rows(RESULTS / f"scenario_{name}.csv") for name in SCENARIOS}

MIXING_VALS = sorted({row["mixing_rate"] for row in sweep})
HETERO_VALS = sorted({row["heterogeneity"] for row in sweep})


def theta_cmap():
    return LinearSegmentedColormap.from_list(
        "theta",
        [
            (0.0, "#1e50dc"),
            (0.2, "#00bec8"),
            (0.5, "#32dc96"),
            (0.8, "#dcb41e"),
            (1.0, "#dc3232"),
        ],
        N=256,
    )


def make_matrix(metric: str):
    mat = np.zeros((len(HETERO_VALS), len(MIXING_VALS)))
    for row in sweep:
        hi = HETERO_VALS.index(row["heterogeneity"])
        di = MIXING_VALS.index(row["mixing_rate"])
        mat[hi, di] = row[metric]
    return mat


def save(fig, name: str):
    fig.savefig(OUT / name)
    plt.close(fig)


def scenario_metric(name: str, metric: str):
    block = summary["scenarios"]["metrics"][name]
    return block[f"{metric}_mean"], block[f"{metric}_std"]


def fig01_concepts():
    fig = plt.figure(figsize=(12.2, 4.4), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig)

    ax = fig.add_subplot(gs[0])
    env = np.linspace(0, 1, 300)
    for beta, color, lw in [(10, LIGHT, 1.6), (20, CYAN, 2.6), (40, DARK, 1.6)]:
        ax.plot(env, np.exp(-beta * (env - 0.5) ** 2), color=color, lw=lw, label=f"beta={beta}")
    ax.axvline(0.5, color=SLATE, ls="--", lw=1.0)
    ax.set_title("A  Niche-matching penalty", loc="left", fontweight="bold")
    ax.set_xlabel("Environment E")
    ax.set_ylabel("exp(-beta(E-theta)^2)")
    ax.legend(frameon=True, framealpha=0.95)

    ax2 = fig.add_subplot(gs[1])
    ax2.set_title("B  Metric intuition", loc="left", fontweight="bold")
    x = np.linspace(0, 1, 300)
    low = 0.90 * np.exp(-95 * (x - 0.5) ** 2) + 0.02
    high = (
        0.30 * np.exp(-85 * (x - 0.20) ** 2)
        + 0.26 * np.exp(-85 * (x - 0.42) ** 2)
        + 0.28 * np.exp(-85 * (x - 0.65) ** 2)
        + 0.22 * np.exp(-85 * (x - 0.84) ** 2)
        + 0.02
    )
    ax2.fill_between(x, low, color=BLUE, alpha=0.18)
    ax2.plot(x, low, color=BLUE, lw=1.8, label="low theta_std, low H'")
    ax2.fill_between(x, high, color=GREEN, alpha=0.18)
    ax2.plot(x, high, color=GREEN, lw=1.8, label="high theta_std, high H'")
    ax2.set_xlabel("theta")
    ax2.set_ylabel("Relative abundance")
    ax2.legend(frameon=True, framealpha=0.95, loc="upper right")

    ax3 = fig.add_subplot(gs[2])
    ax3.set_title("C  Definitions", loc="left", fontweight="bold")
    ax3.axis("off")
    lines = [
        "theta: replicator niche optimum on [0, 1]",
        "beta: selection strength / niche narrowness",
        "",
        "theta_std = std(theta_i over occupied cells)",
        "Functional diversity = spread of ecological strategies",
        "",
        "H' = -sum_k p_k ln(p_k)",
        "with p_k from 20 theta bins",
        "Ecological diversity = breadth and evenness of occupied niche space",
        "",
        "fitness = r * exp(-beta(E-theta)^2) * min(1, R)",
    ]
    y = 0.98
    for line in lines:
        size = 10 if line else 6
        color = DARK if line else "white"
        ax3.text(0.0, y, line, transform=ax3.transAxes, fontsize=size, color=color, va="top")
        y -= 0.095 if line else 0.06

    fig.suptitle("Concept Figure: Traits, Metrics, and Model Mathematics", fontsize=12.5, fontweight="bold")
    save(fig, "fig01_concepts.png")


def fig02_phase_maps():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.9), sharey=True)
    fig.subplots_adjust(top=0.83, wspace=0.26)
    handles = []
    for name, spec in SCENARIOS.items():
        handles.append(
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                   markeredgecolor=spec["color"], markeredgewidth=2.0, markersize=8,
                   label=spec["label"])
        )

    configs = [
        ("theta_std_mean", "Functional diversity (theta_std)", "YlOrRd", "theta_std"),
        ("shannon_eco_mean", "Ecological diversity (H')", "viridis", "H'"),
    ]
    for ax, (metric, title, cmap, cbar_label) in zip(axes, configs):
        mat = make_matrix(metric)
        im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap, interpolation="nearest")
        ax.grid(False)
        ax.set_xticks(range(len(MIXING_VALS)))
        ax.set_xticklabels([f"{v:.2f}" for v in MIXING_VALS], rotation=45, ha="right")
        ax.set_yticks(range(len(HETERO_VALS)))
        ax.set_yticklabels([f"{v:.1f}" for v in HETERO_VALS])
        ax.set_xlabel("Mixing rate D")
        ax.set_title(title, fontweight="bold")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.08)
        plt.colorbar(im, cax=cax).set_label(cbar_label)
        for spec in SCENARIOS.values():
            di = min(range(len(MIXING_VALS)), key=lambda i: abs(MIXING_VALS[i] - spec["D"]))
            hi = min(range(len(HETERO_VALS)), key=lambda i: abs(HETERO_VALS[i] - spec["h"]))
            ax.plot(di, hi, "o", ms=9, mfc="white", mec=spec["color"], mew=2.0, zorder=5)

    axes[0].set_ylabel("Heterogeneity h")
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False)
    axes[0].text(
        0.02, 0.98,
        "Best theta_std occurs at D=0.0 for every h slice tested.",
        transform=axes[0].transAxes,
        va="top",
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.25", fc="#fff7ed", ec="#fdba74"),
    )
    save(fig, "fig02_phase_maps.png")


def fig03_h_effects():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)

    for d, color, label in [(0.0, CYAN, "D=0.00"), (0.15, AMBER, "D=0.15"), (0.80, RED, "D=0.80")]:
        rows = sorted([r for r in sweep if abs(r["mixing_rate"] - d) < 1e-9], key=lambda r: r["heterogeneity"])
        hs = np.array([r["heterogeneity"] for r in rows])
        theta_vals = np.array([r["theta_std_mean"] for r in rows])
        theta_stds = np.array([r["theta_std_std"] for r in rows])
        h_vals = np.array([r["shannon_eco_mean"] for r in rows])
        h_stds = np.array([r["shannon_eco_std"] for r in rows])

        axes[0].plot(hs, theta_vals, "o-", color=color, lw=2.2, ms=4.5, label=label)
        axes[0].fill_between(hs, theta_vals - theta_stds, theta_vals + theta_stds, color=color, alpha=0.10)

        axes[1].plot(hs, h_vals, "o-", color=color, lw=2.2, ms=4.5, label=label)
        axes[1].fill_between(hs, h_vals - h_stds, h_vals + h_stds, color=color, alpha=0.10)

    axes[0].set_title("A  Heterogeneity raises theta_std", loc="left", fontweight="bold")
    axes[0].set_xlabel("Heterogeneity h")
    axes[0].set_ylabel("theta_std")
    axes[0].legend(frameon=True, framealpha=0.95, loc="upper left")

    axes[1].set_title("B  Heterogeneity raises ecological H'", loc="left", fontweight="bold")
    axes[1].set_xlabel("Heterogeneity h")
    axes[1].set_ylabel("H'")

    axes[0].text(
        0.03, 0.08,
        "At D=0.0: theta_std increases 0.060 -> 0.207\nfrom h=0.0 to h=1.0 (3.44x).",
        transform=axes[0].transAxes,
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.25", fc="#eff6ff", ec="#93c5fd"),
    )
    save(fig, "fig03_heterogeneity_effects.png")


def fig04_scenarios_and_sensitivity():
    fig = plt.figure(figsize=(14.0, 4.6), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    order = ["deep_ocean", "pond_network", "tidal_zone", "hydrothermal_vent"]
    labels = ["Deep\nOcean", "Shallow\nPonds", "Tidal\nZone", "Hydrothermal\nVents"]
    x = np.arange(len(order))
    colors = [SCENARIOS[name]["color"] for name in order]

    theta_means = [scenario_metric(name, "theta_std")[0] for name in order]
    theta_stds = [scenario_metric(name, "theta_std")[1] for name in order]
    h_means = [scenario_metric(name, "shannon_eco")[0] for name in order]
    h_stds = [scenario_metric(name, "shannon_eco")[1] for name in order]

    axes[0].bar(x, theta_means, color=colors, alpha=0.88, ec="white", lw=0.6)
    axes[0].errorbar(x, theta_means, yerr=theta_stds, fmt="none", color=DARK, capsize=4, lw=1.1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title("A  Scenario theta_std", loc="left", fontweight="bold")
    axes[0].set_ylabel("theta_std")

    axes[1].bar(x, h_means, color=colors, alpha=0.45, ec="white", lw=0.6)
    axes[1].errorbar(x, h_means, yerr=h_stds, fmt="none", color=DARK, capsize=4, lw=1.1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("B  Scenario ecological H'", loc="left", fontweight="bold")
    axes[1].set_ylabel("H'")

    palette = {10.0: LIGHT, 20.0: GREEN, 40.0: DARK}
    styles = {10.0: "--", 20.0: "-", 40.0: ":"}
    for beta in [10.0, 20.0, 40.0]:
        rows = sorted([r for r in sensitivity if abs(r["niche_width"] - beta) < 1e-9], key=lambda r: r["mixing_rate"])
        ds = [r["mixing_rate"] for r in rows]
        vals = [r["theta_std_mean"] for r in rows]
        axes[2].plot(ds, vals, "o", color=palette[beta], ls=styles[beta], lw=2.2 if beta == 20 else 1.6,
                     ms=4.5, label=f"beta={int(beta)}")
    axes[2].set_title("C  Sensitivity at h=0.8", loc="left", fontweight="bold")
    axes[2].set_xlabel("Mixing rate D")
    axes[2].set_ylabel("theta_std")
    axes[2].legend(frameon=True, framealpha=0.95)
    axes[2].text(
        0.03, 0.84,
        "All tested beta values still peak\nat the minimum sampled mixing.",
        transform=axes[2].transAxes,
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.25", fc="#f0fdf4", ec="#86efac"),
    )

    save(fig, "fig04_scenarios_and_sensitivity.png")


def fig05_dynamics():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.7), constrained_layout=True)

    for name, spec in SCENARIOS.items():
        rows = scenario_histories[name]
        t = np.array([r["tick"] for r in rows])
        theta_vals = np.array([r["theta_std"] for r in rows])
        shannon_vals = np.array([r["shannon_eco"] for r in rows])
        kernel = np.ones(15) / 15
        axes[0].plot(t[7:-7], np.convolve(theta_vals, kernel, mode="valid"), color=spec["color"], lw=2.2, label=spec["label"])
        axes[1].plot(t[7:-7], np.convolve(shannon_vals, kernel, mode="valid"), color=spec["color"], lw=2.2, label=spec["label"])

    axes[0].set_title("A  Functional diversity through time", loc="left", fontweight="bold")
    axes[0].set_xlabel("Tick")
    axes[0].set_ylabel("theta_std")
    axes[0].legend(frameon=True, framealpha=0.95, loc="upper left")

    axes[1].set_title("B  Ecological diversity through time", loc="left", fontweight="bold")
    axes[1].set_xlabel("Tick")
    axes[1].set_ylabel("H'")
    save(fig, "fig05_dynamics.png")


def fig06_core_result():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    rows = sorted([r for r in sweep if abs(r["heterogeneity"] - 0.8) < 1e-9], key=lambda r: r["mixing_rate"])
    ds = np.array([r["mixing_rate"] for r in rows])
    theta_vals = np.array([r["theta_std_mean"] for r in rows])
    theta_stds = np.array([r["theta_std_std"] for r in rows])
    h_vals = np.array([r["shannon_eco_mean"] for r in rows])
    h_stds = np.array([r["shannon_eco_std"] for r in rows])

    axes[0].fill_between(ds, theta_vals - theta_stds, theta_vals + theta_stds, color=GREEN, alpha=0.16)
    axes[0].plot(ds, theta_vals, "o-", color=GREEN, lw=2.8, ms=6, mfc="white", mew=1.8)
    axes[0].set_title("A  Core result at h=0.8", loc="left", fontweight="bold")
    axes[0].set_xlabel("Mixing rate D")
    axes[0].set_ylabel("theta_std")
    axes[0].text(
        0.03, 0.11,
        f"D=0.0 -> D=0.8 drop: {summary['sweep']['h08']['drop_pct_d0_to_d08']:.1f}%.\n"
        "No intermediate optimum in this slice.",
        transform=axes[0].transAxes,
        fontsize=8.4,
        bbox=dict(boxstyle="round,pad=0.26", fc="#f0fdf4", ec="#86efac"),
    )

    axes[1].fill_between(ds, h_vals - h_stds, h_vals + h_stds, color=AMBER, alpha=0.16)
    axes[1].plot(ds, h_vals, "o-", color=AMBER, lw=2.8, ms=6, mfc="white", mew=1.8)
    axes[1].set_title("B  Same slice using ecological H'", loc="left", fontweight="bold")
    axes[1].set_xlabel("Mixing rate D")
    axes[1].set_ylabel("H'")
    save(fig, "fig06_core_result.png")


def write_readme():
    text = """# Reviewed Poster Figure Pack v1

This folder contains a fresh figure set generated from the current saved results.

Design choices:
- larger spacing and padding than the original figure pack,
- minimal in-plot annotation,
- legends moved outside or into unused corners,
- wording aligned with the current verified summary and Python-backed frontend.

Recommended poster order:
1. fig01_concepts.png
2. fig02_phase_maps.png
3. fig06_core_result.png
4. fig03_heterogeneity_effects.png
5. fig04_scenarios_and_sensitivity.png
6. fig05_dynamics.png
"""
    (OUT / "README.md").write_text(text)


def main():
    print("Generating reviewed poster figure pack...")
    fig01_concepts()
    fig02_phase_maps()
    fig03_h_effects()
    fig04_scenarios_and_sensitivity()
    fig05_dynamics()
    fig06_core_result()
    write_readme()
    print(f"Saved reviewed pack to {OUT}")
    for path in sorted(OUT.glob("*")):
        if path.is_file():
            print(f"  {path.name} ({path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
