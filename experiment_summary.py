"""
Helpers for turning experiment outputs into a stable JSON summary.

The goal is to keep manuscript text and README claims tied to the data that
was actually generated, instead of relying on hardcoded numbers.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List


SUMMARY_FILENAME = "experiment_summary.json"

SCENARIO_LABELS = {
    "deep_ocean": "Deep Ocean",
    "pond_network": "Shallow Ponds",
    "tidal_zone": "Tidal Zone",
    "hydrothermal_vent": "Hydrothermal Vents",
}


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den in (None, 0):
        return None
    return float(num / den)


def _drop_pct(start: float | None, end: float | None) -> float | None:
    if start in (None, 0) or end is None:
        return None
    return float(100.0 * (start - end) / start)


def _pick_metrics(row: Dict[str, Any], prefixes: Iterable[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for prefix in prefixes:
        mean_key = f"{prefix}_mean"
        std_key = f"{prefix}_std"
        if mean_key in row:
            out[mean_key] = float(row[mean_key])
        if std_key in row:
            out[std_key] = float(row[std_key])
    return out


def classify_run(meta: Dict[str, Any]) -> str:
    seeds = meta.get("seeds", [])
    sweep_d = meta.get("sweep_d", [])
    sweep_h = meta.get("sweep_h", [])
    ticks = meta.get("ticks", {})
    if (
        len(seeds) >= 5
        and len(sweep_d) >= 10
        and len(sweep_h) >= 10
        and ticks.get("scenario", 0) >= 800
        and ticks.get("sweep", 0) >= 600
    ):
        return "full"
    return "exploratory"


def build_experiment_summary(
    meta: Dict[str, Any],
    scenario_results: Dict[str, Dict[str, Any]],
    sweep_results: List[Dict[str, Any]],
    sensitivity_results: Dict[float, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "run_class": classify_run(meta),
        "seed_count": len(meta.get("seeds", [])),
        "sweep_dimensions": {
            "mixing_values": len(meta.get("sweep_d", [])),
            "heterogeneity_values": len(meta.get("sweep_h", [])),
            "points": len(sweep_results),
        },
        "meta": meta,
    }

    scenario_metrics: Dict[str, Any] = {}
    for name, payload in scenario_results.items():
        eq = payload.get("equilibrium", {})
        scenario_metrics[name] = {
            "label": SCENARIO_LABELS.get(name, name),
            **_pick_metrics(
                eq,
                ["theta_std", "shannon_eco", "population", "occupancy", "mean_fitness"],
            ),
        }

    ranking = sorted(
        scenario_metrics.items(),
        key=lambda item: item[1].get("theta_std_mean", float("-inf")),
        reverse=True,
    )
    ranked_labels = [item[1]["label"] for item in ranking]
    pond = scenario_metrics.get("pond_network", {})
    deep = scenario_metrics.get("deep_ocean", {})

    summary["scenarios"] = {
        "metrics": scenario_metrics,
        "ranking_by_theta_std": ranked_labels,
        "pond_vs_deep_ocean_theta_std_fold_change": _safe_ratio(
            pond.get("theta_std_mean"), deep.get("theta_std_mean")
        ),
    }

    if sweep_results:
        best_overall = max(sweep_results, key=lambda row: row.get("theta_std_mean", float("-inf")))
        heterogeneity_profiles = []
        for h in sorted({float(row["heterogeneity"]) for row in sweep_results}):
            rows = sorted(
                (row for row in sweep_results if float(row["heterogeneity"]) == h),
                key=lambda row: float(row["mixing_rate"]),
            )
            best = max(rows, key=lambda row: row.get("theta_std_mean", float("-inf")))
            lowest = rows[0]
            highest = rows[-1]
            heterogeneity_profiles.append(
                {
                    "heterogeneity": h,
                    "best_mixing_rate": float(best["mixing_rate"]),
                    "best_theta_std": float(best["theta_std_mean"]),
                    "lowest_mixing_theta_std": float(lowest["theta_std_mean"]),
                    "highest_mixing_theta_std": float(highest["theta_std_mean"]),
                    "drop_to_high_mixing_pct": _drop_pct(
                        float(lowest["theta_std_mean"]),
                        float(highest["theta_std_mean"]),
                    ),
                }
            )

        h08_rows = sorted(
            (
                row
                for row in sweep_results
                if abs(float(row["heterogeneity"]) - 0.8) < 1e-9
            ),
            key=lambda row: float(row["mixing_rate"]),
        )
        h08_best = max(h08_rows, key=lambda row: row.get("theta_std_mean", float("-inf"))) if h08_rows else None
        h08_low = h08_rows[0] if h08_rows else None
        h08_high = h08_rows[-1] if h08_rows else None

        summary["sweep"] = {
            "best_overall": {
                "mixing_rate": float(best_overall["mixing_rate"]),
                "heterogeneity": float(best_overall["heterogeneity"]),
                "theta_std_mean": float(best_overall["theta_std_mean"]),
                "shannon_eco_mean": float(best_overall.get("shannon_eco_mean", 0.0)),
            },
            "best_mixing_zero_for_all_h": all(
                abs(profile["best_mixing_rate"]) < 1e-12 for profile in heterogeneity_profiles
            ),
            "heterogeneity_profiles": heterogeneity_profiles,
            "h08": {
                "best_mixing_rate": float(h08_best["mixing_rate"]) if h08_best else None,
                "best_theta_std": float(h08_best["theta_std_mean"]) if h08_best else None,
                "theta_std_at_d0": float(h08_low["theta_std_mean"]) if h08_low else None,
                "theta_std_at_d08": float(h08_high["theta_std_mean"]) if h08_high else None,
                "drop_pct_d0_to_d08": _drop_pct(
                    float(h08_low["theta_std_mean"]) if h08_low else None,
                    float(h08_high["theta_std_mean"]) if h08_high else None,
                ),
            },
        }

    if sensitivity_results:
        beta_profiles = []
        for beta in sorted(sensitivity_results):
            rows = sorted(sensitivity_results[beta], key=lambda row: float(row["mixing_rate"]))
            best = max(rows, key=lambda row: row.get("theta_std_mean", float("-inf")))
            beta_profiles.append(
                {
                    "niche_width": float(beta),
                    "best_mixing_rate": float(best["mixing_rate"]),
                    "best_theta_std": float(best["theta_std_mean"]),
                    "theta_std_at_lowest_mixing": float(rows[0]["theta_std_mean"]),
                    "theta_std_at_highest_mixing": float(rows[-1]["theta_std_mean"]),
                    "drop_to_high_mixing_pct": _drop_pct(
                        float(rows[0]["theta_std_mean"]),
                        float(rows[-1]["theta_std_mean"]),
                    ),
                }
            )

        summary["sensitivity"] = {
            "beta_profiles": beta_profiles,
            "all_betas_peak_at_min_mixing": all(
                abs(profile["best_mixing_rate"]) < 1e-12 for profile in beta_profiles
            ),
        }

    return summary


def summary_path(base_dir: str) -> str:
    return os.path.join(base_dir, "results", SUMMARY_FILENAME)


def write_experiment_summary(base_dir: str, summary: Dict[str, Any]) -> str:
    path = summary_path(base_dir)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path


def load_experiment_summary(base_dir: str) -> Dict[str, Any]:
    with open(summary_path(base_dir)) as f:
        return json.load(f)
