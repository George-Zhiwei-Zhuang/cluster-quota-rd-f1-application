#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/f1_replication_mpl")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, ensure_dirs


BLUE = "#1f5a85"
ORANGE = "#d97706"
GREY = "#8a939b"


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save(fig, stem: str) -> None:
    out = ROOT / "output" / "figures"
    fig.savefig(out / f"{stem}.png")
    fig.savefig(out / f"{stem}.pdf")
    plt.close(fig)


def figure5(pairs: pd.DataFrame) -> None:
    gaps = pairs["score_gap_sec"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    bins = np.geomspace(gaps.min() * 0.9, gaps.max() * 1.05, 20)
    axes[0].hist(gaps, bins=bins, color=BLUE, edgecolor="white", linewidth=0.6)
    axes[0].set_xscale("log")
    axes[0].axvline(np.median(gaps), color=ORANGE, linestyle="--", linewidth=1.5, label=f"Median = {np.median(gaps):.3f}s")
    axes[0].set_xlabel("Q2 rank-11 minus rank-10 lap-time gap (seconds, log scale)")
    axes[0].set_ylabel("Qualifying sessions")
    axes[0].set_title("Distribution of score gaps")
    axes[0].legend(frameon=False)

    x = np.sort(gaps)
    y = np.arange(1, len(x) + 1) / len(x)
    axes[1].step(x, y, where="post", color=BLUE, linewidth=2)
    for quantile in [.05, .25, .50, .75, .95]:
        value = np.quantile(gaps, quantile)
        axes[1].scatter([value], [quantile], color=ORANGE, s=22, zorder=3)
    axes[1].set_xlabel("Q2 lap-time gap (seconds)")
    axes[1].set_ylabel("Empirical cumulative probability")
    axes[1].set_title("Empirical distribution function")
    axes[1].set_ylim(0, 1.02)
    fig.suptitle("Score-gap heterogeneity at the Formula One Q3 capacity boundary", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "figure_5_score_gap_distribution")


def figure6(quartiles: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), sharex=True)
    specs = [
        ("race_points", "Race-points contrast", "Rank 10 minus rank 11 race points"),
        ("prior_points_complete", "Predetermined performance", "Prior-season points difference\n(complete focal pairs)"),
    ]
    for ax, (key, title, ylabel) in zip(axes, specs):
        data = quartiles[quartiles["variable"].eq(key)].sort_values("gap_quartile")
        estimate = data["estimate"].to_numpy(float)
        low = data["ci_low"].to_numpy(float)
        high = data["ci_high"].to_numpy(float)
        ax.errorbar(data["gap_quartile"], estimate, yerr=[estimate-low, high-estimate], fmt="o-", color=BLUE, capsize=4, linewidth=1.5)
        ax.axhline(0, color="black", linewidth=.8, alpha=.6)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Q2 score-gap quartile")
        ax.set_xticks([1, 2, 3, 4])
        for x, y, n in zip(data["gap_quartile"], estimate, data["n"]):
            ax.annotate(f"N={int(n)}", (x, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8)
    fig.suptitle("Outcome and predetermined contrasts by score-gap quartile", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "figure_6_gap_quartile_diagnostics")


def figure7(adjacent: pd.DataFrame, nearest: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for ax, key, ylabel in zip(axes, ["race_points", "top_ten"], ["Race-points contrast", "Top-ten probability contrast"]):
        data = adjacent[adjacent["variable"].eq(key)].copy()
        data["order"] = data["pair"].map({"8-9": 0, "9-10": 1, "10-11": 2, "11-12": 3, "12-13": 4})
        data = data.sort_values("order")
        colors = [ORANGE if pair == "10-11" else GREY for pair in data["pair"]]
        ax.bar(data["pair"], data["estimate"], color=colors, width=.72)
        ax.errorbar(
            np.arange(len(data)), data["estimate"],
            yerr=[data["estimate"]-data["ci_low"], data["ci_high"]-data["estimate"]],
            fmt="none", ecolor="black", capsize=3, linewidth=1,
        )
        ax.axhline(0, color="black", linewidth=.8)
        row = nearest[nearest["variable"].eq(key)].iloc[0]
        ax.text(.03, .96, f"Boundary excess = {row['estimate']:.3f}\np = {row['p_value']:.3f}", transform=ax.transAxes, va="top", fontsize=9)
        ax.set_xlabel("Adjacent Q2 ranks")
        ax.set_ylabel(ylabel)
    fig.suptitle("Q3-boundary contrast relative to neighboring rank gradients", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "figure_7_quota_vs_neighboring_gradients")


def figure8(primary: pd.DataFrame, extreme: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.0))
    methods = ["race_iid", "season_cluster", "season_block_bootstrap", "race_driver_two_way"]
    labels = ["Race i.i.d.", "Season cluster", "Season bootstrap", "Race x driver"]
    for row_index, key in enumerate(["race_points", "top_ten"]):
        data = primary[(primary["variable"].eq(key)) & primary["method"].isin(methods)].set_index("method").loc[methods]
        ax = axes[row_index, 0]
        y = np.arange(len(methods))
        ax.errorbar(data["estimate"], y, xerr=[data["estimate"]-data["ci_low"], data["ci_high"]-data["estimate"]], fmt="o", color=BLUE, capsize=3)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("Estimate and 95% interval")
        ax.set_title("Race points" if key == "race_points" else "Top-ten finish")

        right = axes[row_index, 1]
        variable = f"{key}_diff"
        d = extreme[extreme["variable"].eq(variable)].sort_values("removed_largest_gaps")
        right.plot(d["removed_largest_gaps"], d["estimate"], marker="o", color=ORANGE)
        right.fill_between(d["removed_largest_gaps"], d["ci_low"], d["ci_high"], color=ORANGE, alpha=.18)
        right.axhline(0, color="black", linewidth=.8)
        right.set_xlabel("Largest score-gap races removed")
        right.set_ylabel("Rank 10 minus rank 11 contrast")
        right.set_title("Extreme-gap sensitivity")
    fig.suptitle("Dependence and extreme-gap sensitivity", y=1.01, fontsize=12)
    fig.tight_layout()
    save(fig, "figure_8_dependence_and_extreme_gap_sensitivity")


def rookie_figure(rookie: pd.DataFrame, quartiles: pd.DataFrame, pairs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8))
    shares = [1-pairs["rank10_has_prior_history"].mean(), 1-pairs["rank11_has_prior_history"].mean()]
    axes[0].bar(["Rank 10", "Rank 11"], shares, color=[BLUE, GREY])
    axes[0].set_ylabel("Share without prior-season starts")
    axes[0].set_title("History availability at focal ranks")
    axes[0].set_ylim(0, max(shares) * 1.35)

    missing = quartiles[quartiles["variable"].eq("any_no_prior_history_share")].sort_values("gap_quartile")
    axes[1].plot(missing["gap_quartile"], missing["estimate"], marker="o", color=ORANGE)
    axes[1].set_xticks([1, 2, 3, 4])
    axes[1].set_xlabel("Score-gap quartile")
    axes[1].set_ylabel("Share of focal pairs with missing history")
    axes[1].set_title("Missing history across score gaps")

    metrics = ["prior_points_zero_inclusive_difference", "prior_points_complete_pair_difference"]
    d = rookie[rookie["metric"].isin(metrics)].set_index("metric").loc[metrics]
    labels = ["Zero-inclusive", "Complete pair"]
    axes[2].errorbar(d["value"], np.arange(2), xerr=[d["value"]-d["ci_low"], d["ci_high"]-d["value"]], fmt="o", color=BLUE, capsize=3)
    axes[2].set_yticks(np.arange(2), labels)
    axes[2].axvline(0, color="black", linewidth=.8)
    axes[2].set_xlabel("Prior-season-points contrast")
    axes[2].set_title("Sensitivity to history coding")
    fig.suptitle("Prior-season-history and rookie sensitivity", y=1.02, fontsize=12)
    fig.tight_layout()
    save(fig, "figure_a1_rookie_history_sensitivity")


def leave_one_out_figure(loo: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.plot(loo["omitted_season"], loo["race_points_estimate"], marker="o", color=BLUE, label="Race points")
    ax.plot(loo["omitted_season"], loo["top_ten_estimate"], marker="s", color=ORANGE, label="Top-ten probability")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set_xlabel("Omitted season")
    ax.set_ylabel("Leave-one-season-out estimate")
    ax.set_title("Leave-one-season-out sensitivity")
    ax.legend(frameon=False)
    save(fig, "figure_a2_leave_one_season_out")


def main() -> None:
    ensure_dirs()
    setup_style()
    p = ROOT / "data" / "processed"
    pairs = pd.read_csv(p / "focal_pairs.csv")
    quartiles = pd.read_csv(p / "gap_quartile_diagnostics.csv")
    adjacent = pd.read_csv(p / "adjacent_rank_contrasts.csv")
    nearest = pd.read_csv(p / "boundary_excess_nearest.csv")
    primary = pd.read_csv(p / "primary_contrasts_long.csv")
    extreme = pd.read_csv(p / "extreme_gap_sensitivity.csv")
    rookie = pd.read_csv(p / "rookie_history_audit.csv")
    loo = pd.read_csv(p / "leave_one_season_out.csv")
    figure5(pairs)
    figure6(quartiles)
    figure7(adjacent, nearest)
    figure8(primary, extreme)
    rookie_figure(rookie, quartiles, pairs)
    leave_one_out_figure(loo)
    figures = sorted(path.name for path in (ROOT / "output" / "figures").glob("*.png"))
    log = {"figures": figures}
    (ROOT / "output" / "logs" / "05_figures.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
