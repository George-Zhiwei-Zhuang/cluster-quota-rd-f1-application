from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with (ROOT / "config" / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dirs() -> None:
    for rel in (
        "data/raw/jolpica",
        "data/interim",
        "data/processed",
        "output/tables",
        "output/figures",
        "output/logs",
    ):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_lap_time(value) -> float:
    """Convert an Ergast/Jolpica lap-time string to seconds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return 60.0 * float(minutes) + float(seconds)
        return float(text)
    except ValueError:
        return np.nan


def _cluster_meat(X: np.ndarray, residual: np.ndarray, groups: np.ndarray) -> np.ndarray:
    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    for group in pd.unique(groups):
        idx = groups == group
        score = X[idx].T @ residual[idx]
        meat += np.outer(score, score)
    return meat


def ols_cluster(
    y,
    X,
    groups,
    coefficient: int = 0,
    alpha: float = 0.05,
) -> dict:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1) & pd.notna(groups)
    y, X, groups = y[keep], X[keep], groups[keep]
    n, k = X.shape
    if n <= k:
        return {"estimate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "p_value": np.nan, "n": n, "clusters": len(pd.unique(groups))}
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ (X.T @ y)
    residual = y - X @ beta
    unique_groups = pd.unique(groups)
    g = len(unique_groups)
    meat = _cluster_meat(X, residual, groups)
    correction = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 else 1.0
    covariance = correction * bread @ meat @ bread
    se = float(np.sqrt(max(covariance[coefficient, coefficient], 0.0)))
    estimate = float(beta[coefficient])
    df = max(g - 1, 1)
    critical = float(stats.t.ppf(1 - alpha / 2, df=df))
    t_stat = estimate / se if se > 0 else np.nan
    p_value = float(2 * stats.t.sf(abs(t_stat), df=df)) if np.isfinite(t_stat) else np.nan
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "p_value": p_value,
        "n": int(n),
        "clusters": int(g),
        "df": int(df),
    }


def mean_iid(values, alpha: float = 0.05) -> dict:
    values = np.asarray(pd.Series(values).dropna(), dtype=float)
    n = len(values)
    estimate = float(np.mean(values)) if n else np.nan
    se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    critical = float(stats.t.ppf(1 - alpha / 2, df=n - 1)) if n > 1 else np.nan
    t_stat = estimate / se if se and se > 0 else np.nan
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - critical * se if np.isfinite(critical) else np.nan,
        "ci_high": estimate + critical * se if np.isfinite(critical) else np.nan,
        "p_value": float(2 * stats.t.sf(abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else np.nan,
        "n": int(n),
    }


def mean_cluster(values, groups, alpha: float = 0.05) -> dict:
    values = np.asarray(values, dtype=float)
    X = np.ones((len(values), 1), dtype=float)
    return ols_cluster(values, X, groups, coefficient=0, alpha=alpha)


def slope_cluster(values, regressor, groups, alpha: float = 0.05) -> dict:
    frame = pd.DataFrame({"y": values, "x": regressor, "group": groups}).dropna()
    X = np.column_stack([np.ones(len(frame)), frame["x"].to_numpy(float)])
    return ols_cluster(frame["y"], X, frame["group"], coefficient=1, alpha=alpha)


def two_way_cluster_ols(y, X, group_a, group_b, coefficient: int = 1, alpha: float = 0.05) -> dict:
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)
    keep = (
        np.isfinite(y)
        & np.all(np.isfinite(X), axis=1)
        & pd.notna(group_a)
        & pd.notna(group_b)
    )
    y, X = y[keep], X[keep]
    group_a, group_b = group_a[keep], group_b[keep]
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ X.T @ y
    residual = y - X @ beta

    def corrected_meat(groups):
        g = len(pd.unique(groups))
        factor = (g / (g - 1)) * ((n - 1) / (n - k)) if g > 1 else 1.0
        return factor * _cluster_meat(X, residual, groups)

    intersection = np.array([f"{a}::{b}" for a, b in zip(group_a, group_b)], dtype=object)
    meat = corrected_meat(group_a) + corrected_meat(group_b) - corrected_meat(intersection)
    covariance = bread @ meat @ bread
    se = float(np.sqrt(max(covariance[coefficient, coefficient], 0.0)))
    estimate = float(beta[coefficient])
    critical = float(stats.norm.ppf(1 - alpha / 2))
    z_stat = estimate / se if se > 0 else np.nan
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "p_value": float(2 * stats.norm.sf(abs(z_stat))) if np.isfinite(z_stat) else np.nan,
        "n": int(n),
        "race_clusters": int(len(pd.unique(group_a))),
        "driver_clusters": int(len(pd.unique(group_b))),
    }


def season_block_bootstrap(values, seasons, replications: int, seed: int, alpha: float = 0.05) -> dict:
    frame = pd.DataFrame({"value": values, "season": seasons}).dropna()
    unique = np.sort(frame["season"].unique())
    rng = np.random.default_rng(seed)
    draws = np.empty(replications, dtype=float)
    blocks = {s: frame.loc[frame["season"] == s, "value"].to_numpy(float) for s in unique}
    for b in range(replications):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[b] = np.mean(np.concatenate([blocks[s] for s in sampled]))
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(frame["value"].mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "replications": int(replications),
        "seed": int(seed),
        "n": int(len(frame)),
        "clusters": int(len(unique)),
    }


def sign_enumeration_pvalue(values, seasons) -> dict:
    frame = pd.DataFrame({"value": values, "season": seasons}).dropna()
    unique = np.sort(frame["season"].unique())
    cluster_sums = frame.groupby("season")["value"].sum().reindex(unique).to_numpy(float)
    observed = abs(cluster_sums.sum())
    count = 0
    total = 2 ** len(unique)
    for mask in range(total):
        signs = np.array([1.0 if (mask >> j) & 1 else -1.0 for j in range(len(unique))])
        if abs(np.sum(signs * cluster_sums)) >= observed - 1e-12:
            count += 1
    return {"p_value": count / total, "enumerations": total, "clusters": len(unique)}


def fmt_number(value, digits: int = 3) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"

