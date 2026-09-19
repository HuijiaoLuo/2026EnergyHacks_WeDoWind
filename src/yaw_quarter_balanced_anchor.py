"""Quarter-balanced robust aggregation for the farm-level yaw anchor C.

This exploration leaves the turbine-specific ``theta_star`` and every dynamic
relative-state component untouched.  It changes only the time aggregation of
the state-consistent daily anchor observations

    a_i(d) = y_i(d) + c_i(d) + theta_star_i.

The current FarmAnchor gives all usable days weight through one quality-
weighted mean per turbine.  Here each usable calendar quarter first receives
one turbine-level estimate.  A turbine's long-run level is then formed from
equal quarter estimates, optionally with a robust Huber or median location,
before the same turbine-balanced Huber centre estimates C_f.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from yaw_farm_anchor import (
    _huber_location,
    build_anchor_observations,
    fit_farm_anchor,
    turbine_farm_key,
)
from yaw_model import predict_from_bundle, score_prediction


@dataclass(frozen=True)
class QuarterBalancedAnchorConfig:
    """Pre-declared support rules for a long-run quarterly C estimate."""

    min_effective_days_per_quarter: float = 20.0
    min_valid_quarters_per_turbine: int = 6


@dataclass(frozen=True)
class QuarterBalancedAnchorFit:
    """Farm/global C estimates for a selected equal-quarter aggregator."""

    method: str
    global_center: float
    farm_centers: Mapping[str, float]
    turbine_summary: pd.DataFrame
    quarterly_observations: pd.DataFrame

    def center_for(
        self,
        turbine: str,
        farm_map: Mapping[str, str] | None = None,
    ) -> float:
        farm = (farm_map or {}).get(turbine, turbine_farm_key(turbine))
        return float(self.farm_centers.get(farm, self.global_center))


def _validate_config(config: QuarterBalancedAnchorConfig) -> None:
    if config.min_effective_days_per_quarter <= 0.0:
        raise ValueError("min_effective_days_per_quarter must be positive.")
    if config.min_valid_quarters_per_turbine < 1:
        raise ValueError("min_valid_quarters_per_turbine must be at least one.")


def summarize_quarter_anchor_daily(
    daily: pd.DataFrame,
    config: QuarterBalancedAnchorConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Reduce one turbine's daily anchor observations to equal-quarter data.

    Daily quality weights retain their existing role *within* a quarter.  The
    long-run quarter aggregators then give every accepted calendar quarter an
    equal vote, preventing a highly sampled season from defining C_f.
    """
    config = config or QuarterBalancedAnchorConfig()
    _validate_config(config)
    required = {"anchor_deg", "quality_weight"}
    if not required.issubset(daily.columns):
        raise ValueError(f"daily must contain {sorted(required)}")

    work = daily[["anchor_deg", "quality_weight"]].copy()
    work.index = pd.to_datetime(work.index)
    work["quarter"] = work.index.to_period("Q").astype(str)
    rows: list[dict[str, float | int | str]] = []
    for quarter, part in work.groupby("quarter", sort=True):
        values = pd.to_numeric(part["anchor_deg"], errors="coerce").to_numpy(dtype=float)
        weights = pd.to_numeric(part["quality_weight"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
        values = values[valid]
        weights = weights[valid]
        if not len(values):
            continue
        n_eff = float(weights.sum() ** 2 / np.square(weights).sum())
        rows.append(
            {
                "quarter": str(quarter),
                "n_days": int(len(values)),
                "n_eff_days": n_eff,
                "mean_quality_weight": float(weights.mean()),
                "anchor_quarter_deg": float(np.average(values, weights=weights)),
                "quarter_accepted": bool(
                    n_eff >= config.min_effective_days_per_quarter
                ),
            }
        )

    quarterly = pd.DataFrame(
        rows,
        columns=[
            "quarter",
            "n_days",
            "n_eff_days",
            "mean_quality_weight",
            "anchor_quarter_deg",
            "quarter_accepted",
        ],
    )
    accepted = quarterly.loc[quarterly["quarter_accepted"]] if len(quarterly) else quarterly
    values = accepted["anchor_quarter_deg"].to_numpy(dtype=float)
    if len(values) >= config.min_valid_quarters_per_turbine:
        equal_mean = float(np.mean(values))
        equal_huber = float(_huber_location(values))
        equal_median = float(np.median(values))
    else:
        equal_mean = equal_huber = equal_median = np.nan

    all_values = pd.to_numeric(daily["anchor_deg"], errors="coerce").to_numpy(dtype=float)
    all_weights = pd.to_numeric(daily["quality_weight"], errors="coerce").to_numpy(dtype=float)
    valid_daily = np.isfinite(all_values) & np.isfinite(all_weights) & (all_weights > 0.0)
    daily_weighted = (
        float(np.average(all_values[valid_daily], weights=all_weights[valid_daily]))
        if valid_daily.any()
        else np.nan
    )
    summary = {
        "n_quarters_total": int(len(quarterly)),
        "n_quarters_accepted": int(len(accepted)),
        "anchor_daily_weighted_deg": daily_weighted,
        "anchor_equal_quarter_mean_deg": equal_mean,
        "anchor_equal_quarter_huber_deg": equal_huber,
        "anchor_equal_quarter_median_deg": equal_median,
    }
    return quarterly, summary


def fit_quarter_balanced_anchor(
    train_ids: Sequence[str],
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
    farm_map: Mapping[str, str] | None = None,
    method: str = "equal_quarter_huber",
    config: QuarterBalancedAnchorConfig | None = None,
) -> QuarterBalancedAnchorFit:
    """Fit C_f from equal-quarter turbine summaries and a Huber farm centre."""
    config = config or QuarterBalancedAnchorConfig()
    _validate_config(config)
    method_to_column = {
        "equal_quarter_mean": "anchor_equal_quarter_mean_deg",
        "equal_quarter_huber": "anchor_equal_quarter_huber_deg",
        "equal_quarter_median": "anchor_equal_quarter_median_deg",
    }
    if method not in method_to_column:
        raise ValueError(f"method must be one of {sorted(method_to_column)}")

    summary_rows: list[dict[str, float | int | str]] = []
    quarterly_tables: list[pd.DataFrame] = []
    for turbine in train_ids:
        daily = build_anchor_observations(turbine, stage_bundles, labels, theta_star)
        quarterly, summary = summarize_quarter_anchor_daily(daily, config)
        quarterly = quarterly.assign(turbine=str(turbine))
        quarterly_tables.append(quarterly)
        summary_rows.append(
            {
                "turbine": str(turbine),
                "farm": (farm_map or {}).get(turbine, turbine_farm_key(turbine)),
                **summary,
            }
        )

    summary_frame = pd.DataFrame(summary_rows)
    selected_column = method_to_column[method]
    usable = summary_frame[np.isfinite(summary_frame[selected_column])].copy()
    if usable.empty:
        raise ValueError("No turbine has enough accepted quarters for the anchor.")
    usable["anchor_location_deg"] = usable[selected_column]

    global_center = float(_huber_location(usable["anchor_location_deg"].to_numpy()))
    farm_centers = {
        str(farm): float(_huber_location(part["anchor_location_deg"].to_numpy()))
        for farm, part in usable.groupby("farm", sort=True)
    }
    quarterly_observations = (
        pd.concat(quarterly_tables, ignore_index=True)
        if quarterly_tables
        else pd.DataFrame()
    )
    if not quarterly_observations.empty:
        quarterly_observations["farm"] = quarterly_observations["turbine"].map(
            lambda turbine: (farm_map or {}).get(turbine, turbine_farm_key(turbine))
        )
        quarterly_observations = quarterly_observations.sort_values(
            ["turbine", "quarter"]
        ).reset_index(drop=True)
    return QuarterBalancedAnchorFit(
        method=method,
        global_center=global_center,
        farm_centers=farm_centers,
        turbine_summary=usable.sort_values("turbine").reset_index(drop=True),
        quarterly_observations=quarterly_observations,
    )


def run_quarter_balanced_anchor_loto(
    train_ids: Sequence[str],
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
    farm_map: Mapping[str, str] | None = None,
    config: QuarterBalancedAnchorConfig | None = None,
    beta: float = 1.0,
) -> pd.DataFrame:
    """Compare the current corrected mean with pre-declared quarter variants."""
    config = config or QuarterBalancedAnchorConfig()
    _validate_config(config)
    rows: list[dict[str, float | str]] = []
    variants = [
        "current_daily_weighted",
        "equal_quarter_mean",
        "equal_quarter_huber",
        "equal_quarter_median",
    ]
    for holdout in train_ids:
        fit_ids = [turbine for turbine in train_ids if turbine != holdout]
        for variant in variants:
            if variant == "current_daily_weighted":
                fit = fit_farm_anchor(
                    fit_ids,
                    stage_bundles,
                    labels,
                    theta_star,
                    farm_map=farm_map,
                    method="mean",
                    shrink_kappa=0.0,
                )
            else:
                fit = fit_quarter_balanced_anchor(
                    fit_ids,
                    stage_bundles,
                    labels,
                    theta_star,
                    farm_map=farm_map,
                    method=variant,
                    config=config,
                )
            center = fit.center_for(holdout, farm_map=farm_map)
            prediction = predict_from_bundle(
                holdout,
                stage_bundles[holdout],
                center,
                beta,
                theta_star,
            )
            metrics = score_prediction(
                prediction,
                labels[holdout],
                center - theta_star[holdout],
            )
            rows.append(
                {
                    "anchor": variant,
                    "holdout": str(holdout),
                    "C": center,
                    "B0": center - float(theta_star[holdout]),
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "constant_mae": metrics["constant_mae"],
                    "constant_rmse": metrics["constant_rmse"],
                    "states": metrics["n_states"],
                }
            )
    return pd.DataFrame(rows)
