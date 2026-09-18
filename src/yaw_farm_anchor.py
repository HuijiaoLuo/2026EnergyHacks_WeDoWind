"""FarmAnchor absolute-anchor exploration for the independent yaw model.

This module deliberately leaves relative-heading boundaries, state levels, and
amplitude blending untouched. It only replaces the estimation of the shared
absolute centre C used by ``B0_i = C - theta_star_i``.

The exploration first forms daily state-consistent observations
``a_i(d) = y_i(d) + c_i(d) + theta_star_i``, reduces them to a
quality-weighted turbine-level arithmetic mean, and then applies a robust
Huber centre across turbines. The historical ``method='mean'`` name is
retained for the default corrected-mean route; its location estimator is the
Huber centre of the turbine-level weighted means. Farm centres are preferred
when the target farm has labelled support, with the global centre as fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


def turbine_farm_key(turbine: str) -> str:
    """Infer a conservative farm key from the published turbine identifier."""
    return str(turbine).split("_", 1)[0]


def legacy_global_anchor(
    train_ids: Sequence[str],
    labels: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
) -> float:
    """Reproduce the release notebook's unweighted mean anchor."""
    values = [
        float(labels[turbine].mean()) + float(theta_star[turbine])
        for turbine in train_ids
    ]
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite labelled anchors are available.")
    return float(values.mean())


def _weighted_median(values: Sequence[float], weights: Sequence[float] | None = None) -> float:
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        return np.nan

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cutoff = 0.5 * weights.sum()
    return float(values[np.searchsorted(np.cumsum(weights), cutoff, side="left")])


def _robust_scale(values: np.ndarray, centre: float) -> float:
    deviation = np.abs(values - centre)
    mad = _weighted_median(deviation)
    if np.isfinite(mad) and mad > 1e-9:
        return float(1.4826 * mad)
    fallback = float(np.nanstd(values))
    return fallback if np.isfinite(fallback) and fallback > 1e-9 else 1.0


def _huber_location(
    values: Sequence[float],
    weights: Sequence[float] | None = None,
    tuning: float = 1.5,
    max_iter: int = 50,
) -> float:
    """Iteratively reweighted Huber location with deterministic fallback."""
    values = np.asarray(values, dtype=float)
    if weights is None:
        weights = np.ones(len(values), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        return np.nan

    centre = _weighted_median(values, weights)
    # Keep the robust scale anchored to the initial median. Re-estimating the
    # scale after every update lets one extreme turbine inflate the scale and
    # pulls the estimator back toward the ordinary mean in tiny folds.
    scale = _robust_scale(values, centre)
    for _ in range(max_iter):
        residual = np.abs(values - centre) / scale
        robust_weight = np.ones(len(values), dtype=float)
        large = residual > tuning
        robust_weight[large] = tuning / residual[large]
        combined = weights * robust_weight
        updated = float(np.sum(combined * values) / np.sum(combined))
        if abs(updated - centre) < 1e-8:
            break
        centre = updated
    return float(centre)


def _daily_median(series: pd.Series) -> pd.Series:
    series = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if series.empty:
        return series
    return series.groupby(series.index.normalize()).median().sort_index()


def build_anchor_observations(
    turbine: str,
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
) -> pd.DataFrame:
    """Build daily absolute-anchor observations for one labelled turbine.

    Under the retained model, y + correction + theta_star should be centred on
    C.  State-quality weights are used only to downweight noisy dates; they do
    not modify the frozen state detector.
    """
    label_daily = _daily_median(labels[turbine])
    if label_daily.empty:
        return pd.DataFrame(
            columns=[
                "label_deg",
                "state_correction_deg",
                "quality_weight",
                "anchor_deg",
            ]
        )

    observables = stage_bundles[turbine]["observables"].copy()
    correction = observables.get(
        "relative_prior_correction",
        pd.Series(0.0, index=observables.index),
    )
    correction_daily = _daily_median(correction)

    quality = observables.get(
        "state_quality_weight",
        pd.Series(1.0, index=observables.index),
    )
    quality_daily = _daily_median(quality).clip(lower=0.0, upper=1.0)

    out = pd.DataFrame(index=label_daily.index)
    out["label_deg"] = label_daily
    out["state_correction_deg"] = correction_daily.reindex(out.index).fillna(0.0)
    out["quality_weight"] = quality_daily.reindex(out.index).fillna(1.0)
    out["anchor_deg"] = (
        out["label_deg"]
        + out["state_correction_deg"]
        + float(theta_star[turbine])
    )
    return out


def summarize_turbine_anchor(
    turbine: str,
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
) -> dict:
    """Build one quality-weighted turbine-level mean for the shared C."""
    daily = build_anchor_observations(
        turbine,
        stage_bundles,
        labels,
        theta_star,
    )
    if daily.empty:
        return {
            "turbine": turbine,
            "n_days": 0,
            "n_eff_days": 0.0,
            "anchor_mean_deg": np.nan,
            "anchor_median_deg": np.nan,
            "anchor_mad_deg": np.nan,
        }

    values = daily["anchor_deg"].to_numpy(dtype=float)
    weights = daily["quality_weight"].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    values = values[valid]
    weights = weights[valid]
    if not len(values):
        return {
            "turbine": turbine,
            "n_days": 0,
            "n_eff_days": 0.0,
            "anchor_mean_deg": np.nan,
            "anchor_median_deg": np.nan,
            "anchor_mad_deg": np.nan,
        }

    centre = _weighted_median(values, weights)
    n_eff = float(weights.sum() ** 2 / np.sum(weights ** 2))
    return {
        "turbine": turbine,
        "n_days": int(len(values)),
        "n_eff_days": n_eff,
        "anchor_mean_deg": float(np.average(values, weights=weights)),
        "anchor_median_deg": centre,
        "anchor_mad_deg": float(_weighted_median(np.abs(values - centre), weights)),
    }


@dataclass
class FarmAnchorFit:
    """Fitted centres and diagnostics for one training fold."""

    method: str
    global_center: float
    farm_centers: dict[str, float]
    farm_centers_raw: dict[str, float]
    shrinkage: dict[str, float]
    turbine_summary: pd.DataFrame

    def center_for(self, turbine: str, farm_map: Mapping[str, str] | None = None) -> float:
        farm = (
            farm_map.get(turbine, turbine_farm_key(turbine))
            if farm_map is not None
            else turbine_farm_key(turbine)
        )
        return float(self.farm_centers.get(farm, self.global_center))


def fit_farm_anchor(
    train_ids: Sequence[str],
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
    farm_map: Mapping[str, str] | None = None,
    method: str = "median",
    shrink_kappa: float = 0.0,
) -> FarmAnchorFit:
    """Fit robust turbine-balanced farm/global centres with optional shrinkage.

    The farm key defaults to the identifier prefix (PPP/SSS in this project).
    A caller can provide a better metadata-derived mapping.  Shrinkage is
    applied between farm-level centres, not between individual dates. The
    default candidate uses ``shrink_kappa=0``: a target with labelled support
    in its farm receives that farm centre, otherwise it receives the global
    centre. Positive shrinkage is retained only as an optional diagnostic. For
    the default corrected-mean route (``method='mean'``), a Huber centre is fitted
    over the turbine-level quality-weighted anchor means. ``method='median'``
    uses turbine-level weighted medians, and ``method='huber'`` applies the
    Huber centre to those medians.
    """
    if method not in {"mean", "median", "huber"}:
        raise ValueError("method must be 'mean', 'median', or 'huber'")

    farm_map = farm_map or {}
    summary = pd.DataFrame(
        [
            summarize_turbine_anchor(
                turbine,
                stage_bundles,
                labels,
                theta_star,
            )
            for turbine in train_ids
        ]
    )
    summary["farm"] = [
        farm_map.get(turbine, turbine_farm_key(turbine))
        for turbine in summary["turbine"]
    ]
    summary = summary[np.isfinite(summary["anchor_median_deg"])].copy()
    if summary.empty:
        raise ValueError("No finite robust turbine anchors are available.")

    def location(values: Sequence[float]) -> float:
        if method == "median":
            return _weighted_median(values)
        return _huber_location(values)

    location_column = (
        "anchor_mean_deg"
        if method == "mean"
        else "anchor_median_deg"
    )
    summary["anchor_location_deg"] = summary[location_column]

    global_center = float(location(summary["anchor_location_deg"].to_numpy()))
    raw_centers: dict[str, float] = {}
    farm_centers: dict[str, float] = {}
    shrinkage: dict[str, float] = {}

    for farm, part in summary.groupby("farm", sort=True):
        raw = float(location(part["anchor_location_deg"].to_numpy()))
        raw_centers[str(farm)] = raw
        other = summary[summary["farm"] != farm]
        prior = (
            float(location(other["anchor_location_deg"].to_numpy()))
            if len(other)
            else global_center
        )
        n_farm = float(len(part))
        alpha = (
            1.0
            if shrink_kappa <= 0.0
            else n_farm / (n_farm + float(shrink_kappa))
        )
        farm_centers[str(farm)] = float(alpha * raw + (1.0 - alpha) * prior)
        shrinkage[str(farm)] = float(alpha)

    return FarmAnchorFit(
        method=method,
        global_center=global_center,
        farm_centers=farm_centers,
        farm_centers_raw=raw_centers,
        shrinkage=shrinkage,
        turbine_summary=summary.sort_values("turbine").reset_index(drop=True),
    )


def run_anchor_loto(
    train_ids: Sequence[str],
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
    farm_map: Mapping[str, str] | None = None,
    shrink_kappa: float = 0.0,
    beta: float = 1.0,
) -> pd.DataFrame:
    """Compare the release C with FarmAnchor variants under turbine LOTO."""
    from yaw_model import predict_from_bundle, score_prediction

    rows: list[dict] = []
    methods = {
        "legacy_mean": None,
        "corrected_mean": "mean",
        "farm_median": "median",
        "farm_huber": "huber",
    }

    for holdout in train_ids:
        fit_ids = [turbine for turbine in train_ids if turbine != holdout]
        for method_name, method in methods.items():
            if method is None:
                centre = legacy_global_anchor(fit_ids, labels, theta_star)
                fit = None
            else:
                fit = fit_farm_anchor(
                    fit_ids,
                    stage_bundles,
                    labels,
                    theta_star,
                    farm_map=farm_map,
                    method=method,
                    shrink_kappa=shrink_kappa,
                )
                centre = fit.center_for(holdout, farm_map=farm_map)

            prediction = predict_from_bundle(
                holdout,
                stage_bundles[holdout],
                centre,
                beta,
                theta_star,
            )
            metrics = score_prediction(
                prediction,
                labels[holdout],
                centre - theta_star[holdout],
            )
            rows.append(
                {
                    "anchor": method_name,
                    "holdout": holdout,
                    "farm": (farm_map or {}).get(
                        holdout,
                        turbine_farm_key(holdout),
                    ),
                    "C": centre,
                    "B0": centre - theta_star[holdout],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "constant_mae": metrics["constant_mae"],
                    "constant_rmse": metrics["constant_rmse"],
                    "states": metrics["n_states"],
                }
            )

    return pd.DataFrame(rows)
