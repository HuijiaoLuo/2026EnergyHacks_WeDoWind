"""Temporal common-mode denoising for the existing theta_star coordinate.

This diagnostic preserves the raw turbine-to-turbine theta_star geometry.  It
does not centre a turbine against peer *absolute* angles, because that would
change the static coordinate represented by ``C - theta_star``.  Instead it
only estimates a same-farm, date-level nuisance term from each peer's
deviation around its own raw theta_star:

    g_-i(d) = CircMedian_j wrap(theta_j(d) - theta_star_j)
    theta_i_temporal = CircMedian_d wrap(theta_i(d) - g_-i(d))

The resulting scalar remains a turbine-specific absolute reference.  Frozen
relative-heading dynamics, state corrections, anchor aggregation, and export
paths are intentionally outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from baseline_loto_ridge import wrap_180
from fleet_context import circular_median_deg
from yaw_farm_anchor import legacy_global_anchor, turbine_farm_key
from yaw_model import predict_from_bundle, score_prediction


@dataclass(frozen=True)
class TemporalThetaFieldConfig:
    """Pre-declared minimum support for a temporal common-mode estimate."""

    min_peer_turbines: int = 2
    min_common_days: int = 180


def _validate_config(config: TemporalThetaFieldConfig) -> None:
    if config.min_peer_turbines < 1:
        raise ValueError("min_peer_turbines must be at least one.")
    if config.min_common_days < 1:
        raise ValueError("min_common_days must be positive.")


def _daily_numeric(series: pd.Series) -> pd.Series:
    """Return a finite daily median indexed by normalized timestamps."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean = clean[np.isfinite(clean.to_numpy(dtype=float))]
    if clean.empty:
        return pd.Series(dtype=float)
    clean.index = pd.to_datetime(clean.index).normalize()
    return clean.groupby(clean.index).median().sort_index()


def build_temporal_common_mode_theta(
    theta_series: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
    farm_map: Mapping[str, str] | None = None,
    turbines: Sequence[str] | None = None,
    config: TemporalThetaFieldConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Denoise each turbine's rolling theta series with peer deviations only.

    The leave-one-out peer field can use unlabeled SCADA from all supplied
    turbines.  A turbine's own series never contributes to its daily nuisance
    estimate.  Returned coordinates are omitted when their common-mode daily
    support is below the configured threshold.
    """
    config = config or TemporalThetaFieldConfig()
    _validate_config(config)
    turbine_ids = list(turbines) if turbines is not None else sorted(theta_series)
    missing = [turbine for turbine in turbine_ids if turbine not in theta_series or turbine not in theta_star]
    if missing:
        raise KeyError("Missing theta series or theta_star for: " + ", ".join(missing))

    farm_for = {
        turbine: (farm_map or {}).get(turbine, turbine_farm_key(turbine))
        for turbine in turbine_ids
    }
    daily = {turbine: _daily_numeric(theta_series[turbine]) for turbine in turbine_ids}
    rows: list[dict[str, float | int | str | pd.Timestamp]] = []
    summary_rows: list[dict[str, float | int | str]] = []
    coordinate: dict[str, float] = {}

    for turbine in turbine_ids:
        peers = [
            peer
            for peer in turbine_ids
            if peer != turbine and farm_for[peer] == farm_for[turbine]
        ]
        corrected_values: list[float] = []
        common_values: list[float] = []
        for date, own_theta in daily[turbine].items():
            deviations = []
            for peer in peers:
                peer_theta = daily[peer].get(date, np.nan)
                if np.isfinite(peer_theta):
                    deviations.append(
                        wrap_180(float(peer_theta) - float(theta_star[peer]))
                    )
            n_peers = len(deviations)
            common_mode = (
                float(circular_median_deg(deviations))
                if n_peers >= config.min_peer_turbines
                else np.nan
            )
            theta_denoised = (
                float(wrap_180(float(own_theta) - common_mode))
                if np.isfinite(common_mode)
                else np.nan
            )
            if np.isfinite(theta_denoised):
                corrected_values.append(theta_denoised)
                common_values.append(common_mode)
            rows.append(
                {
                    "turbine": str(turbine),
                    "farm": str(farm_for[turbine]),
                    "date": pd.Timestamp(date),
                    "theta_raw_daily_deg": float(own_theta),
                    "peer_common_mode_deg": common_mode,
                    "peer_count": int(n_peers),
                    "theta_denoised_daily_deg": theta_denoised,
                }
            )

        n_common = len(corrected_values)
        theta_temporal = (
            float(circular_median_deg(corrected_values))
            if n_common >= config.min_common_days
            else np.nan
        )
        if np.isfinite(theta_temporal):
            coordinate[str(turbine)] = theta_temporal
        raw_theta = float(theta_star[turbine])
        summary_rows.append(
            {
                "turbine": str(turbine),
                "farm": str(farm_for[turbine]),
                "theta_star_raw_deg": raw_theta,
                "theta_temporal_common_mode_deg": theta_temporal,
                "temporal_adjustment_deg": (
                    float(wrap_180(theta_temporal - raw_theta))
                    if np.isfinite(theta_temporal)
                    else np.nan
                ),
                "n_raw_days": int(len(daily[turbine])),
                "n_common_mode_days": int(n_common),
                "common_mode_daily_std_deg": float(np.std(common_values, ddof=0))
                if common_values
                else np.nan,
                "common_mode_daily_mad_deg": _circular_mad(common_values),
            }
        )

    daily_table = pd.DataFrame(rows).sort_values(["turbine", "date"]).reset_index(drop=True)
    summary = pd.DataFrame(summary_rows).sort_values("turbine").reset_index(drop=True)
    return daily_table, summary, coordinate


def _circular_mad(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    center = float(circular_median_deg(values))
    return float(np.median(np.abs(wrap_180(values - center))))


def run_temporal_common_mode_loto(
    train_ids: Sequence[str],
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
    temporal_theta: Mapping[str, float],
    beta: float = 1.0,
) -> pd.DataFrame:
    """Strict label LOTO with raw theta as the exact release control."""
    rows: list[dict[str, float | str]] = []
    train_ids = list(train_ids)
    if any(turbine not in temporal_theta for turbine in train_ids):
        missing = [turbine for turbine in train_ids if turbine not in temporal_theta]
        raise KeyError("Missing temporal theta for: " + ", ".join(missing))

    for holdout in train_ids:
        fit_ids = [turbine for turbine in train_ids if turbine != holdout]
        for name, coordinate in {
            "release_raw_theta": theta_star,
            "temporal_common_mode": temporal_theta,
        }.items():
            center = legacy_global_anchor(fit_ids, labels, coordinate)
            prediction = predict_from_bundle(
                holdout,
                stage_bundles[holdout],
                center,
                beta,
                dict(coordinate),
            )
            base = center - float(coordinate[holdout])
            metrics = score_prediction(prediction, labels[holdout], base)
            rows.append(
                {
                    "coordinate": name,
                    "holdout": str(holdout),
                    "C": center,
                    "B0": base,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "constant_mae": metrics["constant_mae"],
                    "constant_rmse": metrics["constant_rmse"],
                    "states": metrics["n_states"],
                }
            )
    return pd.DataFrame(rows)
