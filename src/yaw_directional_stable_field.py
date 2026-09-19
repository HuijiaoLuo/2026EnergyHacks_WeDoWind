"""Sector-standardised, label-free turbine coordinates for the yaw anchor.

This is an exploration-only alternative representation of the static term in
``B0_i = C - theta_star_i``.  It deliberately does not touch the frozen
relative-heading state detector, its corrections, pair-quality weights, or
amplitude blending.

The raw SCADA power-vane optimum is estimated independently in fixed wind
direction sectors.  A turbine coordinate is then its equal-sector robust
offset from the *other turbines'* same-farm sector field.  In symbols,

    theta_{i,s} = field_{farm,s} + u_i + noise_{i,s}

where ``u_i`` is used only as a reparameterised long-run coordinate.  The
labelled centre is fitted separately as ``mean_i(y_i + u_i)``.  Consequently
this experiment changes the representation of the stable field, rather than
introducing a daily or state-dependent predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

import baseline_openoa_power_vane_updated as b0
from baseline_loto_ridge import wrap_180
from fleet_context import circular_median_deg
from yaw_farm_anchor import turbine_farm_key
from yaw_model import predict_from_bundle, score_prediction


@dataclass(frozen=True)
class DirectionalStableFieldConfig:
    """Pre-declared settings for the sector-standardised field audit.

    Every valid sector carries one vote in a turbine coordinate, so common
    prevailing-direction exposure cannot dominate the long-run reference.
    These parameters only govern the diagnostic field estimator.
    """

    sector_width_deg: float = 30.0
    wind_bin_width: float = 1.0
    angle_bin_width: float = 1.0
    min_samples_per_angle_bin: int = 20
    min_angle_bins: int = 8
    gamma_min: float = -30.0
    gamma_max: float = 30.0
    min_peer_turbines: int = 2
    min_valid_sectors: int = 6


@dataclass(frozen=True)
class RelativeFieldAnchorFit:
    """Release-style labelled centre for an arbitrary turbine coordinate."""

    global_center: float
    farm_centers: Mapping[str, float]

    def center_for(
        self,
        turbine: str,
        farm_map: Mapping[str, str] | None = None,
    ) -> float:
        farm = (farm_map or {}).get(turbine, turbine_farm_key(turbine))
        return float(self.farm_centers.get(farm, self.global_center))


def _validate_config(config: DirectionalStableFieldConfig) -> int:
    if config.sector_width_deg <= 0.0:
        raise ValueError("sector_width_deg must be positive.")
    n_sectors = int(round(360.0 / config.sector_width_deg))
    if not np.isclose(n_sectors * config.sector_width_deg, 360.0):
        raise ValueError("sector_width_deg must divide 360 degrees exactly.")
    if config.min_peer_turbines < 1:
        raise ValueError("min_peer_turbines must be at least one.")
    if not 1 <= config.min_valid_sectors <= n_sectors:
        raise ValueError("min_valid_sectors must be within the sector count.")
    return n_sectors


def _sector_index(values: pd.Series, sector_width_deg: float) -> np.ndarray:
    return np.floor(
        (pd.to_numeric(values, errors="coerce").to_numpy(dtype=float) % 360.0)
        / sector_width_deg
    ).astype("float")


def _empty_peak_result() -> dict[str, float | int | str]:
    """Match the power-vane estimator schema when a sector has no rows."""
    return {
        "theta_hat": np.nan,
        "theta_hat_argmax": np.nan,
        "theta_hat_quadratic": np.nan,
        "n_rows": 0,
        "n_curve_bins": 0,
        "status": "no_rows",
        "quadratic_status": "no_rows",
        "fit_r2": np.nan,
        "curvature": np.nan,
        "quadratic_a": np.nan,
        "quadratic_b": np.nan,
        "quadratic_c": np.nan,
        "gamma_min_used": np.nan,
        "gamma_max_used": np.nan,
        "gamma_span": np.nan,
        "argmax_observability": np.nan,
        "quadratic_observability": np.nan,
    }


def estimate_directional_theta_by_sector(
    raw_by_turbine: Mapping[str, pd.DataFrame],
    turbines: Sequence[str] | None = None,
    config: DirectionalStableFieldConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> pd.DataFrame:
    """Estimate full-period power-vane optima in fixed wind-direction sectors.

    This uses only the raw SCADA for each turbine.  It intentionally uses one
    estimate per sector over the full observed period, not a rolling series:
    the goal is a static stable-field coordinate, not a dynamic correction.
    """
    config = config or DirectionalStableFieldConfig()
    n_sectors = _validate_config(config)
    turbine_ids = list(turbines) if turbines is not None else sorted(raw_by_turbine)
    rows: list[dict[str, float | int | str]] = []

    for position, turbine in enumerate(turbine_ids, start=1):
        if turbine not in raw_by_turbine:
            raise KeyError(f"Missing raw SCADA for {turbine}.")
        data = b0.prepare(raw_by_turbine[turbine]).dropna(subset=["WindDir"])
        data = data.copy()
        data["sector"] = _sector_index(data["WindDir"], config.sector_width_deg)

        for sector in range(n_sectors):
            window = data.loc[data["sector"] == sector].copy()
            result = (
                b0.estimate_peak_angle(
                    window,
                    config.wind_bin_width,
                    config.angle_bin_width,
                    config.min_samples_per_angle_bin,
                    config.min_angle_bins,
                    gamma_min=config.gamma_min,
                    gamma_max=config.gamma_max,
                )
                if not window.empty
                else _empty_peak_result()
            )
            rows.append(
                {
                    "turbine": str(turbine),
                    "sector": int(sector),
                    "sector_center_deg": float(
                        (sector + 0.5) * config.sector_width_deg
                    ),
                    **result,
                }
            )

        if progress_callback is not None:
            progress_callback(
                f"  directional field {position}/{len(turbine_ids)}: {turbine}"
            )

    return pd.DataFrame(rows).sort_values(["turbine", "sector"]).reset_index(
        drop=True
    )


def _circular_mad_deg(values: Sequence[float], center: float) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or not np.isfinite(center):
        return np.nan
    return float(np.median(np.abs(wrap_180(values - center))))


def build_directional_stable_field(
    sector_theta: pd.DataFrame,
    farm_map: Mapping[str, str] | None = None,
    config: DirectionalStableFieldConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Remove a leave-one-out farm directional field from sector estimates.

    Returns, in order: augmented turbine-sector estimates, a turbine-level
    summary, and the finite ``turbine -> stable_field_coordinate`` map.
    The peer field is leave-one-out so a turbine cannot define its own
    reference.  Labels never enter this function.
    """
    config = config or DirectionalStableFieldConfig()
    _validate_config(config)
    required = {"turbine", "sector", "theta_hat_argmax"}
    if not required.issubset(sector_theta.columns):
        raise ValueError(f"sector_theta must contain {sorted(required)}")

    out = sector_theta.copy()
    out["turbine"] = out["turbine"].astype(str)
    out["farm"] = out["turbine"].map(
        lambda turbine: (farm_map or {}).get(turbine, turbine_farm_key(turbine))
    )
    out["theta_hat_argmax"] = pd.to_numeric(
        out["theta_hat_argmax"], errors="coerce"
    )
    usable = np.isfinite(out["theta_hat_argmax"].to_numpy(dtype=float))
    if "status" in out:
        usable &= out["status"].fillna("").eq("ok").to_numpy()
    out["field_usable"] = usable
    out["peer_theta_field_deg"] = np.nan
    out["peer_count"] = 0
    out["theta_residual_deg"] = np.nan

    for (farm, sector), indices in out.groupby(["farm", "sector"], sort=True).groups.items():
        index = list(indices)
        part = out.loc[index]
        valid = part.loc[part["field_usable"]]
        for row_index, row in valid.iterrows():
            peers = valid.loc[
                valid["turbine"].ne(row["turbine"]), "theta_hat_argmax"
            ].to_numpy(dtype=float)
            peers = peers[np.isfinite(peers)]
            out.loc[row_index, "peer_count"] = int(len(peers))
            if len(peers) < config.min_peer_turbines:
                continue
            peer_center = float(circular_median_deg(peers))
            out.loc[row_index, "peer_theta_field_deg"] = peer_center
            out.loc[row_index, "theta_residual_deg"] = float(
                wrap_180(float(row["theta_hat_argmax"]) - peer_center)
            )

    summary_rows: list[dict[str, float | int | str]] = []
    coordinates: dict[str, float] = {}
    for turbine, part in out.groupby("turbine", sort=True):
        residual = pd.to_numeric(part["theta_residual_deg"], errors="coerce")
        residual = residual[np.isfinite(residual.to_numpy(dtype=float))]
        n_valid = int(len(residual))
        coordinate = (
            float(circular_median_deg(residual.to_numpy(dtype=float)))
            if n_valid >= config.min_valid_sectors
            else np.nan
        )
        if np.isfinite(coordinate):
            coordinates[str(turbine)] = coordinate
        summary_rows.append(
            {
                "turbine": str(turbine),
                "farm": str(part["farm"].iloc[0]),
                "stable_field_coordinate_deg": coordinate,
                "n_valid_sectors": n_valid,
                "n_total_sectors": int(len(part)),
                "sector_coverage": float(n_valid / len(part)) if len(part) else np.nan,
                "residual_sector_mad_deg": _circular_mad_deg(residual, coordinate),
                "mean_peer_count": float(
                    pd.to_numeric(part.loc[residual.index, "peer_count"], errors="coerce").mean()
                )
                if n_valid
                else np.nan,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("turbine").reset_index(drop=True)
    return out.sort_values(["farm", "turbine", "sector"]).reset_index(drop=True), summary, coordinates


def fit_relative_field_anchor(
    train_ids: Sequence[str],
    labels: Mapping[str, pd.Series],
    coordinate: Mapping[str, float],
    farm_map: Mapping[str, str] | None = None,
) -> RelativeFieldAnchorFit:
    """Fit ``A_f = mean_i(mean(y_i) + u_i)`` from labelled turbines only.

    This intentionally mirrors the release notebook's unweighted turbine
    mean.  It is not the corrected FarmAnchor aggregation, so a result can be
    attributed solely to the stable-field coordinate representation.
    """
    rows: list[dict[str, float | str]] = []
    for turbine in train_ids:
        if turbine not in labels or turbine not in coordinate:
            raise KeyError(f"Missing label or stable-field coordinate for {turbine}.")
        label_mean = float(pd.to_numeric(labels[turbine], errors="coerce").mean())
        value = label_mean + float(coordinate[turbine])
        if not np.isfinite(value):
            raise ValueError(f"Non-finite relative-field anchor for {turbine}.")
        rows.append(
            {
                "turbine": str(turbine),
                "farm": (farm_map or {}).get(turbine, turbine_farm_key(turbine)),
                "anchor_value": value,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No labelled turbines were supplied.")
    farm_centers = {
        str(farm): float(part["anchor_value"].mean())
        for farm, part in frame.groupby("farm", sort=True)
    }
    return RelativeFieldAnchorFit(
        global_center=float(frame["anchor_value"].mean()),
        farm_centers=farm_centers,
    )


def run_directional_stable_field_loto(
    train_ids: Sequence[str],
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    release_theta: Mapping[str, float],
    directional_coordinate: Mapping[str, float],
    farm_map: Mapping[str, str] | None = None,
    beta: float = 1.0,
) -> pd.DataFrame:
    """Compare the frozen release theta coordinate with the directional one.

    Each fold sees the held-out turbine's raw SCADA field coordinate but not
    its yaw labels.  The ``release_raw_theta`` row uses the exact release
    calibration ``mean_i(mean(y_i) + theta_i)`` as a reparameterisation
    control; any difference in the experimental row is therefore due only to
    the sector-standardised coordinate.
    """
    rows: list[dict[str, float | str]] = []
    train_ids = list(train_ids)
    for holdout in train_ids:
        fit_ids = [turbine for turbine in train_ids if turbine != holdout]
        release_fit = fit_relative_field_anchor(
            fit_ids, labels, release_theta, farm_map=farm_map
        )
        field_fit = fit_relative_field_anchor(
            fit_ids, labels, directional_coordinate, farm_map=farm_map
        )
        variants = {
            "release_raw_theta": (release_fit, release_theta),
            "directional_stable_field": (field_fit, directional_coordinate),
        }
        for name, (fit, coordinate) in variants.items():
            center = fit.center_for(holdout, farm_map=farm_map)
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
                    "farm": (farm_map or {}).get(holdout, turbine_farm_key(holdout)),
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
