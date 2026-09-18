"""Stability-aware theta_star diagnostics and anchor sensitivity tests.

This module only changes the absolute reference angle used in
``B0_i = C - theta_star_i``.  Frozen relative-heading boundaries, state
levels, quality weights, and amplitude blending are intentionally untouched.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd

import baseline_openoa_power_vane_updated as b0
from yaw_farm_anchor import _huber_location, fit_farm_anchor, turbine_farm_key
from yaw_model import predict_from_bundle, score_prediction


def estimate_theta_star_series(
    raw: pd.DataFrame,
    half_window_days: int = 10,
    wind_bin_width: float = 1.0,
    angle_bin_width: float = 1.0,
    min_samples_per_angle_bin: int = 20,
    min_angle_bins: int = 8,
    gamma_min: float = -30.0,
    gamma_max: float = 30.0,
) -> pd.Series:
    """Return the daily rolling theta series for anchor diagnostics.

    This is intentionally kept in the exploration-only module.  The release
    model continues to expose only the scalar ``estimate_theta_star`` API.
    """
    data = b0.prepare(raw)

    if data.empty:
        raise ValueError("No operating rows available for theta_star.")

    dates = pd.date_range(
        data["date_dt"].min().normalize(),
        data["date_dt"].max().normalize(),
        freq="D",
    )
    estimates = np.full(len(dates), np.nan, dtype=float)

    for k, date in enumerate(dates):
        window = data[
            data["date_dt"].between(
                date - pd.Timedelta(days=half_window_days),
                date + pd.Timedelta(days=half_window_days),
            )
        ]
        result = b0.estimate_peak_angle(
            window,
            wind_bin_width,
            angle_bin_width,
            min_samples_per_angle_bin,
            min_angle_bins,
            gamma_min,
            gamma_max,
        )
        estimates[k] = result["theta_hat_argmax"]

    series = pd.Series(estimates, index=dates, name="theta_hat_argmax")
    if not series.notna().any():
        raise ValueError("No valid rolling theta_hat_argmax estimates.")
    return series


def theta_location_estimates(
    series: pd.Series,
    trim_fraction: float = 0.10,
) -> dict[str, float]:
    """Return simple long-run location estimators for one theta series."""
    clean = pd.to_numeric(series, errors="coerce").dropna()
    clean = clean[np.isfinite(clean.to_numpy(dtype=float))]
    values = clean.to_numpy(dtype=float)
    if not len(values):
        raise ValueError("No finite theta observations are available.")
    if not 0.0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5).")

    ordered = np.sort(values)
    trim = int(np.floor(trim_fraction * len(ordered)))
    trimmed = ordered[trim: len(ordered) - trim] if trim else ordered
    if not len(trimmed):
        trimmed = ordered

    frame = pd.DataFrame({"value": values}, index=pd.to_datetime(clean.index))
    quarter_means = frame.groupby(frame.index.to_period("Q"))["value"].mean()
    return {
        "global_median": float(np.median(values)),
        "daily_mean": float(np.mean(values)),
        "trimmed_mean": float(np.mean(trimmed)),
        "quarter_balanced_mean": float(quarter_means.mean()),
    }


def bootstrap_theta_locations(
    series: pd.Series,
    n_boot: int = 1000,
    block_freq: str = "Q",
    trim_fraction: float = 0.10,
    random_state: int = 0,
) -> pd.DataFrame:
    """Block-bootstrap theta locations while preserving within-block dependence."""
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    clean = clean[np.isfinite(clean.to_numpy(dtype=float))]
    if clean.empty:
        raise ValueError("No finite theta observations are available.")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive.")

    blocks = [
        block.to_numpy(dtype=float)
        for _, block in clean.groupby(clean.index.to_period(block_freq))
    ]
    if not blocks:
        raise ValueError("No bootstrap blocks are available.")

    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float]] = []
    for _ in range(n_boot):
        sampled_indices = rng.integers(0, len(blocks), size=len(blocks))
        sampled_blocks = [blocks[index] for index in sampled_indices]
        values = np.concatenate(sampled_blocks)
        ordered = np.sort(values)
        trim = int(np.floor(trim_fraction * len(ordered)))
        trimmed = ordered[trim: len(ordered) - trim] if trim else ordered
        if not len(trimmed):
            trimmed = ordered
        rows.append(
            {
                "global_median": float(np.median(values)),
                "daily_mean": float(np.mean(values)),
                "trimmed_mean": float(np.mean(trimmed)),
                "quarter_balanced_mean": float(
                    np.mean([np.mean(blocks[index]) for index in sampled_indices])
                ),
            }
        )
    return pd.DataFrame(rows)


def theta_location_map(
    theta_series: Mapping[str, pd.Series],
    method: str,
    trim_fraction: float = 0.10,
) -> dict[str, float]:
    """Build a turbine-specific theta map from one location estimator."""
    if method not in {
        "global_median",
        "daily_mean",
        "trimmed_mean",
        "quarter_balanced_mean",
    }:
        raise ValueError("Unknown theta location method.")
    return {
        turbine: theta_location_estimates(series, trim_fraction)[method]
        for turbine, series in theta_series.items()
    }


def corrected_mean_anchor_center(
    base_anchor_means: Mapping[str, float],
    theta_map: Mapping[str, float],
    train_ids: Sequence[str],
) -> float:
    """Reproduce the current ``fit_farm_anchor(method='mean')`` centre.

    The release name is ``corrected_mean``; internally its three turbine-level
    centres still pass through the existing robust Huber location.  Keeping
    that detail here makes bootstrap B0 propagation identical to the current
    candidate rather than replacing it with an ordinary arithmetic mean.
    """
    values = [
        float(base_anchor_means[turbine]) + float(theta_map[turbine])
        for turbine in train_ids
    ]
    return float(_huber_location(values))


def run_theta_location_loto(
    train_ids: Sequence[str],
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    theta_series: Mapping[str, pd.Series],
    farm_map: Mapping[str, str] | None = None,
    shrink_kappa: float = 2.0,
    beta: float = 1.0,
    trim_fraction: float = 0.10,
) -> pd.DataFrame:
    """Evaluate theta location estimators with the corrected-mean C fixed."""
    rows: list[dict[str, float | str]] = []
    methods = (
        "global_median",
        "daily_mean",
        "trimmed_mean",
        "quarter_balanced_mean",
    )
    for method in methods:
        theta_map = theta_location_map(theta_series, method, trim_fraction)
        for holdout in train_ids:
            fit_ids = [turbine for turbine in train_ids if turbine != holdout]
            fit = fit_farm_anchor(
                fit_ids,
                stage_bundles,
                labels,
                theta_map,
                farm_map=farm_map,
                method="mean",
                shrink_kappa=shrink_kappa,
            )
            centre = fit.center_for(holdout, farm_map=farm_map)
            prediction = predict_from_bundle(
                holdout,
                stage_bundles[holdout],
                centre,
                beta,
                theta_map,
            )
            metrics = score_prediction(
                prediction,
                labels[holdout],
                centre - theta_map[holdout],
            )
            rows.append(
                {
                    "theta_location": method,
                    "holdout": holdout,
                    "C": centre,
                    "B0": centre - theta_map[holdout],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "constant_mae": metrics["constant_mae"],
                    "constant_rmse": metrics["constant_rmse"],
                    "states": metrics["n_states"],
                }
            )
    return pd.DataFrame(rows)


def theta_mode_summary(
    series: pd.Series,
    mode_band_deg: float = 1.0,
) -> dict[str, float]:
    """Summarize the densest local mode of a rolling theta series.

    The mode is found as the densest interval of radius ``mode_band_deg``.
    The returned centre is the median of the observations in that interval,
    which is more stable than selecting a single quantized argmax value.
    """
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        raise ValueError("No finite theta observations are available.")
    if mode_band_deg <= 0.0:
        raise ValueError("mode_band_deg must be positive.")

    ordered = np.sort(values)
    left = np.searchsorted(ordered, ordered - mode_band_deg, side="left")
    right = np.searchsorted(ordered, ordered + mode_band_deg, side="right")
    counts = right - left
    best_count = int(np.max(counts))
    candidate_indices = np.flatnonzero(counts == best_count)

    # Deterministic tie-break: choose the tied interval with the smallest
    # within-band MAD, then the one closest to the global median.
    global_median = float(np.median(values))
    candidates: list[tuple[float, float, float]] = []
    for index in candidate_indices:
        band = values[np.abs(values - ordered[index]) <= mode_band_deg]
        centre = float(np.median(band))
        mad = float(np.median(np.abs(band - centre)))
        candidates.append((mad, abs(centre - global_median), centre))
    _, _, mode_centre = min(candidates)

    mode_values = values[np.abs(values - mode_centre) <= mode_band_deg]
    mode_mad = float(np.median(np.abs(mode_values - mode_centre)))
    mode_fraction = float(len(mode_values) / len(values))
    return {
        "theta_median_deg": global_median,
        "theta_mode_deg": float(mode_centre),
        "theta_mode_fraction": mode_fraction,
        "theta_mode_mad_deg": mode_mad,
        "theta_mode_min_deg": float(np.min(mode_values)),
        "theta_mode_max_deg": float(np.max(mode_values)),
        "n_valid_windows": float(len(values)),
    }


def _robust_reference(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return float(np.median(values))


def stabilize_theta_star_map(
    theta_series: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
    apply_ids: Sequence[str],
    reference_ids: Sequence[str],
    farm_map: Mapping[str, str] | None = None,
    mode_band_deg: float = 1.0,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Create median, main-mode, and conservative mode-shrink theta maps.

    The reference for a turbine is the robust median of available reference
    turbines in the same farm.  If no same-farm reference exists, the robust
    median across all reference turbines is used.  ``mode_shrink`` gives the
    detected main mode at most 80% weight; the remaining weight is the farm or
    global reference.  This prevents an unstable mode switch from fully
    moving the absolute anchor.
    """
    farm_map = farm_map or {}
    apply_ids = list(apply_ids)
    reference_ids = list(reference_ids)
    reference_values = {
        turbine: float(theta_star[turbine])
        for turbine in reference_ids
        if turbine in theta_star and np.isfinite(theta_star[turbine])
    }
    global_reference = _robust_reference(list(reference_values.values()))
    if not np.isfinite(global_reference):
        raise ValueError("No finite theta reference turbines are available.")

    variants = {
        "global_median": {},
        "main_mode": {},
        "mode_shrink": {},
    }
    rows: list[dict[str, float | str]] = []

    for turbine in apply_ids:
        summary = theta_mode_summary(theta_series[turbine], mode_band_deg)
        farm = farm_map.get(turbine, turbine_farm_key(turbine))
        same_farm = [
            value
            for reference_turbine, value in reference_values.items()
            if farm_map.get(
                reference_turbine,
                turbine_farm_key(reference_turbine),
            ) == farm
        ]
        farm_reference = _robust_reference(same_farm)
        reference = farm_reference if np.isfinite(farm_reference) else global_reference

        # A mode fraction below 35% is not trusted.  Above 65%, mode trust
        # grows linearly but is capped at 80% to retain a reference pull.
        mode_confidence = float(
            np.clip((summary["theta_mode_fraction"] - 0.35) / 0.30, 0.0, 1.0)
        )
        mode_weight = 0.80 * mode_confidence
        theta_mode = float(summary["theta_mode_deg"])
        theta_median = float(summary["theta_median_deg"])
        theta_shrink = float(mode_weight * theta_mode + (1.0 - mode_weight) * reference)

        variants["global_median"][turbine] = theta_median
        variants["main_mode"][turbine] = theta_mode
        variants["mode_shrink"][turbine] = theta_shrink
        rows.append(
            {
                "turbine": turbine,
                "farm": farm,
                "theta_global_median_deg": theta_median,
                "theta_main_mode_deg": theta_mode,
                "theta_reference_deg": reference,
                "theta_mode_fraction": summary["theta_mode_fraction"],
                "theta_mode_mad_deg": summary["theta_mode_mad_deg"],
                "mode_confidence": mode_confidence,
                "mode_weight": mode_weight,
                "theta_mode_shrink_deg": theta_shrink,
            }
        )

    return variants, pd.DataFrame(rows).sort_values("turbine").reset_index(drop=True)


def run_theta_anchor_loto(
    train_ids: Sequence[str],
    stage_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    theta_series: Mapping[str, pd.Series],
    theta_star: Mapping[str, float],
    farm_map: Mapping[str, str] | None = None,
    shrink_kappa: float = 2.0,
    beta: float = 1.0,
    mode_band_deg: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate theta stabilization while holding the C estimator fixed.

    Each fold estimates C with the existing corrected weighted-mean anchor.
    Only the theta map is changed, and the held-out turbine's theta is derived
    from its own unlabeled rolling series plus the training-fold reference.
    """
    rows: list[dict[str, float | str]] = []
    diagnostics: list[dict[str, float | str]] = []
    train_ids = list(train_ids)

    for holdout in train_ids:
        fit_ids = [turbine for turbine in train_ids if turbine != holdout]
        theta_variants, theta_diag = stabilize_theta_star_map(
            theta_series,
            theta_star,
            apply_ids=train_ids,
            reference_ids=fit_ids,
            farm_map=farm_map,
            mode_band_deg=mode_band_deg,
        )
        for variant_name, theta_variant in theta_variants.items():
            fit = fit_farm_anchor(
                fit_ids,
                stage_bundles,
                labels,
                theta_variant,
                farm_map=farm_map,
                method="mean",
                shrink_kappa=shrink_kappa,
            )
            centre = fit.center_for(holdout, farm_map=farm_map)
            prediction = predict_from_bundle(
                holdout,
                stage_bundles[holdout],
                centre,
                beta,
                theta_variant,
            )
            metrics = score_prediction(
                prediction,
                labels[holdout],
                centre - theta_variant[holdout],
            )
            rows.append(
                {
                    "theta_variant": variant_name,
                    "holdout": holdout,
                    "farm": (farm_map or {}).get(
                        holdout,
                        turbine_farm_key(holdout),
                    ),
                    "C": centre,
                    "B0": centre - theta_variant[holdout],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "constant_mae": metrics["constant_mae"],
                    "constant_rmse": metrics["constant_rmse"],
                    "states": metrics["n_states"],
                }
            )
        diagnostics.append(
            theta_diag.assign(
                holdout=holdout,
                fit_reference=",".join(fit_ids),
            )
        )

    loto = pd.DataFrame(rows)
    diagnostics_table = pd.concat(diagnostics, ignore_index=True)
    return loto, diagnostics_table
