from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import baseline_openoa_power_vane_updated as b0
from baseline_loto_ridge import wrap_180
from fleet_context import (
    FleetConfig,
    circular_mean_deg,
    circular_median_deg,
)
from yaw_relative_state import (
    RelativeStateConfig,
    clusters_from_boundaries,
    fleet_relative_series,
    label_free_boundaries,
)
from copy import deepcopy
from typing import Mapping


def estimate_theta_star(
    raw: pd.DataFrame,
    half_window_days: int = 10,
    wind_bin_width: float = 1.0,
    angle_bin_width: float = 1.0,
    min_samples_per_angle_bin: int = 20,
    min_angle_bins: int = 8,
    gamma_min: float = -30.0,
    gamma_max: float = 30.0,
) -> float:
    """
    Estimate the long-term B0 angle used by the constant-line prior.

    For each calendar day, use a centred rolling SCADA window and the frozen
    empirical power-vs-vane argmax estimator. ``theta_star`` is the median of
    all finite daily argmax estimates.
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

    valid = estimates[np.isfinite(estimates)]

    if len(valid) == 0:
        raise ValueError("No valid rolling theta_hat_argmax estimates.")

    return float(np.median(valid))


def estimate_theta_star_from_vane(
    aggregated: pd.DataFrame,
    half_window_days: int = 10,
    wind_bin_width: float = 1.0,
    angle_bin_width: float = 1.0,
    min_samples_per_angle_bin: int = 20,
    min_angle_bins: int = 8,
    gamma_min: float = -30.0,
    gamma_max: float = 30.0,
    wind_min: float = 3.0,
    wind_max: float = 13.0,
) -> float:
    """Estimate ``theta_star`` from an already-cleaned vane aggregate.

    This is deliberately separate from :func:`estimate_theta_star`.  The
    latter is for raw SCADA and applies the historical operating filter via
    ``b0.prepare``.  Here the cleaning mask has already been applied, so the
    direct aggregated ``vane = wrap180(WindDir - NacDir)`` is used.  Only the
    fixed 3--13 wind-speed identifiable regime is retained; no second
    quantile/pitch filter or subtraction of separately averaged headings is
    made.
    """
    frame = aggregated.copy()
    if "vane" not in frame.columns:
        if not {"WindDir", "NacDir"}.issubset(frame.columns):
            raise ValueError("Aggregated frame needs vane or WindDir/NacDir.")
        frame["vane"] = wrap_180(frame["WindDir"] - frame["NacDir"])

    time_col = next(
        (c for c in ("timestamp", "ts", "date", "date_dt") if c in frame),
        None,
    )
    if time_col is None:
        if isinstance(frame.index, pd.DatetimeIndex):
            frame["date_dt"] = frame.index
        else:
            raise ValueError("Aggregated frame needs a timestamp/date column.")
    else:
        frame["date_dt"] = pd.to_datetime(frame[time_col])

    required = ["date_dt", "WindSpeed", "Power", "vane"]
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
    # Keep the same identifiable partial-load wind regime as the existing
    # estimator, without reapplying its quantile/pitch filter.
    frame = frame[frame["WindSpeed"].between(wind_min, wind_max)]
    if frame.empty:
        raise ValueError("No valid cleaned rows available for theta_star.")

    frame["gamma"] = wrap_180(frame["vane"])
    dates = pd.date_range(
        frame["date_dt"].min().normalize(),
        frame["date_dt"].max().normalize(),
        freq="D",
    )
    estimates = []
    for date in dates:
        window = frame[
            frame["date_dt"].between(
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
        theta = result["theta_hat_argmax"]
        if np.isfinite(theta):
            estimates.append(float(theta))

    if not estimates:
        raise ValueError("No valid rolling theta_hat_argmax estimates.")
    return float(np.median(np.asarray(estimates, dtype=float)))


def estimate_theta_star_map(
    raw: dict[str, pd.DataFrame],
    turbine_ids: list[str],
) -> dict[str, float]:
    """Estimate ``theta_star`` for a collection of turbines."""
    return {
        turbine: estimate_theta_star(raw[turbine])
        for turbine in turbine_ids
    }


@dataclass(frozen=True)
class ModelConfig:
    """
    Constant-line prior + sparse state corrections.

    The constant model is the default prediction. Dynamic relative-heading
    states are allowed to move away from it only when label-free evidence is
    sufficiently strong.
    """

    beta_prior: float = 1.0
    beta_ridge: float = 12.0
    beta_min: float = 0.8
    beta_max: float = 1.2

    # With only three labelled turbines, do not fit beta from one or two
    # surviving events. Fall back to the physical prior beta=1 instead.
    min_beta_changes: int = 3
    beta_window_days: int = 21

    # A detected event may exist diagnostically but cannot create a yaw state
    # unless its label-free evidence clears this threshold.
    min_event_confidence: float = 0.25
    max_event_shrinkage: float = 0.90

    # Same-site synchronous events are treated as site/reference common mode.
    site_common_mode_window_days: int = 7
    site_common_mode_min_turbines: int = 3


def _apply_confidence_gate(
    boundaries: pd.DataFrame,
    config: ModelConfig,
) -> pd.DataFrame:
    """
    Apply the constant-prior gate AFTER site-level common-mode rejection.

    Weak candidate events remain in diagnostics but are not allowed to create
    a new dynamic state.
    """
    boundaries = boundaries.copy()

    if boundaries.empty:
        if "shrinkage" not in boundaries.columns:
            boundaries["shrinkage"] = pd.Series(dtype=float)
        return boundaries

    if "shrinkage" not in boundaries.columns:
        boundaries["shrinkage"] = 0.0

    candidate = boundaries["accepted"].fillna(False)
    confidence = boundaries["event_confidence"].fillna(0.0)

    weak = candidate & (
        confidence < config.min_event_confidence
    )
    if weak.any():
        boundaries.loc[weak, "accepted"] = False
        boundaries.loc[weak, "reason"] = "low_confidence_prior"

    accepted = boundaries["accepted"].fillna(False)
    boundaries["shrinkage"] = 0.0
    boundaries.loc[accepted, "shrinkage"] = np.clip(
        boundaries.loc[accepted, "event_confidence"].to_numpy(dtype=float),
        0.0,
        config.max_event_shrinkage,
    )

    return boundaries


def _state_confidence_from_boundaries(
    accepted: pd.DataFrame,
    n_states: int,
) -> np.ndarray:
    """
    Convert boundary shrinkage into state confidence.

    A first/last state is supported by its adjacent boundary. An interior
    state must be supported by both surrounding boundaries, so the smaller
    adjacent confidence is used.
    """
    if n_states <= 1 or accepted.empty:
        return np.zeros(max(n_states, 1), dtype=float)

    # Boundary proposals may repeat the same daily date.  Collapse those
    # duplicates conservatively before mapping boundary confidence to states;
    # this keeps the confidence vector aligned with the deduplicated clusters.
    boundary_dates = pd.DatetimeIndex(
        pd.to_datetime(accepted.index)
    ).normalize()
    shrinkage = pd.Series(
        accepted["shrinkage"].to_numpy(dtype=float),
        index=boundary_dates,
    )
    s = shrinkage.groupby(level=0, sort=True).min().to_numpy(dtype=float)

    # Defensive shape guard for unusual boundary tables.  Normally
    # len(s) == n_states - 1 after date de-duplication.
    needed = max(int(n_states) - 1, 0)
    if len(s) > needed:
        s = s[:needed]
    elif len(s) < needed:
        if len(s) == 0:
            s = np.zeros(needed, dtype=float)
        else:
            s = np.pad(s, (0, needed - len(s)), mode="edge")

    confidence = np.zeros(n_states, dtype=float)
    confidence[0] = s[0]
    confidence[-1] = s[-1]

    if n_states > 2:
        confidence[1:-1] = np.minimum(
            s[:-1],
            s[1:],
        )

    return confidence


def _direct_state_prior_correction(
    obs: pd.DataFrame,
    boundaries: pd.DataFrame,
    force_full_amplitude: bool = False,
    day_quality: pd.Series | None = None,
) -> pd.Series:
    """
    Estimate each state directly relative to the global relative-heading centre.

    This replaces cumulative event summation. Therefore several small accepted
    events cannot accumulate into an unbounded drift away from B0.
    """
    correction = pd.Series(
        0.0,
        index=obs.index,
        dtype=float,
    )

    accepted = boundaries[
        boundaries["accepted"].fillna(False)
    ].sort_index()

    if accepted.empty:
        return correction

    signal = obs["relative_heading_smooth"].copy()
    valid_signal = signal.dropna()

    if valid_signal.empty:
        return correction

    global_centre = circular_median_deg(valid_signal)

    clusters = obs["cluster"].astype(int)
    state_ids = np.sort(clusters.unique())
    state_confidence = _state_confidence_from_boundaries(
        accepted,
        len(state_ids),
    )
    if force_full_amplitude:
        # Stable-boundary experiments may test the measured state difference
        # without using confidence as an amplitude multiplier.  Boundary
        # acceptance and common-mode/sensor vetoes have already happened.
        state_confidence = np.ones_like(state_confidence, dtype=float)

    state_offset: dict[int, float] = {}

    for position, state_id in enumerate(state_ids):
        values = signal.loc[
            clusters.eq(state_id)
        ].dropna()

        if values.empty:
            state_offset[int(state_id)] = 0.0
            continue

        if day_quality is not None:
            weights = day_quality.reindex(values.index).to_numpy(dtype=float)
            weighted_level = weighted_circular_median_deg(
                values.to_numpy(dtype=float),
                weights,
            )
            state_level = (
                weighted_level
                if np.isfinite(weighted_level)
                else circular_median_deg(values)
            )
        else:
            state_level = circular_median_deg(values)
        raw_offset = float(
            wrap_180(state_level - global_centre)
        )

        state_offset[int(state_id)] = (
            raw_offset
            * float(state_confidence[position])
        )

    correction = clusters.map(state_offset).astype(float)

    # B0 is the long-term absolute centre. Keep the dynamic correction
    # duration-weighted around zero rather than allowing state offsets to
    # shift the whole turbine level.
    finite = np.isfinite(correction.to_numpy(dtype=float))
    if finite.any():
        centre = float(
            correction.to_numpy(dtype=float)[finite].mean()
        )
        correction = correction - centre

    return correction.fillna(0.0)


def weighted_circular_median_deg(
    values: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series,
) -> float:
    """Return a missing-safe weighted circular median in degrees.

    The median is taken after unwrapping values around their circular mean;
    this keeps values near +/-180 degrees on the same local axis.  It is used
    only for state-level aggregation, not for boundary detection.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(good):
        return np.nan

    values = values[good]
    weights = weights[good]
    origin = circular_mean_deg(values)
    local = wrap_180(values - origin)
    order = np.argsort(local)
    cumulative = np.cumsum(weights[order])
    position = int(
        np.searchsorted(
            cumulative,
            0.5 * cumulative[-1],
            side="left",
        )
    )
    position = min(position, len(order) - 1)
    return float(wrap_180(origin + local[order[position]]))


def _pair_quality_day_weights(
    bundle: dict,
    spread_scale_deg: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build continuous day weights from fixed pair quality diagnostics.

    Pair residuals are already sector-corrected in ``pair_daily``.  The
    returned quality is a soft reliability weight: robust cross-pair spread
    penalises disagreement, while finite-pair coverage and fixed historical
    pair reliability penalise missing or weak pairs.  Rows with no finite pair
    never reach ``nanmedian``.
    """
    pair_daily = bundle.get("pair_daily")
    if pair_daily is None or pair_daily.empty:
        index = bundle["observables"].index
        empty = pd.Series(0.0, index=index, dtype=float)
        return empty, empty.copy(), empty.copy()

    values = pair_daily.to_numpy(dtype=float)
    has_pair = np.isfinite(values).any(axis=1)
    pair_median = np.full(len(values), np.nan, dtype=float)
    if has_pair.any():
        pair_median[has_pair] = np.nanmedian(
            values[has_pair],
            axis=1,
        )

    deviations = np.abs(wrap_180(values - pair_median[:, None]))
    pair_spread = np.full(len(values), np.nan, dtype=float)
    if has_pair.any():
        pair_spread[has_pair] = np.nanmedian(
            deviations[has_pair],
            axis=1,
        )

    n_pairs = np.isfinite(values).sum(axis=1).astype(float)
    coverage = n_pairs / max(values.shape[1], 1)

    diagnostics = bundle.get("pair_diagnostics", pd.DataFrame())
    fixed = diagnostics.reindex(pair_daily.columns).get(
        "reliability_weight",
        pd.Series(1.0, index=pair_daily.columns),
    )
    fixed = pd.to_numeric(fixed, errors="coerce").to_numpy(dtype=float)
    fixed = np.where(np.isfinite(fixed) & (fixed > 0.0), fixed, 0.0)
    total_fixed = float(fixed.sum())
    if total_fixed > 0.0:
        weighted_coverage = (
            np.isfinite(values).astype(float) @ fixed
        ) / total_fixed
    else:
        weighted_coverage = coverage.copy()

    scale = max(float(spread_scale_deg), 1e-6)
    q_spread = 1.0 / (1.0 + (pair_spread / scale) ** 2)
    q_day = q_spread * np.sqrt(coverage * weighted_coverage)
    q_day = np.where(
        has_pair & np.isfinite(q_day),
        np.clip(q_day, 0.0, 1.0),
        0.0,
    )

    index = pair_daily.index
    return (
        pd.Series(q_day, index=index, dtype=float),
        pd.Series(pair_spread, index=index, dtype=float),
        pd.Series(weighted_coverage, index=index, dtype=float),
    )


def apply_pair_quality_state_weights(
    bundles: Mapping[str, dict],
    model_config: ModelConfig | None = None,
    spread_scale_deg: float = 2.5,
    force_full_amplitude: bool = False,
) -> dict[str, dict]:
    """Re-estimate frozen state levels with continuous pair-quality weights.

    Boundaries and clusters are kept exactly as supplied.  This is an
    amplitude/state-level ablation: no detector, sector baseline, beta, or
    absolute anchor is recalculated.  A returned bundle remains compatible
    with ``predict_from_bundle`` and the existing LOTO helpers.
    """
    if model_config is None:
        model_config = ModelConfig()

    out: dict[str, dict] = {}
    for turbine, bundle in bundles.items():
        temporary = deepcopy(bundle)
        obs = temporary["observables"].copy()
        boundaries = temporary["boundaries"].copy()
        quality, spread, weighted_coverage = _pair_quality_day_weights(
            temporary,
            spread_scale_deg,
        )
        quality = quality.reindex(obs.index).fillna(0.0)
        obs["state_quality_weight"] = quality
        obs["state_pair_spread_deg"] = spread.reindex(obs.index)
        obs["state_weighted_pair_coverage"] = weighted_coverage.reindex(
            obs.index
        )
        obs["relative_prior_correction"] = _direct_state_prior_correction(
            obs,
            boundaries,
            force_full_amplitude=force_full_amplitude,
            day_quality=quality,
        )
        temporary["observables"] = obs
        temporary["boundaries"] = boundaries
        temporary["state_level_weighting"] = "soft_pair_quality"
        temporary["state_quality_spread_scale_deg"] = float(
            spread_scale_deg
        )
        out[turbine] = temporary

    return out


def _rebuild_prior_trajectory(
    bundle: dict,
    config: ModelConfig,
    apply_confidence_gate: bool,
    force_full_amplitude: bool = False,
) -> dict:
    """
    Rebuild clusters and the B0-centred state correction trajectory.

    The confidence threshold is deliberately applied only after the site-level
    common-mode veto, so weak confidence filtering cannot prevent detection of
    a synchronous site-wide event.
    """
    obs = bundle["observables"].copy()
    boundaries = bundle["boundaries"].copy()

    if apply_confidence_gate:
        boundaries = _apply_confidence_gate(
            boundaries,
            config,
        )
    else:
        # Rebuild from the current accepted set. Rejected boundaries must not
        # retain stale shrinkage from a previous model stage.
        boundaries["shrinkage"] = 0.0
        accepted = boundaries["accepted"].fillna(False) if len(boundaries) else pd.Series(dtype=bool)
        if len(boundaries) and accepted.any():
            boundaries.loc[accepted, "shrinkage"] = np.clip(
                boundaries.loc[accepted, "event_confidence"].to_numpy(dtype=float),
                0.0,
                config.max_event_shrinkage,
            )

    obs["cluster"] = clusters_from_boundaries(
        obs.index,
        boundaries,
    )

    obs["relative_prior_correction"] = (
        _direct_state_prior_correction(
            obs,
            boundaries,
            force_full_amplitude=force_full_amplitude,
        )
    )

    bundle["observables"] = obs
    bundle["boundaries"] = boundaries
    return bundle


def build_turbine_bundle(
    turbine: str,
    binned: dict[str, pd.DataFrame],
    layout: pd.DataFrame,
    fleet_config: FleetConfig,
    relative_config: RelativeStateConfig,
    model_config: ModelConfig | None = None,
) -> dict:
    """
    Build one label-free relative-heading bundle.

    Confidence gating is postponed until after the cross-turbine site veto.
    """
    if model_config is None:
        model_config = ModelConfig()

    daily, pair_diagnostics, pair_daily = (
        fleet_relative_series(
            turbine,
            binned,
            layout,
            fleet_config,
            relative_config,
        )
    )

    if daily.empty:
        target_index = binned[turbine].index
        if len(target_index):
            days = pd.date_range(
                target_index.min().normalize(),
                target_index.max().normalize(),
                freq="D",
            )
        else:
            days = pd.DatetimeIndex([])

        daily = pd.DataFrame(index=days)
        daily["relative_heading"] = np.nan
        daily["relative_heading_smooth"] = np.nan
        daily["background_residual"] = np.nan
        daily["background_residual_smooth"] = np.nan

    pair_weight = pair_diagnostics.get(
        "reliability_weight",
        pd.Series(dtype=float),
    ).to_numpy(dtype=float)

    boundaries = label_free_boundaries(
        daily,
        pair_daily,
        pair_weight,
        relative_config,
    )

    bundle = {
        "turbine": turbine,
        "observables": daily.copy(),
        "pair_diagnostics": pair_diagnostics,
        "pair_daily": pair_daily,
        "boundaries": boundaries,
    }

    # Keep preliminary accepted flags intact for the subsequent site-wide veto.
    return _rebuild_prior_trajectory(
        bundle,
        model_config,
        apply_confidence_gate=False,
    )


def apply_full_amplitude_stable_states(
    bundles: Mapping[str, dict],
    model_config: ModelConfig | None = None,
) -> dict[str, dict]:
    """Test full measured amplitude on already stability-approved boundaries.

    This is an explicit comparison stage, not a replacement for the
    conservative default.  Boundary acceptance, site/common-mode vetoes and
    state dates are preserved; only the confidence multiplier on accepted
    stable boundaries is removed from the state amplitude.
    """
    if model_config is None:
        model_config = ModelConfig()

    out: dict[str, dict] = {}
    for turbine, bundle in bundles.items():
        boundaries = bundle["boundaries"].copy()
        if boundaries.empty:
            out[turbine] = deepcopy(bundle)
            out[turbine]["amplitude_mode"] = "full_stable"
            continue

        active = pd.Series(False, index=boundaries.index, dtype=bool)
        for column in ("prediction_active", "accepted"):
            if column in boundaries:
                active = boundaries[column].fillna(False).astype(bool)
                break
        if "stability_active" in boundaries:
            active &= boundaries["stability_active"].fillna(False).astype(bool)

        prediction_boundaries = boundaries.copy()
        prediction_boundaries["accepted"] = active
        if "event_confidence" in prediction_boundaries:
            prediction_boundaries.loc[active, "event_confidence"] = 1.0

        temporary = deepcopy(bundle)
        temporary["boundaries"] = prediction_boundaries
        rebuilt = _rebuild_prior_trajectory(
            temporary,
            model_config,
            apply_confidence_gate=False,
            force_full_amplitude=True,
        )
        # Keep the original stable-boundary diagnostics; the rebuilt
        # observables contain the candidate full-amplitude trajectory.
        rebuilt["boundaries"] = boundaries
        rebuilt["amplitude_mode"] = "full_stable"
        out[turbine] = rebuilt

    return out


def fit_stable_amplitude_weight(
    train_ids: list[str],
    stable_bundles: Mapping[str, dict],
    full_bundles: Mapping[str, dict],
    labels: Mapping[str, pd.Series],
    window_days: int = 21,
    prior: float = 0.5,
    ridge: float = 4.0,
) -> tuple[float, int]:
    """Fit one conservative blend weight from labelled state changes only.

    The weight interpolates between the default shrunk correction and the
    measured full-amplitude correction.  A fixed prior/ridge is used because
    only a few turbine-level transitions are available; no target labels enter
    this fit.
    """
    if not 0.0 <= prior <= 1.0 or ridge < 0.0:
        raise ValueError("prior must be in [0, 1] and ridge must be non-negative")

    design = []
    residual = []
    for turbine in train_ids:
        stable = stable_bundles[turbine]
        full = full_bundles[turbine]
        table = stable["boundaries"]
        if table is None or table.empty:
            continue

        active_col = (
            "prediction_active"
            if "prediction_active" in table.columns
            else "accepted"
        )
        dates = table.index[table[active_col].fillna(False)]
        stable_correction = stable["observables"][
            "relative_prior_correction"
        ]
        full_correction = full["observables"][
            "relative_prior_correction"
        ]

        for date in dates:
            stable_delta = _daily_event_change(
                stable_correction,
                pd.Timestamp(date),
                window_days,
            )
            full_delta = _daily_event_change(
                full_correction,
                pd.Timestamp(date),
                window_days,
            )
            yaw_delta = _daily_event_change(
                labels[turbine],
                pd.Timestamp(date),
                window_days,
            )
            if not all(np.isfinite(v) for v in (stable_delta, full_delta, yaw_delta)):
                continue

            extra = float(full_delta - stable_delta)
            if abs(extra) < 0.25:
                continue
            design.append(extra)
            residual.append(float(yaw_delta + stable_delta))

    if not design:
        return float(prior), 0

    x = np.asarray(design, dtype=float)
    y = np.asarray(residual, dtype=float)
    weight = (
        float(ridge) * float(prior) - float(np.dot(x, y))
    ) / (
        float(np.dot(x, x)) + float(ridge)
    )
    return float(np.clip(weight, 0.0, 1.0)), int(len(x))


def blend_stable_amplitude(
    stable_bundles: Mapping[str, dict],
    full_bundles: Mapping[str, dict],
    weight: float,
) -> dict[str, dict]:
    """Blend stable and full state corrections without changing boundaries."""
    weight = float(weight)
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")

    out = deepcopy(dict(stable_bundles))
    for turbine, stable in out.items():
        if turbine not in full_bundles:
            raise KeyError(f"Missing full-amplitude bundle for {turbine}")
        stable_correction = stable["observables"][
            "relative_prior_correction"
        ].astype(float)
        full_correction = full_bundles[turbine]["observables"][
            "relative_prior_correction"
        ].reindex(stable_correction.index).astype(float)
        stable["observables"]["relative_prior_correction"] = (
            stable_correction
            + weight * (full_correction - stable_correction)
        ).fillna(stable_correction).fillna(0.0)
        stable["amplitude_mode"] = f"blend_{weight:.3f}"

    return out


def apply_site_common_mode_veto(
    bundles: dict[str, dict],
    config: ModelConfig,
) -> dict[str, dict]:
    """
    Reject accepted yaw candidates that occur across many turbines at the same
    site within a short window, then apply the constant-prior confidence gate.
    """
    events = []

    for turbine, bundle in bundles.items():
        table = bundle["boundaries"]
        if table is None or table.empty:
            continue

        accepted = table[
            table["accepted"].fillna(False)
        ]
        for date, row in accepted.iterrows():
            events.append({
                "turbine": turbine,
                "site": turbine.split("_", 1)[0],
                "date": pd.Timestamp(date),
                "change": float(row["change"]),
            })

    if events:
        events = pd.DataFrame(events)

        veto_keys: set[
            tuple[str, pd.Timestamp]
        ] = set()
        window = config.site_common_mode_window_days

        for _, site_events in events.groupby("site"):
            rows = site_events.to_dict("records")

            for event in rows:
                nearby = {
                    other["turbine"]
                    for other in rows
                    if abs(
                        (
                            other["date"]
                            - event["date"]
                        ).days
                    )
                    <= window
                }

                if (
                    len(nearby)
                    >= config.site_common_mode_min_turbines
                ):
                    for other in rows:
                        if (
                            other["turbine"] in nearby
                            and abs(
                                (
                                    other["date"]
                                    - event["date"]
                                ).days
                            )
                            <= window
                        ):
                            veto_keys.add(
                                (
                                    other["turbine"],
                                    other["date"],
                                )
                            )

        for turbine, date in veto_keys:
            table = bundles[turbine]["boundaries"].copy()
            if date not in table.index:
                continue

            table.at[date, "accepted"] = False
            table.at[date, "reason"] = "site_common_mode"
            table.at[date, "event_confidence"] = 0.0
            table.at[date, "shrinkage"] = 0.0
            bundles[turbine]["boundaries"] = table

    # The low-confidence prior gate is applied here, after common-mode logic.
    for turbine in bundles:
        bundles[turbine] = _rebuild_prior_trajectory(
            bundles[turbine],
            config,
            apply_confidence_gate=True,
        )

    return bundles


def _daily_event_change(
    series: pd.Series,
    date: pd.Timestamp,
    window_days: int,
) -> float:
    """Wrapped robust before/after change for a daily angular series."""
    daily = series.groupby(
        series.index.normalize()
    ).median().sort_index()

    pre = daily.reindex(
        pd.date_range(
            date - pd.Timedelta(days=window_days),
            date - pd.Timedelta(days=1),
        )
    ).dropna()
    post = daily.reindex(
        pd.date_range(
            date,
            date + pd.Timedelta(days=window_days - 1),
        )
    ).dropna()

    if len(pre) < 5 or len(post) < 5:
        return np.nan

    return float(wrap_180(np.median(post) - np.median(pre)))


def fit_global_beta(
    train_ids: list[str],
    bundles: dict[str, dict],
    labels: dict[str, pd.Series],
    theta_star: dict[str, float],
    config: ModelConfig,
) -> tuple[float, float]:
    """
    Fit C and, only with enough reliable labelled transitions, one global beta.

    Otherwise beta remains the physical prior 1.0.
    """
    C = float(
        np.mean([
            labels[t].mean()
            + theta_star[t]
            for t in train_ids
        ])
    )

    x, y = [], []

    for turbine in train_ids:
        table = bundles[turbine]["boundaries"]

        if table is None or table.empty:
            continue

        active_col = (
            "prediction_active"
            if "prediction_active" in table.columns
            else "accepted"
        )
        accepted = table[table[active_col].fillna(False)]
        correction = bundles[turbine]["observables"][
            "relative_prior_correction"
        ]

        for date, _ in accepted.iterrows():
            relative_delta = _daily_event_change(
                correction,
                pd.Timestamp(date),
                config.beta_window_days,
            )

            if (
                not np.isfinite(relative_delta)
                or abs(relative_delta) < 0.25
            ):
                continue

            yaw_delta = _daily_event_change(
                labels[turbine],
                pd.Timestamp(date),
                config.beta_window_days,
            )

            if not np.isfinite(yaw_delta):
                continue

            x.append(relative_delta)
            y.append(yaw_delta)

    if len(x) < config.min_beta_changes:
        return C, config.beta_prior

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    beta = (
        config.beta_ridge
        * config.beta_prior
        - np.dot(x, y)
    ) / (
        np.dot(x, x)
        + config.beta_ridge
    )

    return (
        C,
        float(
            np.clip(
                beta,
                config.beta_min,
                config.beta_max,
            )
        ),
    )


def predict_from_bundle(
    turbine: str,
    bundle: dict,
    C: float,
    beta: float,
    theta_star: dict[str, float],
) -> pd.DataFrame:
    """
    Constant-line prior plus direct, evidence-weighted state offsets.
    """
    obs = bundle["observables"].copy()

    correction = obs["relative_prior_correction"].fillna(0.0)
    levels_per_cluster = (
        pd.DataFrame({"cluster": obs["cluster"], "correction": correction.round(10)})
        .groupby("cluster")["correction"]
        .nunique()
    )
    if len(levels_per_cluster) and int(levels_per_cluster.max()) > 1:
        raise RuntimeError(
            "relative_prior_correction is stale relative to the current state clusters"
        )

    base = C - theta_star[turbine]
    obs["constant_prior"] = base
    obs["prediction"] = (
        base
        - beta
        * correction
    )
    return obs


def _boundary_scores(
    prediction: pd.DataFrame,
    target: pd.Series,
    tolerance_days: int = 21,
) -> dict:
    truth = (
        target.groupby(
            target.index.normalize()
        )
        .median()
        .sort_index()
    )
    truth_change = pd.Series(
        truth.to_numpy()[1:]
        - truth.to_numpy()[:-1],
        index=truth.index[1:],
    )
    true_dates = list(
        truth_change.index[
            truth_change.abs() >= 0.5
        ]
    )

    clusters = (
        prediction["cluster"]
        .groupby(
            prediction.index.normalize()
        )
        .last()
        .sort_index()
    )
    pred_dates = list(
        clusters.index[1:][
            clusters.to_numpy()[1:]
            != clusters.to_numpy()[:-1]
        ]
    )

    if not true_dates and not pred_dates:
        return {
            "boundary_precision": np.nan,
            "boundary_recall": np.nan,
            "true_boundaries": 0,
            "predicted_boundaries": 0,
        }

    unmatched = set(true_dates)
    matched = 0

    for date in pred_dates:
        nearby = [
            d
            for d in unmatched
            if abs((d - date).days)
            <= tolerance_days
        ]
        if nearby:
            closest = min(
                nearby,
                key=lambda d: abs(
                    (d - date).days
                ),
            )
            unmatched.remove(closest)
            matched += 1

    return {
        "boundary_precision": (
            matched
            / max(len(pred_dates), 1)
        ),
        "boundary_recall": (
            matched
            / max(len(true_dates), 1)
        ),
        "true_boundaries": len(true_dates),
        "predicted_boundaries": len(pred_dates),
    }


def score_prediction(
    prediction: pd.DataFrame,
    label: pd.Series,
    base_level: float,
) -> dict:
    scored = prediction.join(
        label.rename("target"),
        how="inner",
    ).dropna(
        subset=["prediction", "target"]
    )

    if scored.empty:
        return {
            "mae": np.nan,
            "rmse": np.nan,
            "constant_mae": np.nan,
            "constant_rmse": np.nan,
            "n_states": 0,
            "boundary_precision": np.nan,
            "boundary_recall": np.nan,
            "true_boundaries": 0,
            "predicted_boundaries": 0,
        }

    error = (
        scored["prediction"]
        - scored["target"]
    )
    constant_error = (
        base_level
        - scored["target"]
    )

    metrics = {
        "mae": float(
            error.abs().mean()
        ),
        "rmse": float(
            np.sqrt(
                np.mean(error ** 2)
            )
        ),
        "constant_mae": float(
            constant_error.abs().mean()
        ),
        "constant_rmse": float(
            np.sqrt(
                np.mean(
                    constant_error ** 2
                )
            )
        ),
        "n_states": int(
            prediction["cluster"].nunique()
        ),
    }
    metrics.update(
        _boundary_scores(
            prediction,
            scored["target"],
        )
    )
    return metrics


# ---------------------------------------------------------------------------
# Conservative multiscale veto
# ---------------------------------------------------------------------------
def apply_multiscale_sign_veto(
    bundles: Mapping[str, dict],
    accepted_audit: pd.DataFrame,
    model_config: ModelConfig,
) -> dict[str, dict]:
    """
    Conservative V7.6 experiment.

    Only events already accepted by frozen V7.5 are eligible for removal.
    `accepted_audit` must contain one row per audited accepted event with:
        turbine, event_date, scale_reject

    No rejected V7.5 event is resurrected.
    No event confidence is re-fit.
    After vetoing, clusters and the direct B0-centred state correction are
    rebuilt from the remaining accepted boundaries.
    """
    required = {"turbine", "event_date", "scale_reject"}
    missing = required - set(accepted_audit.columns)
    if missing:
        raise KeyError(
            f"accepted_audit missing required columns: {sorted(missing)}"
        )

    out = deepcopy(dict(bundles))

    reject_rows = accepted_audit[
        accepted_audit["scale_reject"].fillna(False)
    ].copy()

    reject_map: dict[str, set[pd.Timestamp]] = {}

    for turbine, group in reject_rows.groupby("turbine"):
        reject_map[turbine] = {
            pd.Timestamp(d).normalize()
            for d in group["event_date"]
        }

    for turbine, bundle in out.items():
        boundaries = bundle["boundaries"].copy()

        dates = pd.DatetimeIndex(
            pd.to_datetime(boundaries.index)
        ).normalize()

        rejected_dates = reject_map.get(turbine, set())

        if rejected_dates and not boundaries.empty:
            original_accepted = boundaries[
                "accepted"
            ].fillna(False).to_numpy(dtype=bool)

            veto = np.array(
                [
                    bool(original_accepted[i])
                    and pd.Timestamp(d) in rejected_dates
                    for i, d in enumerate(dates)
                ],
                dtype=bool,
            )

            if veto.any():
                boundaries.loc[veto, "accepted"] = False
                boundaries.loc[
                    veto,
                    "reason",
                ] = "multiscale_sign_veto"

                if "shrinkage" in boundaries.columns:
                    boundaries.loc[veto, "shrinkage"] = 0.0

        bundle["boundaries"] = boundaries

        # V7.5 confidence/site-common-mode gates have already been applied.
        # Do not rerun them; only rebuild the state trajectory after the
        # conservative additional veto.
        out[turbine] = _rebuild_prior_trajectory(
            bundle,
            model_config,
            apply_confidence_gate=False,
        )

    return out


# Apply prediction state floor
def apply_prediction_state_floor(
    bundle: dict,
    min_jump_deg: float = 0.5,
    model_config: ModelConfig | None = None,
) -> dict:
    """
    Preserve detector decisions, but suppress accepted boundaries whose
    current state-level change is too small to justify a prediction state.

    `accepted` = detection decision
    `prediction_active` = prediction-state decision
    """
    if model_config is None:
        model_config = ModelConfig()

    # Always rebuild the current multiscale trajectory first.
    base = _rebuild_prior_trajectory(
        bundle,
        model_config,
        apply_confidence_gate=False,
    )

    obs = base["observables"].copy()
    boundaries = base["boundaries"].copy()

    boundaries["state_jump_deg"] = np.nan
    boundaries["prediction_active"] = False

    if obs.empty or boundaries.empty:
        base["boundaries"] = boundaries
        return base

    accepted = boundaries["accepted"].fillna(False).astype(bool)

    if not accepted.any():
        base["boundaries"] = boundaries
        return base

    # Detection clusters from the current accepted boundaries.
    detection_cluster = clusters_from_boundaries(
        obs.index,
        boundaries,
    )

    obs["detection_cluster"] = detection_cluster

    # Current multiscale state levels.
    state_levels = (
        obs.assign(_state=detection_cluster)
        .groupby("_state")["relative_prior_correction"]
        .median()
        .sort_index()
    )

    accepted_dates = pd.DatetimeIndex(
        pd.to_datetime(boundaries.index[accepted])
    ).sort_values()

    # Boundary k separates state k and k+1.
    for k, date in enumerate(accepted_dates):
        if (
            k not in state_levels.index
            or (k + 1) not in state_levels.index
        ):
            continue

        left = float(state_levels.loc[k])
        right = float(state_levels.loc[k + 1])

        if not np.isfinite(left) or not np.isfinite(right):
            continue

        jump = abs(
            float(wrap_180(right - left))
        )

        boundaries.at[date, "state_jump_deg"] = jump
        boundaries.at[date, "prediction_active"] = (
            jump >= min_jump_deg
        )

    # Build a temporary boundary table for prediction states.
    prediction_boundaries = boundaries.copy()
    prediction_boundaries["accepted"] = (
        prediction_boundaries["prediction_active"]
        .fillna(False)
        .astype(bool)
    )

    prediction_bundle = dict(base)
    prediction_bundle["boundaries"] = prediction_boundaries

    # Rebuild final prediction trajectory using only major states.
    prediction_bundle = _rebuild_prior_trajectory(
        prediction_bundle,
        model_config,
        apply_confidence_gate=False,
    )

    # Restore original detector boundaries and attach prediction metadata.
    prediction_bundle["boundaries"] = boundaries
    prediction_bundle["prediction_state_floor_deg"] = float(
        min_jump_deg
    )

    return prediction_bundle



# Run loto
def run_loto(
    train_ids,
    bundles,
    labels,
    theta_star,
    model_config,
):
    """Strict leave-one-turbine-out evaluation."""
    rows = []

    for holdout in train_ids:
        fit_ids = [tid for tid in train_ids if tid != holdout]

        C, beta = fit_global_beta(
            fit_ids,
            bundles,
            labels,
            theta_star,
            model_config,
        )

        pred = predict_from_bundle(
            holdout,
            bundles[holdout],
            C,
            beta,
            theta_star,
        )

        metrics = score_prediction(
            pred,
            labels[holdout],
            C - theta_star[holdout],
        )

        boundaries = bundles[holdout]["boundaries"]
        active_col = (
            "prediction_active"
            if "prediction_active" in boundaries.columns
            else "accepted"
        )
        accepted = (
            boundaries.index[boundaries[active_col].fillna(False)]
            if len(boundaries)
            else []
        )

        rows.append({
            "holdout": holdout,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "constant_mae": metrics["constant_mae"],
            "constant_rmse": metrics["constant_rmse"],
            "beta": beta,
            "states": metrics["n_states"],
            "boundaries": ", ".join(
                pd.Timestamp(d).date().isoformat()
                for d in accepted
            ) or "none",
        })

    table = pd.DataFrame(rows)

    macro = table[
        ["mae", "rmse", "constant_mae", "constant_rmse"]
    ].mean()

    return table, macro


def fit_and_predict(
    train_ids,
    target_ids,
    bundles,
    labels,
    theta_star,
    model_config,
):
    """Fit final calibration and predict target turbines."""
    C, beta = fit_global_beta(
        list(train_ids),
        bundles,
        labels,
        theta_star,
        model_config,
    )

    predictions = {
        tid: predict_from_bundle(
            tid,
            bundles[tid],
            C,
            beta,
            theta_star,
        )
        for tid in target_ids
    }

    rows = []

    for tid in target_ids:
        boundaries = bundles[tid]["boundaries"]

        detected = (
            boundaries.index[
                boundaries["accepted"].fillna(False)
            ]
            if len(boundaries)
            else []
        )

        if "prediction_active" in boundaries.columns:
            active = boundaries.index[
                boundaries["prediction_active"].fillna(False)
            ]
        else:
            active = detected

        pred = predictions[tid]
        prediction_levels = int(
            pred["prediction"].round(10).nunique()
        )

        rows.append({
            "turbine": tid,
            "detected_boundaries": ", ".join(
                pd.Timestamp(d).date().isoformat()
                for d in detected
            ) or "none",
            "prediction_boundaries": ", ".join(
                pd.Timestamp(d).date().isoformat()
                for d in active
            ) or "none",
            "states": int(pred["cluster"].nunique()),
            "prediction_levels": prediction_levels,
            "prediction_min": float(pred["prediction"].min()),
            "prediction_max": float(pred["prediction"].max()),
        })

    return C, beta, predictions, pd.DataFrame(rows)


# Comparison
def compare_model_stages(
    stages,
    train_ids,
    target_ids,
    labels,
    theta_star,
    model_config,
):
    """Compare several bundle stages with one compact API."""
    stage_rows = []
    target_rows = []
    predictions = {}

    for name, bundle_set in stages.items():
        loto, macro = run_loto(
            train_ids,
            bundle_set,
            labels,
            theta_star,
            model_config,
        )

        C, beta, pred, targets = fit_and_predict(
            train_ids,
            target_ids,
            bundle_set,
            labels,
            theta_star,
            model_config,
        )

        predictions[name] = pred

        stage_rows.append({
            "stage": name,
            "MAE": macro["mae"],
            "RMSE": macro["rmse"],
            "C": C,
            "beta": beta,
        })

        tmp = targets.copy()
        tmp.insert(0, "stage", name)
        target_rows.append(tmp)

    return (
        pd.DataFrame(stage_rows),
        pd.concat(target_rows, ignore_index=True),
        predictions,
    )


def compare_submission_structure(
    prediction,
    submission_path,
    turbine,
):
    """Structural comparison only; the submission is not treated as truth."""
    from sklearn.metrics import adjusted_rand_score

    ref = pd.read_csv(submission_path)
    ref["date"] = pd.to_datetime(ref["date"]).dt.normalize()

    ref = (
        ref.loc[ref["turbine_id"].eq(turbine)]
        .set_index("date")
        .sort_index()
        [["yaw_misalignment_deg", "cluster"]]
        .rename(columns={
            "yaw_misalignment_deg": "reference_prediction",
            "cluster": "reference_cluster",
        })
    )

    model = prediction[["prediction", "cluster"]].copy()
    model.index = pd.to_datetime(model.index).normalize()

    model = (
        model.groupby(model.index)
        .agg(
            model_prediction=("prediction", "median"),
            model_cluster=("cluster", "last"),
        )
    )

    joined = ref.join(model, how="inner").dropna()

    diff = (
        joined["model_prediction"]
        - joined["reference_prediction"]
    )

    def transition_dates(cluster):
        changed = cluster.ne(cluster.shift())
        return ", ".join(
            d.date().isoformat()
            for d in cluster.index[changed][1:]
        ) or "none"

    summary = pd.DataFrame([{
        "days": len(joined),
        "reference_states":
            joined["reference_cluster"].nunique(),
        "model_states":
            joined["model_cluster"].nunique(),
        "ARI": adjusted_rand_score(
            joined["reference_cluster"],
            joined["model_cluster"],
        ),
        "MAE_between_models": diff.abs().mean(),
        "RMSE_between_models":
            np.sqrt(np.mean(diff ** 2)),
        "bias_model_minus_reference": diff.mean(),
        "reference_transitions":
            transition_dates(joined["reference_cluster"]),
        "model_transitions":
            transition_dates(joined["model_cluster"]),
    }])

    return summary, joined
