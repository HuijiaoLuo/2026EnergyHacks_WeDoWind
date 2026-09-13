from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import baseline_openoa_power_vane_updated as b0
from baseline_loto_ridge import wrap_180
from fleet_context import FleetConfig, circular_median_deg
from yaw_relative_state import (
    RelativeStateConfig,
    clusters_from_boundaries,
    fleet_relative_series,
    label_free_boundaries,
)


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

    s = accepted.sort_index()["shrinkage"].to_numpy(dtype=float)

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

    state_offset: dict[int, float] = {}

    for position, state_id in enumerate(state_ids):
        values = signal.loc[
            clusters.eq(state_id)
        ].dropna()

        if values.empty:
            state_offset[int(state_id)] = 0.0
            continue

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


def _refresh_prior_trajectory(
    bundle: dict,
    config: ModelConfig,
    apply_confidence_gate: bool,
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
        if "shrinkage" not in boundaries.columns:
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
    return _refresh_prior_trajectory(
        bundle,
        model_config,
        apply_confidence_gate=False,
    )


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
        bundles[turbine] = _refresh_prior_trajectory(
            bundles[turbine],
            config,
            apply_confidence_gate=True,
        )

    return bundles


def _label_event_change(
    label: pd.Series,
    date: pd.Timestamp,
    window_days: int,
) -> float:
    daily = label.groupby(
        label.index.normalize()
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

    return float(
        np.median(post)
        - np.median(pre)
    )


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

        accepted = table[
            table["accepted"].fillna(False)
        ]

        for date, row in accepted.iterrows():
            shrinkage = float(
                row.get("shrinkage", 0.0)
            )
            relative_delta = (
                shrinkage
                * float(row["change"])
            )

            if (
                not np.isfinite(relative_delta)
                or abs(relative_delta) < 0.25
            ):
                continue

            yaw_delta = _label_event_change(
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

    base = C - theta_star[turbine]
    obs["constant_prior"] = base
    obs["prediction"] = (
        base
        - beta
        * obs["relative_prior_correction"].fillna(0.0)
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
