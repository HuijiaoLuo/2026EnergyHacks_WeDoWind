from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from baseline_loto_ridge import wrap_180
from fleet_context import circular_mean_deg


@dataclass(frozen=True)
class SelfResponseConfig:
    ws_bin_width: float = 0.5
    sector_width_deg: float = 30.0

    # Automatic partial-load / identifiability mask.
    min_field_confidence: float = 0.20
    min_refs: int = 2
    power_quantile_cap: float = 0.90
    pitch_quantile_cap: float = 0.80
    slope_positive_quantile: float = 0.35

    # Matched +/- vane response.
    max_abs_delta_deg: float = 8.0
    delta_bin_width_deg: float = 1.0
    min_abs_delta_deg: float = 1.0
    min_side_samples: int = 8
    min_total_pairs: int = 3


def identify_partial_load(
    target_bins: pd.DataFrame,
    local_ref: pd.DataFrame,
    config: SelfResponseConfig,
) -> pd.DataFrame:
    """
    Learn the yaw-identifiable operating regime from the turbine itself.
    No absolute Power/Pitch threshold is hard-coded.
    """
    x = target_bins.join(local_ref, how="inner")
    x["identifiable"] = False

    required = {"ws_ref", "n_refs", "field_confidence"}
    if x.empty or not required.issubset(x.columns):
        return x.iloc[0:0].copy()

    x = x[
        x["ws_ref"].notna()
        & (x["n_refs"] >= config.min_refs)
        & (x["field_confidence"] >= config.min_field_confidence)
        & (x["Power"] > 0)
    ].copy()
    if x.empty:
        return x

    x["ws_bin"] = (
        x["ws_ref"] / config.ws_bin_width
    ).round() * config.ws_bin_width

    curve = x.groupby("ws_bin").agg(
        power_med=("Power", "median"),
        pitch_med=("PitchAngle", "median"),
        n=("Power", "size"),
    ).sort_index()

    if len(curve) < 3:
        return x

    curve["dP_dU"] = np.gradient(
        curve["power_med"].to_numpy(dtype=float),
        curve.index.to_numpy(dtype=float),
    )
    positive = curve.loc[curve["dP_dU"] > 0, "dP_dU"]
    slope_cut = (
        float(positive.quantile(config.slope_positive_quantile))
        if len(positive) else np.inf
    )
    usable_ws_bins = curve.index[curve["dP_dU"] >= slope_cut]

    power_cap = float(x["Power"].quantile(config.power_quantile_cap))
    pitch_cap = float(x["PitchAngle"].quantile(config.pitch_quantile_cap))

    x["identifiable"] = (
        x["ws_bin"].isin(usable_ws_bins)
        & (x["Power"] <= power_cap)
        & (x["PitchAngle"] <= pitch_cap)
    )
    return x


def build_power_residual(
    operating: pd.DataFrame,
    config: SelfResponseConfig,
) -> pd.DataFrame:
    if operating.empty or "identifiable" not in operating.columns:
        return operating.iloc[0:0].copy()

    x = operating[operating["identifiable"].fillna(False)].copy()
    if x.empty:
        return x

    x["log_power"] = np.log(x["Power"].clip(lower=1.0))
    x["sector"] = np.floor(
        (x["wdir_ref"] % 360.0) / config.sector_width_deg
    ).astype(int)

    group = x.groupby(["ws_bin", "sector"])["log_power"]
    count = group.transform("size")
    sector_base = group.transform("median")
    ws_base = x.groupby("ws_bin")["log_power"].transform("median")
    x["power_resid"] = x["log_power"] - sector_base.where(count >= 8, ws_base)
    return x


def prepare_self_response_data(
    target_bins: pd.DataFrame,
    local_ref: pd.DataFrame,
    config: SelfResponseConfig,
) -> pd.DataFrame:
    """
    Expensive operating-regime and power-residual preparation.
    This is independent of the self-response window and should be cached once.
    """
    return build_power_residual(
        identify_partial_load(target_bins, local_ref, config),
        config,
    )


def matched_window_estimate(
    window: pd.DataFrame,
    config: SelfResponseConfig,
) -> dict:
    """
    Vectorized within-window +/-delta matching.

    No loop over delta bins: groupby/unstack computes all positive/negative
    matched bins in one operation.
    """
    w = window.dropna(subset=["vane", "power_resid"]).copy()
    if len(w) < 2 * config.min_side_samples:
        return {
            "self_observable_raw": np.nan,
            "self_confidence": 0.0,
            "matched_pairs": 0,
        }

    vane0 = circular_mean_deg(w["vane"])
    w["delta"] = wrap_180(w["vane"] - vane0)
    w = w[
        w["delta"].abs().between(
            config.min_abs_delta_deg,
            config.max_abs_delta_deg,
        )
    ].copy()
    if w.empty:
        return {
            "self_observable_raw": np.nan,
            "self_confidence": 0.0,
            "matched_pairs": 0,
        }

    w["abs_bin"] = (
        w["delta"].abs() / config.delta_bin_width_deg
    ).round() * config.delta_bin_width_deg
    w["side"] = np.where(w["delta"] > 0, 1, -1)

    agg = (
        w.groupby(["abs_bin", "side"])["power_resid"]
        .agg(["median", "count"])
        .unstack("side")
    )

    if agg.empty or ("median", 1) not in agg or ("median", -1) not in agg:
        return {
            "self_observable_raw": np.nan,
            "self_confidence": 0.0,
            "matched_pairs": 0,
        }

    pos_med = agg[("median", 1)]
    neg_med = agg[("median", -1)]
    pos_n = agg[("count", 1)].fillna(0)
    neg_n = agg[("count", -1)].fillna(0)

    good = (
        pos_med.notna()
        & neg_med.notna()
        & (pos_n >= config.min_side_samples)
        & (neg_n >= config.min_side_samples)
    )
    if good.sum() < config.min_total_pairs:
        return {
            "self_observable_raw": np.nan,
            "self_confidence": float(good.sum() / max(config.min_total_pairs, 1)),
            "matched_pairs": int(good.sum()),
        }

    dmag = agg.index.to_numpy(dtype=float)[good.to_numpy()]
    diff = (pos_med - neg_med).to_numpy(dtype=float)[good.to_numpy()]
    # Raw matched odd response.  Do not turn this into degrees here: the old
    # small-angle division by (pi / 180)^2 was numerically unstable.
    asymmetry = diff / dmag
    pair_weight = np.minimum(
        pos_n.to_numpy(dtype=float)[good.to_numpy()],
        neg_n.to_numpy(dtype=float)[good.to_numpy()],
    )

    observable = float(np.average(asymmetry, weights=pair_weight))
    dispersion = float(
        np.median(np.abs(asymmetry - np.median(asymmetry)))
    )
    dispersion_scale = max(abs(observable), 0.01)
    confidence = float(
        min(1.0, len(asymmetry) / 6.0)
        * np.exp(-dispersion / dispersion_scale)
    )

    return {
        "self_observable_raw": observable,
        "self_confidence": confidence,
        "matched_pairs": int(len(asymmetry)),
        "self_dispersion": dispersion,
    }


def self_response_series_prepared(
    prepared: pd.DataFrame,
    window_days: int,
    config: SelfResponseConfig,
) -> pd.DataFrame:
    """
    Only the rolling/window part varies with window_days.

    A single loop over ~100 weekly window centres remains; there is no nested
    timestamp x neighbour loop and no inner delta-bin loop.
    """
    x = prepared
    if x.empty:
        return pd.DataFrame(
            columns=["self_observable_raw", "self_confidence", "matched_pairs"]
        )

    start = x.index.min().normalize() + pd.Timedelta(days=window_days // 2)
    end = x.index.max().normalize() - pd.Timedelta(days=window_days // 2)
    dates = pd.date_range(start, end, freq="7D")
    half = pd.Timedelta(days=window_days / 2)

    rows = [
        {"date": d, **matched_window_estimate(x.loc[d - half : d + half], config)}
        for d in dates
    ]
    if not rows:
        return pd.DataFrame(
            columns=["self_observable_raw", "self_confidence", "matched_pairs"]
        )

    sparse = pd.DataFrame(rows).set_index("date")
    full = pd.date_range(
        x.index.min().normalize(),
        x.index.max().normalize(),
        freq="D",
    )

    out = sparse.reindex(full)
    # Keep coverage measured on the actual weekly estimates.  Interpolation is
    # only for constructing a smooth state descriptor and must not make the
    # diagnostic look as though every day had an identifiable self response.
    raw_coverage = float(sparse["self_observable_raw"].notna().mean())
    out["self_observable_coverage"] = raw_coverage
    out["self_observable_raw"] = (
        out["self_observable_raw"]
        .interpolate(limit_direction="both")
        .rolling(7, center=True, min_periods=3)
        .median()
    )
    out["self_confidence"] = (
        out["self_confidence"]
        .interpolate(limit_direction="both")
        .rolling(7, center=True, min_periods=3)
        .median()
        .fillna(0.0)
    )
    return out


def self_response_series(
    target_bins: pd.DataFrame,
    local_ref: pd.DataFrame,
    window_days: int,
    config: SelfResponseConfig,
) -> pd.DataFrame:
    return self_response_series_prepared(
        prepare_self_response_data(target_bins, local_ref, config),
        window_days,
        config,
    )
