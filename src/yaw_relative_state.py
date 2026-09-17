from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from baseline_loto_ridge import wrap_180
from fleet_context import FleetConfig, candidate_neighbours, circular_median_deg


@dataclass(frozen=True)
class RelativeStateConfig:
    """Fixed settings for label-free relative-heading state detection."""

    sector_width_deg: float = 30.0
    max_pair_residual_deg: float = 45.0
    min_pair_bins: int = 50
    min_pair_coverage: float = 0.05
    pair_noise_scale_deg: float = 8.0
    min_pair_weight: float = 0.01
    smooth_days: int = 7

    change_window_days: int = 21
    min_stable_days: int = 12
    min_state_days: int = 21
    candidate_stride_days: int = 7
    z_threshold: float = 3.0
    min_degree_change: float = 1.0
    noise_floor_deg: float = 0.5
    sensor_jump_deg: float = 15.0
    sensor_exclusion_days: int = 35
    min_pair_agreement: float = 0.60
    common_mode_min_deg: float = 0.75
    common_mode_ratio: float = 0.60

    # Evidence-to-prior shrinkage.  z=z0 gives only half of the z contribution
    # before agreement/background/support are applied.
    shrinkage_z0: float = 4.0

    # Conservative state-level L2 supplement. It is used only for long,
    # two-sided regimes; short excursions are retained as diagnostics but
    # are not allowed to create yaw states.
    pelt_penalty: float = 400.0
    pelt_min_days: int = 14
    pelt_context_days: int = 28
    pelt_min_change_deg: float = 2.0
    pelt_merge_days: int = 14

    # Optional fixed historical wind-field prior for neighbour reliability.
    # It is computed once per target-neighbour pair; it is not a time-varying
    # gate and does not alter the label-free boundary logic.
    use_overlap_prior: bool = False
    overlap_min_days: int = 60
    overlap_floor: float = 0.50
    use_dynamic_overlap_prior: bool = False
    dynamic_overlap_window_days: int = 56
    dynamic_overlap_smooth_days: int = 14
    dynamic_overlap_min_days: int = 30


def _aligned_matrix(
    binned: dict[str, pd.DataFrame],
    ids: list[str],
    column: str,
    index: pd.DatetimeIndex,
) -> np.ndarray:
    if not ids:
        return np.empty((len(index), 0), dtype=float)
    return pd.concat(
        {tid: binned[tid][column] for tid in ids}, axis=1
    ).reindex(index).to_numpy(dtype=float)


def _weighted_row_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Vectorized weighted median across pairs with missing-safe weights."""
    if values.shape[1] == 0:
        return np.full(values.shape[0], np.nan)

    if np.ndim(weights) == 1:
        weight_matrix = np.broadcast_to(weights, values.shape)
    else:
        weight_matrix = np.asarray(weights, dtype=float)
        if weight_matrix.shape != values.shape:
            raise ValueError("Dynamic weight matrix must match values shape")

    valid = np.isfinite(values) & np.isfinite(weight_matrix) & (weight_matrix > 0)
    safe = np.where(valid, values, np.inf)
    order = np.argsort(safe, axis=1)
    sorted_values = np.take_along_axis(safe, order, axis=1)

    sorted_weights = np.take_along_axis(
        np.where(valid, weight_matrix, 0.0), order, axis=1
    )

    total = sorted_weights.sum(axis=1)
    cumulative = np.cumsum(sorted_weights, axis=1)
    position = (cumulative >= 0.5 * total[:, None]).argmax(axis=1)

    out = sorted_values[np.arange(len(values)), position]
    out[total <= 0] = np.nan
    return out


def _sector_residual(
    pair_angle: np.ndarray,
    pair_wind: np.ndarray,
    config: RelativeStateConfig,
) -> np.ndarray:
    """Return e_pair = wrap(pair_angle - pair sector baseline)."""
    active = np.isfinite(pair_angle) & np.isfinite(pair_wind)
    residual = np.full(pair_angle.shape, np.nan)

    row_id, pair_id = np.where(active)
    if len(row_id) == 0:
        return residual

    sector = np.floor(
        (pair_wind[row_id, pair_id] % 360.0) / config.sector_width_deg
    ).astype(int)

    long = pd.DataFrame({
        "pair": pair_id,
        "sector": sector,
        "angle": pair_angle[row_id, pair_id],
    })
    baseline = long.groupby(["pair", "sector"])["angle"].agg(
        circular_median_deg
    )

    lookup = pd.MultiIndex.from_arrays(
        [pair_id, sector], names=["pair", "sector"]
    )
    residual[row_id, pair_id] = wrap_180(
        pair_angle[row_id, pair_id]
        - baseline.reindex(lookup).to_numpy()
    )

    residual[np.abs(residual) > config.max_pair_residual_deg] = np.nan
    return residual


def _pair_statistics(
    residual: np.ndarray,
    distance_weight: np.ndarray,
    denominator: int,
    config: RelativeStateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = np.isfinite(residual).sum(axis=0)
    coverage = count / max(denominator, 1)

    mad = np.full(residual.shape[1], np.nan)
    good = count > 0
    if good.any():
        mad[good] = 1.4826 * np.nanmedian(
            np.abs(residual[:, good]), axis=0
        )

    reliability = (
        distance_weight
        * coverage
        * np.exp(-np.square(mad / config.pair_noise_scale_deg))
    )
    usable = (
        (count >= config.min_pair_bins)
        & (coverage >= config.min_pair_coverage)
        & np.isfinite(mad)
        & (reliability >= config.min_pair_weight)
    )
    return coverage, mad, np.where(usable, reliability, 0.0), usable


def _spectral_overlap(a: pd.Series, b: pd.Series, min_points: int) -> float:
    """Similarity of normalized daily fluctuation spectra in [0, 1]."""
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < min_points:
        return np.nan
    x = aligned.iloc[:, 0].to_numpy(dtype=float)
    y = aligned.iloc[:, 1].to_numpy(dtype=float)
    x = x - np.mean(x)
    y = y - np.mean(y)
    window = np.hanning(len(aligned))
    px = np.abs(np.fft.rfft(x * window))[1:] ** 2
    py = np.abs(np.fft.rfft(y * window))[1:] ** 2
    sx, sy = px.sum(), py.sum()
    if sx <= 0 or sy <= 0:
        return np.nan
    return float(np.minimum(px / sx, py / sy).sum())


def _overlap_prior_weights(
    turbine: str,
    ids: list[str],
    binned: dict[str, pd.DataFrame],
    config: RelativeStateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed speed, direction and combined overlap priors per pair."""
    speed = np.full(len(ids), np.nan, dtype=float)
    direction = np.full(len(ids), np.nan, dtype=float)
    target = binned[turbine]
    target_speed = target["WindSpeed"].resample("D").median()
    target_direction = target["WindDir"].resample("D").median()
    target_sin = np.sin(np.deg2rad(target_direction))
    target_cos = np.cos(np.deg2rad(target_direction))

    for j, neighbour in enumerate(ids):
        other = binned[neighbour]
        speed_other = other["WindSpeed"].resample("D").median()
        direction_other = other["WindDir"].resample("D").median()
        direction_sin = np.sin(np.deg2rad(direction_other))
        direction_cos = np.cos(np.deg2rad(direction_other))
        speed[j] = _spectral_overlap(
            target_speed, speed_other, config.overlap_min_days
        )
        values = np.asarray([
            _spectral_overlap(target_sin, direction_sin, config.overlap_min_days),
            _spectral_overlap(target_cos, direction_cos, config.overlap_min_days),
        ], dtype=float)
        valid = np.isfinite(values)
        direction[j] = float(values[valid].mean()) if valid.any() else np.nan

    combined = np.zeros(len(ids), dtype=float)
    for j in range(len(ids)):
        values = np.asarray([speed[j], direction[j]], dtype=float)
        valid = np.isfinite(values)
        combined[j] = float(values[valid].mean()) if valid.any() else 0.0
    return speed, direction, np.clip(combined, 0.0, 1.0)


def _pair_dynamic_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    full_days: pd.DatetimeIndex,
    config: RelativeStateConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Estimate a smooth daily wind-field overlap for one pair.

    The window is deliberately long relative to a daily state boundary.  This
    is a reliability prior, not a boundary signal.  WindDir is compared by
    cosine of its wrapped circular difference.
    """
    ls = left["WindSpeed"].resample("D").median().reindex(full_days)
    rs = right["WindSpeed"].resample("D").median().reindex(full_days)
    ld = left["WindDir"].resample("D").median().reindex(full_days)
    rd = right["WindDir"].resample("D").median().reindex(full_days)

    valid = ls.notna() & rs.notna() & ld.notna() & rd.notna()
    count = valid.astype(float).rolling(
        config.dynamic_overlap_window_days,
        min_periods=config.dynamic_overlap_min_days,
        center=True,
    ).sum()
    coverage = count / max(config.dynamic_overlap_window_days, 1)

    speed_corr = ls.rolling(
        config.dynamic_overlap_window_days,
        min_periods=config.dynamic_overlap_min_days,
        center=True,
    ).corr(rs)
    speed_similarity = speed_corr.clip(lower=0.0, upper=1.0)

    direction_cos = np.cos(
        np.deg2rad(wrap_180(ld.to_numpy(dtype=float) - rd.to_numpy(dtype=float)))
    )
    direction_similarity = pd.Series(
        direction_cos, index=full_days
    ).rolling(
        config.dynamic_overlap_window_days,
        min_periods=config.dynamic_overlap_min_days,
        center=True,
    ).mean().add(1.0).div(2.0).clip(0.0, 1.0)

    combined = pd.concat(
        [speed_similarity, direction_similarity], axis=1
    ).mean(axis=1, skipna=True)
    combined = (combined * coverage).clip(0.0, 1.0)
    combined = combined.rolling(
        config.dynamic_overlap_smooth_days,
        min_periods=1,
        center=True,
    ).median()
    return speed_similarity, direction_similarity, combined


def _daily_pair_matrix(
    residual: np.ndarray,
    index: pd.DatetimeIndex,
    columns: list,
    full_days: pd.DatetimeIndex,
) -> pd.DataFrame:
    if residual.shape[1] == 0:
        return pd.DataFrame(index=full_days, columns=columns, dtype=float)

    frame = pd.DataFrame(residual, index=index, columns=columns)
    return frame.groupby(frame.index.normalize()).agg(
        circular_median_deg
    ).reindex(full_days)


def fleet_relative_series(
    turbine: str,
    binned: dict[str, pd.DataFrame],
    layout: pd.DataFrame,
    fleet_config: FleetConfig,
    rel_config: RelativeStateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build target-pair residuals and a neighbour-only background residual."""
    candidates = candidate_neighbours(turbine, layout, fleet_config)

    if candidates.empty or turbine not in binned:
        diagnostics = candidates.assign(
            coverage=np.nan,
            pair_mad=np.nan,
            reliability_weight=0.0,
            usable=False,
        )
        return pd.DataFrame(), diagnostics, pd.DataFrame()

    ids = candidates.index.tolist()
    idx = binned[turbine].index.sort_values()
    full_days = pd.date_range(
        idx.min().normalize(), idx.max().normalize(), freq="D"
    )

    target_heading = (
        binned[turbine]["NacDir"]
        .reindex(idx)
        .to_numpy(dtype=float)
    )
    neighbour_heading = _aligned_matrix(
        binned, ids, "NacDir", idx
    )
    neighbour_wind = _aligned_matrix(
        binned, ids, "WindDir", idx
    )

    d_target = wrap_180(
        target_heading[:, None] - neighbour_heading
    )
    e_target = _sector_residual(
        d_target, neighbour_wind, rel_config
    )

    coverage, pair_mad, reliability, usable = _pair_statistics(
        e_target,
        candidates["distance_weight"].to_numpy(dtype=float),
        int(np.isfinite(target_heading).sum()),
        rel_config,
    )

    overlap_speed = np.full(len(ids), np.nan, dtype=float)
    overlap_direction = np.full(len(ids), np.nan, dtype=float)
    overlap_combined = np.ones(len(ids), dtype=float)
    if rel_config.use_overlap_prior:
        overlap_speed, overlap_direction, overlap_combined = _overlap_prior_weights(
            turbine, ids, binned, rel_config
        )
        overlap_factor = rel_config.overlap_floor + (
            1.0 - rel_config.overlap_floor
        ) * overlap_combined
        reliability = reliability * overlap_factor
        usable &= reliability >= rel_config.min_pair_weight

    e_target[:, ~usable] = np.nan
    pair_daily = _daily_pair_matrix(
        e_target, idx, ids, full_days
    )
    target_dynamic_factor = np.ones((len(full_days), len(ids)), dtype=float)
    if rel_config.use_dynamic_overlap_prior:
        dynamic_rows = []
        for neighbour in ids:
            _, _, overlap = _pair_dynamic_overlap(
                binned[turbine], binned[neighbour], full_days, rel_config
            )
            dynamic_rows.append(overlap.to_numpy(dtype=float))
        dynamic_overlap = np.column_stack(dynamic_rows)
        dynamic_overlap[~np.isfinite(dynamic_overlap)] = 0.0
        target_dynamic_factor = rel_config.overlap_floor + (
            1.0 - rel_config.overlap_floor
        ) * dynamic_overlap
        target_dynamic_factor[:, ~usable] = 0.0
    else:
        target_dynamic_factor[:, ~usable] = 0.0
    relative_daily = _weighted_row_median(
        pair_daily.to_numpy(dtype=float),
        reliability[None, :] * target_dynamic_factor,
    )

    diagnostics = candidates.copy()
    diagnostics["coverage"] = coverage
    diagnostics["pair_mad"] = pair_mad
    diagnostics["reliability_weight"] = reliability
    diagnostics["speed_overlap"] = overlap_speed
    diagnostics["direction_overlap"] = overlap_direction
    diagnostics["overlap_prior"] = overlap_combined
    dynamic_median = np.zeros(len(ids), dtype=float)
    for j in range(len(ids)):
        values = target_dynamic_factor[:, j]
        values = values[np.isfinite(values) & (values > 0)]
        if len(values):
            dynamic_median[j] = float(np.median(values))
    diagnostics["dynamic_overlap_median"] = dynamic_median
    diagnostics["usable"] = usable

    background_daily = np.full(len(full_days), np.nan)

    usable_id = np.flatnonzero(usable)
    if len(usable_id) >= 2:
        ia0, ib0 = np.triu_indices(len(usable_id), k=1)
        ia, ib = usable_id[ia0], usable_id[ib0]

        d_background = wrap_180(
            neighbour_heading[:, ia] - neighbour_heading[:, ib]
        )
        e_background = _sector_residual(
            d_background,
            neighbour_wind[:, ia],
            rel_config,
        )

        base_weight = np.sqrt(
            reliability[ia] * reliability[ib]
        )
        _, bg_mad, bg_weight, bg_usable = _pair_statistics(
            e_background,
            base_weight,
            len(idx),
            rel_config,
        )
        bg_weight = np.where(
            np.isfinite(bg_mad) & bg_usable,
            bg_weight,
            0.0,
        )

        background_pairs = _daily_pair_matrix(
            e_background,
            idx,
            list(range(len(ia))),
            full_days,
        )
        background_dynamic_factor = np.ones(
            (len(full_days), len(ia)), dtype=float
        )
        if rel_config.use_dynamic_overlap_prior:
            bg_rows = []
            for left_idx, right_idx in zip(ia, ib):
                _, _, overlap = _pair_dynamic_overlap(
                    binned[ids[left_idx]],
                    binned[ids[right_idx]],
                    full_days,
                    rel_config,
                )
                bg_rows.append(overlap.to_numpy(dtype=float))
            bg_overlap = np.column_stack(bg_rows)
            bg_overlap[~np.isfinite(bg_overlap)] = 0.0
            background_dynamic_factor = rel_config.overlap_floor + (
                1.0 - rel_config.overlap_floor
            ) * bg_overlap
            background_dynamic_factor[:, ~bg_usable] = 0.0
        else:
            background_dynamic_factor[:, ~bg_usable] = 0.0
        background_daily = _weighted_row_median(
            background_pairs.to_numpy(dtype=float),
            bg_weight[None, :] * background_dynamic_factor,
        )

    daily = pd.DataFrame({
        "relative_heading": relative_daily,
        "background_residual": background_daily,
    }, index=full_days)

    daily["relative_heading_smooth"] = (
        daily["relative_heading"]
        .rolling(
            rel_config.smooth_days,
            center=True,
            min_periods=3,
        )
        .apply(circular_median_deg, raw=False)
    )
    daily["background_residual_smooth"] = (
        daily["background_residual"]
        .rolling(
            rel_config.smooth_days,
            center=True,
            min_periods=3,
        )
        .apply(circular_median_deg, raw=False)
    )

    return daily, diagnostics, pair_daily


def _persistent_shift(
    signal: pd.Series,
    date: pd.Timestamp,
    config: RelativeStateConfig,
) -> tuple[float, float, float]:
    window = config.change_window_days

    pre = signal.reindex(
        pd.date_range(
            date - pd.Timedelta(days=window),
            date - pd.Timedelta(days=1),
        )
    ).dropna()
    post = signal.reindex(
        pd.date_range(
            date,
            date + pd.Timedelta(days=window - 1),
        )
    ).dropna()

    if (
        len(pre) < config.min_stable_days
        or len(post) < config.min_stable_days
    ):
        return np.nan, np.nan, 0.0

    pre_level = circular_median_deg(pre)
    post_level = circular_median_deg(post)
    change = float(wrap_180(post_level - pre_level))

    scale = np.asarray([
        1.4826 * np.median(
            np.abs(wrap_180(pre.to_numpy() - pre_level))
        ),
        1.4826 * np.median(
            np.abs(wrap_180(post.to_numpy() - post_level))
        ),
    ])
    scale = scale[np.isfinite(scale)]
    noise = (
        float(np.median(scale))
        if len(scale)
        else np.nan
    )
    support = (len(pre) + len(post)) / (2.0 * window)
    return change, noise, float(support)


def _pair_step_agreement(
    pair_daily: pd.DataFrame,
    pair_weight: np.ndarray,
    date: pd.Timestamp,
    target_change: float,
    config: RelativeStateConfig,
) -> float:
    window = config.change_window_days

    pre = pair_daily.reindex(
        pd.date_range(
            date - pd.Timedelta(days=window),
            date - pd.Timedelta(days=1),
        )
    ).to_numpy(dtype=float)

    post = pair_daily.reindex(
        pd.date_range(
            date,
            date + pd.Timedelta(days=window - 1),
        )
    ).to_numpy(dtype=float)

    enough = (
        (
            np.isfinite(pre).sum(axis=0)
            >= config.min_stable_days // 2
        )
        & (
            np.isfinite(post).sum(axis=0)
            >= config.min_stable_days // 2
        )
        & (pair_weight > 0)
    )

    if not enough.any() or not np.isfinite(target_change):
        return 0.0

    pre_level = np.full(pre.shape[1], np.nan)
    post_level = np.full(post.shape[1], np.nan)

    pre_level[enough] = np.nanmedian(
        pre[:, enough], axis=0
    )
    post_level[enough] = np.nanmedian(
        post[:, enough], axis=0
    )

    pair_change = wrap_180(
        post_level - pre_level
    )
    tolerance = max(
        1.5,
        min(4.0, 0.5 * abs(target_change)),
    )
    agrees = (
        np.abs(
            wrap_180(
                pair_change - target_change
            )
        )
        <= tolerance
    )

    weight = np.where(enough, pair_weight, 0.0)
    denominator = np.sum(weight)
    if denominator <= 0:
        return 0.0

    return float(
        np.sum(weight * agrees) / denominator
    )


def _event_confidence(
    z: float,
    support: float,
    agreement: float,
    change: float,
    background_change: float,
    config: RelativeStateConfig,
) -> float:
    """
    Convert dynamic evidence into a shrinkage factor toward the constant prior.

    Weak events are allowed to exist diagnostically while contributing almost
    no yaw correction.  Strong, pair-consistent, target-specific events can
    approach weight 1.
    """
    if (
        not np.isfinite(z)
        or not np.isfinite(change)
        or z <= 0
    ):
        return 0.0

    z2 = z * z
    z_factor = z2 / (
        z2 + config.shrinkage_z0 ** 2
    )

    agreement_factor = np.clip(
        (agreement - 0.50) / 0.50,
        0.0,
        1.0,
    )

    if np.isfinite(background_change):
        background_ratio = abs(background_change) / max(
            abs(change),
            config.noise_floor_deg,
        )
        background_factor = np.clip(
            1.0 - background_ratio,
            0.0,
            1.0,
        )
    else:
        background_factor = 1.0

    support_factor = np.clip(
        support,
        0.0,
        1.0,
    )

    return float(
        z_factor
        * agreement_factor
        * background_factor
        * support_factor
    )



def _l2_state_segments(
    signal: pd.Series,
    penalty: float,
    min_days: int,
) -> pd.DataFrame:
    """
    Exact 1-D L2 dynamic-programming segmentation on a daily series.

    n is only ~731, so O(n^2) is inexpensive and avoids another dependency.
    Missing days remain in calendar time but do not contribute to SSE.
    """
    x = signal.copy().sort_index()
    x.index = pd.to_datetime(x.index).normalize()
    x = x.groupby(x.index).median()

    if x.empty:
        return pd.DataFrame(
            columns=["start", "end", "days", "n_valid", "level", "noise"]
        )

    days = pd.date_range(x.index.min(), x.index.max(), freq="D")
    y = x.reindex(days).to_numpy(dtype=float)
    valid = np.isfinite(y)
    yy = np.where(valid, y, 0.0)

    n = len(y)
    count = np.concatenate([[0], np.cumsum(valid.astype(int))])
    csum = np.concatenate([[0.0], np.cumsum(yy)])
    csq = np.concatenate([[0.0], np.cumsum(yy * yy)])

    dp = np.full(n + 1, np.inf)
    prev = np.full(n + 1, -1, dtype=int)
    dp[0] = -float(penalty)

    min_valid = max(3, min_days // 2)

    for end in range(min_days, n + 1):
        starts = np.arange(0, end - min_days + 1)
        nv = count[end] - count[starts]
        ok = (nv >= min_valid) & np.isfinite(dp[starts])
        if not ok.any():
            continue

        s = starts[ok]
        nvf = nv[ok].astype(float)
        seg_sum = csum[end] - csum[s]
        seg_sq = csq[end] - csq[s]
        sse = seg_sq - seg_sum * seg_sum / nvf
        objective = dp[s] + sse + penalty

        j = int(np.argmin(objective))
        dp[end] = objective[j]
        prev[end] = s[j]

    if prev[n] < 0:
        return pd.DataFrame(
            columns=["start", "end", "days", "n_valid", "level", "noise"]
        )

    cuts = [n]
    end = n
    while end > 0:
        start = prev[end]
        if start < 0:
            break
        cuts.append(start)
        end = start

    cuts = sorted(set(cuts))
    rows = []

    for start, end in zip(cuts[:-1], cuts[1:]):
        seg = y[start:end]
        good = seg[np.isfinite(seg)]
        if len(good):
            level = float(np.median(good))
            noise = float(
                1.4826
                * np.median(
                    np.abs(wrap_180(good - level))
                )
            )
        else:
            level = np.nan
            noise = np.nan

        rows.append({
            "start": days[start],
            "end": days[end - 1],
            "days": int(end - start),
            "n_valid": int(len(good)),
            "level": level,
            "noise": noise,
        })

    return pd.DataFrame(rows)


def _stable_pelt_candidates(
    daily: pd.DataFrame,
    pair_daily: pd.DataFrame,
    pair_weight: np.ndarray,
    config: RelativeStateConfig,
) -> pd.DataFrame:
    """
    Conservative state-level supplement.

    Keep only boundaries with long states on BOTH sides.  This automatically
    rejects the short +/- excursion pairs seen in WTG13 (May and early Nov)
    and leaves the long July regime change as a candidate.
    """
    target = daily["relative_heading_smooth"].dropna().sort_index()

    columns = [
        "change",
        "background_change",
        "noise",
        "z",
        "support",
        "pair_agreement",
        "event_confidence",
        "common_mode",
        "sensor_like",
        "reason",
        "accepted",
        "source",
    ]

    if target.empty:
        return pd.DataFrame(columns=columns)

    seg = _l2_state_segments(
        target,
        penalty=config.pelt_penalty,
        min_days=config.pelt_min_days,
    )

    if len(seg) < 2:
        return pd.DataFrame(columns=columns)

    rows = []

    for k in range(1, len(seg)):
        left = seg.iloc[k - 1]
        right = seg.iloc[k]
        date = pd.Timestamp(right["start"])

        if not (
            np.isfinite(left["level"])
            and np.isfinite(right["level"])
        ):
            continue

        change = float(
            wrap_180(
                float(right["level"]) - float(left["level"])
            )
        )

        pooled_noise = np.asarray(
            [left["noise"], right["noise"]],
            dtype=float,
        )
        pooled_noise = pooled_noise[np.isfinite(pooled_noise)]
        noise = (
            float(np.median(pooled_noise))
            if len(pooled_noise)
            else np.nan
        )
        z = (
            abs(change) / max(noise, config.noise_floor_deg)
            if np.isfinite(noise)
            else 0.0
        )

        stable_two_sided = (
            int(left["days"]) >= config.pelt_context_days
            and int(right["days"]) >= config.pelt_context_days
        )
        magnitude_ok = (
            abs(change) >= config.pelt_min_change_deg
        )

        background_change, _, _ = _persistent_shift(
            daily["background_residual_smooth"],
            date,
            config,
        )

        agreement = _pair_step_agreement(
            pair_daily,
            pair_weight,
            date,
            change,
            config,
        )

        common_mode = bool(
            np.isfinite(background_change)
            and abs(background_change)
            >= max(
                config.common_mode_min_deg,
                config.common_mode_ratio * abs(change),
            )
        )

        sensor_like = bool(
            abs(change) > config.sensor_jump_deg
        )

        support = min(
            1.0,
            min(
                float(left["n_valid"]),
                float(right["n_valid"]),
            )
            / max(float(config.pelt_context_days), 1.0),
        )

        base_conf = _event_confidence(
            z,
            support,
            agreement,
            change,
            background_change,
            config,
        )

        if not stable_two_sided:
            reason = "pelt_short_excursion"
            accepted = False
            confidence = 0.0
        elif sensor_like:
            reason = "sensor"
            accepted = False
            confidence = 0.0
        elif common_mode:
            reason = "common_mode"
            accepted = False
            confidence = 0.0
        elif not magnitude_ok:
            reason = "pelt_small"
            accepted = False
            confidence = 0.0
        elif agreement < config.min_pair_agreement:
            reason = "pair_disagreement"
            accepted = False
            confidence = 0.0
        else:
            reason = "pelt_yaw"
            accepted = True

            # A long two-sided PELT regime is itself strong structural evidence.
            # Keep the existing evidence score but ensure it can pass the
            # constant-prior gate; shrinkage still remains modest.
            structural = (
                0.25
                + 0.35
                * min(1.0, abs(change) / 6.0)
                * min(1.0, agreement)
                * support
            )
            confidence = float(
                np.clip(max(base_conf, structural), 0.0, 0.75)
            )

        rows.append({
            "date": date,
            "change": change,
            "background_change": background_change,
            "noise": noise,
            "z": z,
            "support": support,
            "pair_agreement": agreement,
            "event_confidence": confidence,
            "common_mode": common_mode,
            "sensor_like": sensor_like,
            "reason": reason,
            "accepted": accepted,
            "source": "pelt",
        })

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).set_index("date").sort_index()


def _merge_boundary_tables(
    rolling: pd.DataFrame,
    pelt: pd.DataFrame,
    config: RelativeStateConfig,
) -> pd.DataFrame:
    """
    Union of the original V7.3 rolling detector and conservative stable PELT.

    If two accepted yaw candidates are within pelt_merge_days, keep only the
    stronger one.  Non-accepted diagnostics are retained.
    """
    if rolling is None or rolling.empty:
        out = pelt.copy()
    elif pelt is None or pelt.empty:
        out = rolling.copy()
    else:
        r = rolling.copy()
        if "source" not in r.columns:
            r["source"] = "rolling"
        out = pd.concat([r, pelt], axis=0).sort_index()

    if out.empty:
        return out

    if "source" not in out.columns:
        out["source"] = "rolling"

    # Rolling and PELT can emit the same calendar date.  Keep one scalar row
    # per normalized date before using ``.at`` below; otherwise ``.at[date]``
    # returns a Series and duplicate candidates can crash the merge.  Sensor
    # rows take precedence, then accepted candidates with stronger evidence.
    out = out.copy()
    out["_boundary_date"] = pd.to_datetime(out.index).normalize()
    sensor_priority = out["reason"].isin(
        ["sensor", "sensor_shadow"]
    ).astype(int)
    accepted_priority = out["accepted"].fillna(False).astype(int)
    out["_merge_priority"] = (
        2 * sensor_priority
        + accepted_priority
    )
    out = (
        out.sort_values(
            ["_boundary_date", "_merge_priority", "event_confidence", "z"],
            ascending=[True, False, False, False],
            na_position="last",
            kind="stable",
        )
        .drop_duplicates("_boundary_date", keep="first")
        .set_index("_boundary_date")
        .drop(columns=["_merge_priority"])
    )

    # Sensor-shadow veto must apply across detector sources. Any accepted
    # yaw-like candidate from either detector that lies inside the configured
    # sensor exclusion window is rejected before duplicate merging.
    sensor_dates = out.index[
        out["reason"].isin(["sensor", "sensor_shadow"])
    ]

    if len(sensor_dates):
        accepted_now = out.index[
            out["accepted"].fillna(False)
        ]

        for date in accepted_now:
            if out.at[date, "reason"] not in {"yaw", "pelt_yaw"}:
                continue

            if any(
                abs(
                    (
                        pd.Timestamp(date)
                        - pd.Timestamp(sensor_date)
                    ).days
                )
                <= config.sensor_exclusion_days
                for sensor_date in sensor_dates
            ):
                out.at[date, "accepted"] = False
                out.at[date, "reason"] = "sensor_shadow"
                out.at[date, "event_confidence"] = 0.0

    accepted_dates = list(
        out.index[out["accepted"].fillna(False)]
    )

    # Stronger event wins inside a short temporal neighbourhood.
    keep = []
    order = (
        out.loc[accepted_dates]
        .sort_values(
            ["event_confidence", "z"],
            ascending=False,
        )
        .index
        if accepted_dates
        else []
    )

    for date in order:
        if all(
            abs((pd.Timestamp(date) - pd.Timestamp(old)).days)
            > config.pelt_merge_days
            for old in keep
        ):
            keep.append(pd.Timestamp(date))

    accepted_mask = out["accepted"].fillna(False)
    for date in out.index[accepted_mask]:
        if pd.Timestamp(date) not in keep:
            out.at[date, "accepted"] = False
            if out.at[date, "reason"] in {"yaw", "pelt_yaw"}:
                out.at[date, "reason"] = "duplicate_candidate"
                out.at[date, "event_confidence"] = 0.0

    return out.sort_index()


def label_free_boundaries(
    daily: pd.DataFrame,
    pair_daily: pd.DataFrame,
    pair_weight: np.ndarray,
    config: RelativeStateConfig,
) -> pd.DataFrame:
    """
    Detect target-specific persistent steps before supervised calibration.

    V7.2 keeps candidate events, but the prediction later treats the constant
    B0 line as a prior: every accepted event carries an evidence-based
    shrinkage weight rather than a full-strength correction.
    """
    target = (
        daily["relative_heading_smooth"]
        .dropna()
        .sort_index()
    )

    columns = [
        "change",
        "background_change",
        "noise",
        "z",
        "support",
        "pair_agreement",
        "event_confidence",
        "common_mode",
        "sensor_like",
        "reason",
        "accepted",
    ]

    if target.empty:
        return pd.DataFrame(columns=columns)

    start = (
        target.index.min()
        + pd.Timedelta(days=config.change_window_days)
    )
    end = (
        target.index.max()
        - pd.Timedelta(days=config.change_window_days)
    )
    dates = pd.date_range(
        start,
        end,
        freq=f"{config.candidate_stride_days}D",
    )

    rows = []

    for date in dates:
        change, noise, support = _persistent_shift(
            target, date, config
        )
        background_change, _, _ = _persistent_shift(
            daily["background_residual_smooth"],
            date,
            config,
        )

        z = (
            abs(change)
            / max(noise, config.noise_floor_deg)
            if np.isfinite(change)
            and np.isfinite(noise)
            else 0.0
        )

        agreement = _pair_step_agreement(
            pair_daily,
            pair_weight,
            date,
            change,
            config,
        )

        persistent = (
            z >= config.z_threshold
            and np.isfinite(change)
            and abs(change)
            >= config.min_degree_change
        )

        common_mode = bool(
            persistent
            and np.isfinite(background_change)
            and abs(background_change)
            >= max(
                config.common_mode_min_deg,
                config.common_mode_ratio
                * abs(change),
            )
        )

        sensor_like = bool(
            persistent
            and not common_mode
            and abs(change)
            > config.sensor_jump_deg
            and agreement
            >= config.min_pair_agreement
        )

        confidence = _event_confidence(
            z,
            support,
            agreement,
            change,
            background_change,
            config,
        )

        if sensor_like:
            reason = "sensor"
        elif common_mode:
            reason = "common_mode"
        elif (
            persistent
            and agreement
            >= config.min_pair_agreement
        ):
            reason = "yaw"
        elif persistent:
            reason = "pair_disagreement"
        else:
            reason = "noise"

        rows.append({
            "date": date,
            "change": change,
            "background_change": background_change,
            "noise": noise,
            "z": z,
            "support": support,
            "pair_agreement": agreement,
            "event_confidence": confidence,
            "common_mode": common_mode,
            "sensor_like": sensor_like,
            "reason": reason,
        })

    if not rows:
        return pd.DataFrame(columns=columns)

    table = pd.DataFrame(rows).set_index("date")

    local_peak = table["z"].eq(
        table["z"].rolling(
            3,
            center=True,
            min_periods=1,
        ).max()
    )
    boundaries = table[
        local_peak
        & table["reason"].ne("noise")
    ].copy()
    boundaries["accepted"] = False

    # Sensor changes can smear through the 21-day windows and reappear as a
    # false yaw boundary a few weeks later.  Veto a neighbourhood around them.
    sensor_dates = boundaries.index[
        boundaries["reason"].eq("sensor")
    ]

    for date in boundaries.index[
        boundaries["reason"].eq("yaw")
    ]:
        if any(
            abs((date - sensor_date).days)
            <= config.sensor_exclusion_days
            for sensor_date in sensor_dates
        ):
            boundaries.at[date, "reason"] = (
                "sensor_shadow"
            )
            boundaries.at[date, "event_confidence"] = 0.0

    selected: list[pd.Timestamp] = []

    for date in boundaries.sort_values(
        ["event_confidence", "z"],
        ascending=False,
    ).index:
        if boundaries.at[date, "reason"] != "yaw":
            continue

        if all(
            abs((date - old).days)
            >= config.min_state_days
            for old in selected
        ):
            selected.append(date)

    boundaries.loc[selected, "accepted"] = True
    boundaries["source"] = "rolling"

    pelt = _stable_pelt_candidates(
        daily,
        pair_daily,
        pair_weight,
        config,
    )

    return _merge_boundary_tables(
        boundaries.sort_index(),
        pelt,
        config,
    )


def clusters_from_boundaries(
    index: pd.DatetimeIndex,
    boundaries: pd.DataFrame,
) -> pd.Series:
    if boundaries is None or boundaries.empty:
        return pd.Series(
            0,
            index=index,
            dtype=int,
        )

    accepted = boundaries[
        boundaries["accepted"]
    ]
    if accepted.empty:
        return pd.Series(
            0,
            index=index,
            dtype=int,
        )

    # The boundary table can contain repeated candidates for the same
    # calendar day after merging rolling and PELT proposals.  A repeated
    # date must not create an artificial two-state jump in a daily series.
    dates = np.unique(
        pd.DatetimeIndex(pd.to_datetime(accepted.index)).normalize().to_numpy(
            dtype="datetime64[ns]"
        )
    )

    return pd.Series(
        np.searchsorted(
            dates,
            index.to_numpy(
                dtype="datetime64[ns]"
            ),
            side="right",
        ),
        index=index,
        dtype=int,
    )
