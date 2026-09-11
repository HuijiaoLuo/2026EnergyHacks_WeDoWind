from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd

from baseline_loto_ridge import filter_operating_rows, wrap_180


@dataclass(frozen=True)
class FleetConfig:
    bin_freq: str = "30min"
    sector_width_deg: float = 30.0
    wake_half_width_deg: float = 20.0

    # Adaptive candidate pool. Broad pool + continuous distance weight;
    # no fixed neighbour count.
    isolated_ratio: float = 3.5
    radius_nn_factor: float = 4.0
    radius_site_factor: float = 3.5
    max_candidates: int = 12
    min_candidate_distance_weight: float = 0.03

    # Dynamic weights.
    distance_power: float = 1.5
    coherence_scale_deg: float = 8.0
    min_dynamic_weight: float = 0.04

    # Rolling field diagnostics.
    entropy_window_bins: int = 24       # 12 h at 30 min
    entropy_direction_bins: int = 12
    entropy_delta_bins: int = 12

    # Local spatial plane.
    plane_ridge: float = 1e-5


def circular_mean_deg(values, weights=None) -> float:
    x = np.asarray(pd.Series(values, dtype=float), dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return np.nan

    x = x[valid]
    if weights is None:
        w = np.ones_like(x)
    else:
        w0 = np.asarray(weights, dtype=float)[valid]
        good = np.isfinite(w0) & (w0 > 0)
        x, w = x[good], w0[good]
        if len(x) == 0:
            return np.nan

    z = np.sum(w * np.exp(1j * np.deg2rad(x))) / np.sum(w)
    return float(np.rad2deg(np.angle(z)))


def circular_median_deg(values) -> float:
    """Robust circular centre, returned in [-180, 180)."""
    x = np.asarray(pd.Series(values, dtype=float), dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    origin = circular_mean_deg(x)
    return float(wrap_180(origin + np.median(wrap_180(x - origin))))


def _aligned_matrix(
    binned: dict[str, pd.DataFrame],
    ids: list[str],
    column: str,
    index: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    if not ids:
        return pd.DataFrame(index=index)
    out = pd.concat(
        {tid: binned[tid][column] for tid in ids},
        axis=1,
    ).sort_index()
    return out.reindex(index) if index is not None else out


def _weighted_circular_mean_matrix(
    degrees: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(degrees) & np.isfinite(weights) & (weights > 0)
    w = np.where(valid, weights, 0.0)
    rad = np.deg2rad(np.where(valid, degrees, 0.0))
    z = np.sum(w * np.exp(1j * rad), axis=1)
    den = np.sum(w, axis=1)
    out = np.full(len(den), np.nan, dtype=float)
    good = den > 0
    out[good] = np.rad2deg(np.angle(z[good]))
    return out


def _weighted_circular_std_matrix(
    degrees: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(degrees) & np.isfinite(weights) & (weights > 0)
    w = np.where(valid, weights, 0.0)
    rad = np.deg2rad(np.where(valid, degrees, 0.0))
    z = np.sum(w * np.exp(1j * rad), axis=1)
    den = np.sum(w, axis=1)
    out = np.full(len(den), np.nan, dtype=float)
    good = den > 0
    R = np.clip(np.abs(z[good]) / den[good], 1e-9, 1.0)
    out[good] = np.rad2deg(np.sqrt(-2.0 * np.log(R)))
    return out


def _rolling_entropy_from_categories(
    categories: np.ndarray,
    n_bins: int,
    window: int,
    min_count: int = 4,
) -> np.ndarray:
    """
    Vectorized rolling normalized entropy.

    categories can be shape (T,) or (T, P); invalid values are -1.
    There is no timestamp loop and no neighbour loop.
    """
    cat = np.asarray(categories, dtype=int)
    if cat.ndim == 1:
        cat = cat[:, None]

    t, p = cat.shape
    counts = np.zeros((t, n_bins), dtype=float)

    row_ids = np.repeat(np.arange(t), p)
    flat = cat.ravel()
    valid = (flat >= 0) & (flat < n_bins)
    np.add.at(counts, (row_ids[valid], flat[valid]), 1.0)

    cs = np.vstack([np.zeros((1, n_bins)), np.cumsum(counts, axis=0)])
    end = np.arange(1, t + 1)
    start = np.maximum(0, end - window)
    rolling_counts = cs[end] - cs[start]

    total = rolling_counts.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = rolling_counts / total[:, None]
        logp = np.where(probs > 0, np.log(probs), 0.0)
        entropy = -np.sum(probs * logp, axis=1) / np.log(n_bins)

    entropy[total < min_count] = np.nan
    return entropy


def _categorize(
    values: np.ndarray,
    n_bins: int,
    lo: float,
    hi: float,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    valid = np.isfinite(x)
    clipped = np.where(
        valid,
        np.clip(x, lo, np.nextafter(hi, lo)),
        lo,
    )
    cat = np.floor((clipped - lo) / (hi - lo) * n_bins).astype(int)
    cat[~valid] = -1
    return cat


def canonical_tid(value, site: str) -> str:
    s = str(value).strip()
    if s.startswith(site + "_"):
        return s
    if s.startswith("WTG"):
        return f"{site}_{s}"
    m = re.search(r"(\d+)$", s)
    return f"{site}_WTG{int(m.group(1)):02d}" if m else s


def load_layout(root: Path, turbines: list[str]) -> pd.DataFrame:
    frames = []
    for site in ("PPP", "SSS"):
        name = f"turbine_locations_{site}.csv"
        candidates = [root / "data" / name, root / name, root.parent / name]
        path = next((x for x in candidates if x.exists()), None)
        if path is None:
            hits = list(root.parent.rglob(name))
            if not hits:
                continue
            path = hits[0]

        df = pd.read_csv(path)
        lower = {c.lower(): c for c in df.columns}
        id_col = next(
            (lower[c] for c in ("turbine_id", "turbine", "name", "id") if c in lower),
            None,
        )
        if id_col is None:
            id_col = next((c for c in df.columns if df[c].dtype == "object"), df.columns[0])

        x_col = next(
            (c for c in df.columns if c.lower() in {"x", "easting", "lon", "longitude", "x_coord"}),
            None,
        )
        y_col = next(
            (c for c in df.columns if c.lower() in {"y", "northing", "lat", "latitude", "y_coord"}),
            None,
        )
        if x_col is None or y_col is None:
            numeric = list(df.select_dtypes(include=np.number).columns)
            if len(numeric) < 2:
                raise ValueError(f"Cannot infer coordinates from {path}")
            x_col, y_col = numeric[:2]

        frames.append(pd.DataFrame({
            "turbine_id": [canonical_tid(v, site) for v in df[id_col]],
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
            "site": site,
        }))

    if not frames:
        raise FileNotFoundError("No turbine location file was found.")

    out = pd.concat(frames, ignore_index=True)
    return (
        out.dropna(subset=["x", "y"])
        .drop_duplicates("turbine_id")
        .query("turbine_id in @turbines")
        .set_index("turbine_id")
    )


def bin_turbine(frame: pd.DataFrame, config: FleetConfig) -> pd.DataFrame:
    x = filter_operating_rows(frame).copy()
    x = x[x["WindSpeed"].between(4.0, 12.0)]
    x["bin"] = pd.to_datetime(x["timestamp"]).dt.floor(config.bin_freq)
    x["vane"] = wrap_180(x["WindDir"] - x["NacDir"])

    g = x.groupby("bin")
    return pd.DataFrame({
        "WindSpeed": g["WindSpeed"].median(),
        "Power": g["Power"].median(),
        "PitchAngle": g["PitchAngle"].median(),
        "RotSpeed": g["RotSpeed"].median(),
        "NacDir": g["NacDir"].apply(circular_mean_deg),
        "WindDir": g["WindDir"].apply(circular_mean_deg),
        "vane": g["vane"].apply(circular_mean_deg),
        "n": g.size(),
    })


def _site_distance_matrix(layout: pd.DataFrame, site: str) -> tuple[list[str], np.ndarray]:
    site_layout = layout[layout["site"] == site]
    ids = site_layout.index.tolist()
    xy = site_layout[["x", "y"]].to_numpy(dtype=float)
    delta = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt(np.sum(delta ** 2, axis=2))
    np.fill_diagonal(d, np.inf)
    return ids, d


def site_nearest_distance_scale(layout: pd.DataFrame, site: str) -> float:
    ids, d = _site_distance_matrix(layout, site)
    if len(ids) < 2:
        return np.nan
    return float(np.median(np.min(d, axis=1)))


def candidate_neighbours(
    turbine: str,
    layout: pd.DataFrame,
    config: FleetConfig,
) -> pd.DataFrame:
    """
    Broad adaptive candidate pool + continuous distance weighting.

    The pool is not a fixed K. Distant turbines are not automatically assigned
    equal status; their base weights decay smoothly and dynamic wind/wake
    conditions decide whether they are useful at each time.
    """
    row = layout.loc[turbine]
    site = str(row["site"])
    site_layout = layout[layout["site"] == site]
    same = site_layout.drop(index=turbine, errors="ignore").copy()
    if same.empty:
        return pd.DataFrame(columns=["distance", "distance_weight"])

    target_xy = row[["x", "y"]].to_numpy(dtype=float)
    xy = same[["x", "y"]].to_numpy(dtype=float)
    same["distance"] = np.sqrt(np.sum((xy - target_xy) ** 2, axis=1))
    same = same.sort_values("distance")

    d1 = float(same["distance"].iloc[0])
    site_scale = site_nearest_distance_scale(layout, site)

    if np.isfinite(site_scale) and d1 > config.isolated_ratio * site_scale:
        return pd.DataFrame(columns=["distance", "distance_weight"])

    radius = max(
        config.radius_nn_factor * d1,
        config.radius_site_factor * site_scale if np.isfinite(site_scale) else 0.0,
    )
    cand = same[same["distance"] <= radius].head(config.max_candidates).copy()

    local_scale = max(
        d1,
        site_scale if np.isfinite(site_scale) else d1,
        1e-9,
    )
    ratio = cand["distance"].to_numpy(dtype=float) / local_scale
    cand["distance_weight"] = 1.0 / (1.0 + ratio ** config.distance_power)
    cand = cand[cand["distance_weight"] >= config.min_candidate_distance_weight]
    return cand[["distance", "distance_weight"]]


def dynamic_neighbour_weights(
    turbine: str,
    binned: dict[str, pd.DataFrame],
    layout: pd.DataFrame,
    config: FleetConfig,
) -> pd.DataFrame:
    """
    Vectorized time-varying weights:
        distance x wake x directional coherence x coverage.

    No timestamp x neighbour nested loop.
    """
    cand = candidate_neighbours(turbine, layout, config)
    if cand.empty:
        return pd.DataFrame()

    ids = cand.index.tolist()
    dirs_df = _aligned_matrix(binned, ids, "WindDir")
    if dirs_df.empty:
        return pd.DataFrame()

    dirs = dirs_df.to_numpy(dtype=float)
    base = cand["distance_weight"].to_numpy(dtype=float)[None, :]
    available = np.isfinite(dirs)

    base_available = np.where(available, base, 0.0)
    consensus = _weighted_circular_mean_matrix(dirs, base_available)

    delta_consensus = np.abs(
        wrap_180(dirs - consensus[:, None])
    )
    coherence = np.exp(
        -np.square(delta_consensus / config.coherence_scale_deg)
    )

    target_xy = layout.loc[turbine, ["x", "y"]].to_numpy(dtype=float)
    nb_xy = layout.loc[ids, ["x", "y"]].to_numpy(dtype=float)
    dx = nb_xy[:, 0] - target_xy[0]
    dy = nb_xy[:, 1] - target_xy[1]
    bearings = np.degrees(np.arctan2(dx, dy)) % 360.0

    d0 = np.abs(wrap_180(dirs - bearings[None, :]))
    d1 = np.abs(wrap_180(dirs - ((bearings + 180.0) % 360.0)[None, :]))
    axis_distance = np.minimum(d0, d1)

    h = config.wake_half_width_deg
    wake = np.where(
        axis_distance <= h,
        0.0,
        np.where(axis_distance >= 2.0 * h, 1.0, (axis_distance - h) / h),
    )

    weights = base * coherence * wake
    weights = np.where(available, weights, 0.0)
    weights = np.where(weights >= config.min_dynamic_weight, weights, 0.0)

    return pd.DataFrame(weights, index=dirs_df.index, columns=ids)


def local_field_context(
    turbine: str,
    binned: dict[str, pd.DataFrame],
    layout: pd.DataFrame,
    config: FleetConfig,
    weights: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorized local field diagnostics.

    H_abs is diagnostic only. H_rel and H_delta measure disorder and reduce
    field confidence. A coherent slow rotation is therefore allowed.
    """
    if weights is None:
        weights = dynamic_neighbour_weights(turbine, binned, layout, config)
    if weights.empty:
        return pd.DataFrame(), weights

    ids = weights.columns.tolist()
    idx = weights.index
    w = weights.to_numpy(dtype=float)

    dirs = _aligned_matrix(binned, ids, "WindDir", idx).to_numpy(dtype=float)
    speeds = _aligned_matrix(binned, ids, "WindSpeed", idx).to_numpy(dtype=float)

    valid_dir = np.isfinite(dirs)
    valid_ws = np.isfinite(speeds)
    w_dir = np.where(valid_dir, w, 0.0)
    w_ws = np.where(valid_ws, w, 0.0)

    n_refs = np.sum(w > 0, axis=1)
    field_dir = _weighted_circular_mean_matrix(dirs, w_dir)

    den_ws = np.sum(w_ws, axis=1)
    ws_ref_mean = np.divide(
        np.sum(w_ws * np.where(valid_ws, speeds, 0.0), axis=1),
        den_ws,
        out=np.full(len(idx), np.nan),
        where=den_ws > 0,
    )
    ws_spread = np.sqrt(
        np.divide(
            np.sum(
                w_ws
                * np.square(
                    np.where(valid_ws, speeds, ws_ref_mean[:, None])
                    - ws_ref_mean[:, None]
                ),
                axis=1,
            ),
            den_ws,
            out=np.full(len(idx), np.nan),
            where=den_ws > 0,
        )
    )

    rel_to_field = wrap_180(dirs - field_dir[:, None])
    rel_spread = _weighted_circular_std_matrix(rel_to_field, w_dir)

    field_delta = np.full(len(idx), np.nan, dtype=float)
    field_delta[1:] = wrap_180(np.diff(field_dir))

    # Pairwise relative field matrix, all pairs at once.
    n = len(ids)
    if n >= 2:
        ia, ib = np.triu_indices(n, k=1)
        pair_rel = wrap_180(dirs[:, ia] - dirs[:, ib])
        pair_active = (w[:, ia] > 0) & (w[:, ib] > 0)
        pair_rel = np.where(pair_active, pair_rel, np.nan)
    else:
        pair_rel = np.empty((len(idx), 0), dtype=float)

    h_abs = _rolling_entropy_from_categories(
        _categorize(field_dir % 360.0, config.entropy_direction_bins, 0.0, 360.0),
        config.entropy_direction_bins,
        config.entropy_window_bins,
    )
    h_delta = _rolling_entropy_from_categories(
        _categorize(np.clip(field_delta, -45.0, 45.0), config.entropy_delta_bins, -45.0, 45.0),
        config.entropy_delta_bins,
        config.entropy_window_bins,
    )
    if pair_rel.shape[1] > 0:
        h_rel = _rolling_entropy_from_categories(
            _categorize(np.clip(pair_rel, -90.0, 90.0), config.entropy_delta_bins, -90.0, 90.0),
            config.entropy_delta_bins,
            config.entropy_window_bins,
        )
    else:
        h_rel = np.full(len(idx), np.nan)

    context = pd.DataFrame({
        "n_refs": n_refs,
        "field_dir": field_dir,
        "ws_ref_mean": ws_ref_mean,
        "ws_spread": ws_spread,
        "rel_spread": rel_spread,
        "field_delta": field_delta,
        "H_abs": h_abs,
        "H_delta": h_delta,
        "H_rel": h_rel,
    }, index=idx)

    coverage_target = max(1, min(3, weights.shape[1]))
    coverage = np.minimum(context["n_refs"] / coverage_target, 1.0)
    rel_score = np.exp(-np.square(context["rel_spread"].fillna(30.0) / 8.0))
    ws_scale = context["ws_ref_mean"].abs().clip(lower=1e-6)
    ws_score = np.exp(
        -np.square((context["ws_spread"] / ws_scale).fillna(1.0) / 0.20)
    )
    entropy_score = np.exp(-1.5 * context["H_rel"].fillna(1.0))
    delta_score = np.exp(-1.2 * context["H_delta"].fillna(1.0))

    context["field_confidence"] = (
        coverage * rel_score * ws_score * entropy_score * delta_score
    ).clip(0.0, 1.0)

    return context, weights


def spatial_wind_reference(
    turbine: str,
    binned: dict[str, pd.DataFrame],
    layout: pd.DataFrame,
    context: pd.DataFrame,
    weights: pd.DataFrame,
    config: FleetConfig,
) -> pd.DataFrame:
    """
    Batched weighted local plane.

    >=3 active references: vectorized 3x3 weighted least-squares solve.
    <3 references: weighted mean fallback.
    """
    if weights.empty:
        return pd.DataFrame()

    ids = weights.columns.tolist()
    idx = weights.index
    w0 = weights.to_numpy(dtype=float)

    ws = _aligned_matrix(binned, ids, "WindSpeed", idx).to_numpy(dtype=float)
    wd = _aligned_matrix(binned, ids, "WindDir", idx).to_numpy(dtype=float)

    valid_ws = np.isfinite(ws)
    w = np.where(valid_ws, w0, 0.0)
    n_refs = np.sum(w > 0, axis=1)

    target_xy = layout.loc[turbine, ["x", "y"]].to_numpy(dtype=float)
    nb_xy = layout.loc[ids, ["x", "y"]].to_numpy(dtype=float)
    rel_xy = nb_xy - target_xy
    scale = max(np.median(np.linalg.norm(rel_xy, axis=1)), 1e-9)
    X = np.column_stack([
        np.ones(len(ids)),
        rel_xy[:, 0] / scale,
        rel_xy[:, 1] / scale,
    ])

    y = np.where(valid_ws, ws, 0.0)
    A = np.einsum("tn,ni,nj->tij", w, X, X)
    b = np.einsum("tn,ni,tn->ti", w, X, y)

    ridge = config.plane_ridge * np.eye(3)[None, :, :]
    pinv = np.linalg.pinv(A + ridge)
    coef = np.einsum("tij,tj->ti", pinv, b)
    ws_plane = coef[:, 0]

    den = np.sum(w, axis=1)
    ws_mean = np.divide(
        np.sum(w * y, axis=1),
        den,
        out=np.full(len(idx), np.nan),
        where=den > 0,
    )
    ws_ref = np.where(n_refs >= 3, ws_plane, ws_mean)

    w_wd = np.where(np.isfinite(wd), w0, 0.0)
    wdir_ref = _weighted_circular_mean_matrix(wd, w_wd)

    field_conf = context["field_confidence"].reindex(idx).fillna(0.0).to_numpy()

    return pd.DataFrame({
        "ws_ref": ws_ref,
        "wdir_ref": wdir_ref,
        "n_refs": n_refs,
        "field_confidence": field_conf,
    }, index=idx)
