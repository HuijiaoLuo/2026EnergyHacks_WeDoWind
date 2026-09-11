"""Frozen long-term vane calibration. No search, context correction or dynamics.

Default CLI emits the explicitly frozen three-decimal point predictions.
--verify-scada checks the same estimator against private SCADA without changing
the frozen export. Notebook-facing functions return data and write nothing.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

TRAIN = ["PPP_WTG12", "PPP_WTG13", "PPP_WTG14"]
TARGETS = ["PPP_WTG17", "SSS_WTG06"]
FROZEN_POINTS = {"PPP_WTG17": -3.596, "SSS_WTG06": -4.568}
FIELDNAMES = ["turbine_id", "date", "yaw_misalignment_deg", "cluster"]
START, END = "2023-01-01", "2024-12-31"
FROZEN_B0 = SimpleNamespace(window_mode="rolling", label_bin_width=0.5,
    rolling_window_days=21, wind_bin_width=1.0, angle_bin_width=1.0,
    min_samples_per_angle_bin=20, min_angle_bins=8, gamma_min=-30., gamma_max=30., normalizer=None)


def load_daily(path):
    frame = pd.read_csv(path, parse_dates=["date"])
    required = {"turbine_id", "date", "target", "theta_hat_argmax"}
    if required - set(frame):
        raise ValueError(f"Missing columns: {required - set(frame)}")
    if frame.duplicated(["turbine_id", "date"]).any() or frame.date.isna().any():
        raise ValueError("Invalid or duplicate turbine/date keys")
    return frame.sort_values(["turbine_id", "date"]).reset_index(drop=True)


def turbine_summary(daily):
    summary = daily[daily.turbine_id.isin(TRAIN)].groupby("turbine_id").agg(
        target_mean=("target", "mean"), theta_star=("theta_hat_argmax", "median"))
    summary = summary.reindex(TRAIN)
    if not np.isfinite(summary.to_numpy()).all():
        raise ValueError("All three labelled turbines need finite target means and vane medians")
    summary["C_i"] = summary.target_mean + summary.theta_star
    return summary


def validate_loto(daily):
    """Fit C on two turbines; select sign on their means only (original cell 26).

    This reproduces training-only sign selection within each outer LOTO fold;
    it is not described as a separate inner cross-validation calculation.
    """
    summary = turbine_summary(daily)
    rows = []
    for holdout in TRAIN:
        train = summary.drop(index=holdout)
        sign_scores = {}
        for sign in [-1., 1.]:
            c = (train.target_mean - sign * train.theta_star).mean()
            sign_scores[sign] = np.abs(c + sign * train.theta_star - train.target_mean).mean()
        sign = min(sign_scores, key=sign_scores.get)
        c = float((train.target_mean - sign * train.theta_star).mean())
        prediction = float(c + sign * summary.loc[holdout, "theta_star"])
        y = daily.loc[daily.turbine_id.eq(holdout), "target"].dropna().to_numpy()
        error = prediction - y
        rows.append(dict(turbine_id=holdout, selected_sign=sign, C_hat=c,
                         prediction=prediction, n_days=len(y), mae=np.abs(error).mean(),
                         rmse=np.sqrt(np.mean(error**2)), bias=error.mean(),
                         oracle_mae=np.abs(np.median(y) - y).mean()))
    return pd.DataFrame(rows).set_index("turbine_id")


def predict_targets(daily):
    c = float(turbine_summary(daily).C_i.mean())
    rows = []
    for turbine in TARGETS:
        theta = daily.loc[daily.turbine_id.eq(turbine), "theta_hat_argmax"].dropna()
        if not len(theta):
            raise ValueError(f"No valid frozen B0 estimates for {turbine}")
        value = c - float(theta.median())
        rows.append(dict(turbine_id=turbine, theta_star=theta.median(),
                         yaw_full_precision=value, frozen_export=FROZEN_POINTS[turbine], n_valid=len(theta)))
    return pd.DataFrame(rows).set_index("turbine_id")


def _blocks(daily, start):
    data = daily.copy()
    data["block"] = (data.date - start).dt.days // 28
    return {t: {b: g[["target", "theta_hat_argmax"]].to_numpy(dtype=float)
                for b, g in tg.groupby("block", sort=False)}
            for t, tg in data.groupby("turbine_id", sort=False)}


def _sample(blocks, selected):
    return np.concatenate([blocks[b] for b in selected])


def bootstrap_loto(daily, n_boot=500, seed=42):
    """Shared 28-day blocks of cached B0 windows; not a raw-SCADA refit."""
    train = daily[daily.turbine_id.isin(TRAIN)]
    blocks = _blocks(train, train.date.min())
    common = sorted(set.intersection(*(set(blocks[t]) for t in TRAIN)))
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(n_boot):
        selected = rng.choice(common, len(common), replace=True)
        samples = {t: _sample(blocks[t], selected) for t in TRAIN}
        c = {t: np.nanmean(s[:, 0]) + np.nanmedian(s[:, 1]) for t, s in samples.items()}
        for holdout in TRAIN:
            prediction = np.mean([c[t] for t in TRAIN if t != holdout]) - np.nanmedian(samples[holdout][:, 1])
            rows.append((replicate, holdout, np.nanmean(np.abs(prediction - samples[holdout][:, 0]))))
    return pd.DataFrame(rows, columns=["replicate", "turbine_id", "mae"])


def bootstrap_targets(daily, n_boot=1000, seed=42):
    """Original cell 51: target blocks + independently sampled training blocks.

    Target rows with unavailable B0 estimates are excluded exactly as in the
    source notebook; training rows keep their labels even when B0 is missing.
    RNG order is targets, replicates, target sample, then each training turbine.
    """
    data = daily[daily.turbine_id.isin(TRAIN) |
                 (daily.turbine_id.isin(TARGETS) & daily.theta_hat_argmax.notna())]
    blocks = _blocks(data, data.date.min())
    rng = np.random.default_rng(seed)
    rows = []
    for target in TARGETS:
        target_blocks = sorted(blocks[target])
        for replicate in range(n_boot):
            selected = rng.choice(target_blocks, len(target_blocks), replace=True)
            theta = np.nanmedian(_sample(blocks[target], selected)[:, 1])
            components = []
            for turbine in TRAIN:
                available = list(blocks[turbine])
                selected = rng.choice(available, len(available), replace=True)
                sample = _sample(blocks[turbine], selected)
                components.append(np.nanmean(sample[:, 0]) + np.nanmedian(sample[:, 1]))
            rows.append((replicate, target, np.mean(components) - theta))
    return pd.DataFrame(rows, columns=["replicate", "turbine_id", "yaw_hat"])


def run_frozen_scada(zip_path, turbines=TRAIN + TARGETS):
    from baseline_openoa_power_vane_updated import prepare, estimate_windows
    from baseline_loto_ridge import read_turbine
    rows = []
    for turbine in turbines:
        data = prepare(read_turbine(zip_path, turbine))
        if "yaw_misalignment_deg" not in data:
            data["yaw_misalignment_deg"] = np.nan
        result = estimate_windows(data, FROZEN_B0)
        rows.append(result[["turbine_id", "date", "target", "theta_hat_argmax", "theta_hat_quadratic"]])
    return pd.concat(rows, ignore_index=True)


def submission(turbine):
    """All 731 required calendar days, including days failing operating filters."""
    dates = pd.date_range(START, END, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({"turbine_id": turbine, "date": dates,
                         "yaw_misalignment_deg": FROZEN_POINTS[turbine], "cluster": 0})[FIELDNAMES]


def validate_submission(frame, turbine, template=None):
    expected = submission(turbine)
    if list(frame.columns) != FIELDNAMES or len(frame) != len(expected):
        raise ValueError("Submission columns or calendar coverage differ")
    if frame.isna().any().any() or frame.duplicated(["turbine_id", "date"]).any():
        raise ValueError("Missing values or duplicate rows")
    if not frame.turbine_id.eq(turbine).all() or list(frame.date) != list(expected.date):
        raise ValueError("Wrong turbine, missing dates, or wrong order")
    if not np.allclose(frame.yaw_misalignment_deg, FROZEN_POINTS[turbine], rtol=0, atol=1e-10):
        raise ValueError("Prediction differs from explicitly frozen point")
    if not frame.cluster.eq(0).all():
        raise ValueError("Constant model must use one cluster (0)")
    if template is not None:
        if list(template.columns) != FIELDNAMES:
            raise ValueError("Previous submission schema differs")
        if not frame[["turbine_id", "date"]].equals(template[["turbine_id", "date"]]):
            raise ValueError("Previous submission turbine/date template differs")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily", type=Path, default=root / "data/vane_daily.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "submissions")
    parser.add_argument("--verify-scada", type=Path, help="Optional private turbines_data.zip; no SCADA is distributed")
    args = parser.parse_args()
    daily = load_daily(args.daily)
    if args.verify_scada:
        recomputed = run_frozen_scada(args.verify_scada)
        for turbine in TRAIN + TARGETS:
            old = daily[daily.turbine_id.eq(turbine)].theta_hat_argmax.median()
            new = recomputed[recomputed.turbine_id.eq(turbine)].theta_hat_argmax.median()
            np.testing.assert_allclose(new, old, rtol=0, atol=1e-8)
        print("Raw SCADA reproduced all five long-term vane levels.")
    print(validate_loto(daily).round(6).to_string())
    points = predict_targets(daily)
    for turbine in TARGETS:
        if round(points.loc[turbine, "yaw_full_precision"], 3) != FROZEN_POINTS[turbine]:
            raise ValueError("Cached data no longer reproduce the frozen prediction; export stopped")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for turbine, name in [("PPP_WTG17", "Results_30_T3_vane_0.csv"),
                          ("SSS_WTG06", "Results_30_T3_vane_final.csv")]:
        frame = submission(turbine)
        validate_submission(frame, turbine)
        frame.to_csv(args.output_dir / name, index=False, float_format="%.6f")
        print(f"{name}: {len(frame)} rows, yaw={FROZEN_POINTS[turbine]:.6f}, cluster=0")


if __name__ == "__main__":
    main()
