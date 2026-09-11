import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from baseline_loto_ridge import (
    detect_labelled_turbines,
    filter_operating_rows,
    metrics,
    read_turbine,
    wrap_180,
)


def prepare(df):
    data = filter_operating_rows(df)
    data = data.copy()
    data["gamma"] = wrap_180(data["WindDir"] - data["NacDir"])
    data["date_dt"] = pd.to_datetime(data["date"])
    return data


def normalise_power_by_wind_bin(data, wind_bin_width):
    """
    B0 normal-power reference.

    Within the current estimation window, WindSpeed is binned and each
    SCADA point is divided by the median Power of its wind-speed bin:

        power_norm = Power / median(Power | wind_bin)

    This removes the dominant wind-speed dependence without fitting an ML
    power model.
    """
    out = data.copy()
    min_ws = np.floor(out["WindSpeed"].min())
    max_ws = np.ceil(out["WindSpeed"].max()) + wind_bin_width
    bins = np.arange(min_ws, max_ws + wind_bin_width, wind_bin_width)

    out["wind_bin"] = pd.cut(
        out["WindSpeed"],
        bins=bins,
        include_lowest=True,
    )
    wind_power = (
        out.groupby("wind_bin", observed=True)["Power"]
        .transform("median")
    )
    out["power_norm"] = out["Power"] / np.maximum(wind_power, 1e-6)
    return out.dropna(subset=["wind_bin", "power_norm"])


def weighted_r2(y, y_hat, weights):
    """Weighted R^2 diagnostic for the quadratic power-vane curve."""
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(y) < 2 or np.sum(weights) <= 0:
        return np.nan

    y_bar = np.average(y, weights=weights)
    ss_res = np.sum(weights * (y - y_hat) ** 2)
    ss_tot = np.sum(weights * (y - y_bar) ** 2)

    if ss_tot <= 0:
        return np.nan
    return float(1.0 - ss_res / ss_tot)


def _peak_observability(theta, gamma_min, gamma_max):
    """
    Distance from an estimated peak to the nearer edge of the observed
    reliable vane-angle support.

    Large positive values mean the peak is bracketed by data on both sides.
    A value near zero means the estimate sits at an edge and is effectively
    one-sided.
    """
    if not (
        np.isfinite(theta)
        and np.isfinite(gamma_min)
        and np.isfinite(gamma_max)
    ):
        return np.nan
    return float(min(theta - gamma_min, gamma_max - theta))


def estimate_peak_angle(
    window,
    wind_bin_width,
    angle_bin_width,
    min_samples_per_angle_bin,
    min_angle_bins,
    gamma_min=-30.0,
    gamma_max=30.0,
    normalizer=None,
):
    """
    Estimate the apparent power-optimal vane angle for one SCADA window.

    Primary B0 estimator:
        theta_hat_argmax = gamma of the valid vane-angle bin with the
        largest median normalised power.

    Diagnostic B0-quadratic estimator:
        Fit a sample-count-weighted quadratic to valid angle bins and use
        its vertex only when the fit is concave and the vertex lies inside
        the observed valid-bin support.

    `theta_hat` is retained as an alias of theta_hat_argmax so older scripts
    remain compatible.
    """
    data = (normalizer or normalise_power_by_wind_bin)(window, wind_bin_width)

    bins = np.arange(
        gamma_min,
        gamma_max + angle_bin_width,
        angle_bin_width,
    )
    data = data[
        (data["gamma"] >= bins[0])
        & (data["gamma"] <= bins[-1])
    ].copy()

    data["angle_bin"] = pd.cut(
        data["gamma"],
        bins=bins,
        include_lowest=True,
    )

    curve = (
        data.groupby("angle_bin", observed=True)
        .agg(
            gamma=("gamma", "median"),
            power_norm=("power_norm", "median"),
            n=("Power", "size"),
        )
        .reset_index(drop=True)
        .sort_values("gamma")
        .reset_index(drop=True)
    )

    curve = curve[
        curve["n"] >= min_samples_per_angle_bin
    ].copy()

    base = {
        "theta_hat": np.nan,  # backward-compatible alias for argmax
        "theta_hat_argmax": np.nan,
        "theta_hat_quadratic": np.nan,
        "n_rows": int(len(window)),
        "n_curve_bins": int(len(curve)),
        "status": "too_few_angle_bins",
        "quadratic_status": "not_attempted",
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

    if len(curve) == 0:
        base["status"] = "no_valid_angle_bins"
        base["quadratic_status"] = "no_valid_angle_bins"
        return base

    gamma_min_used = float(curve["gamma"].min())
    gamma_max_used = float(curve["gamma"].max())
    gamma_span = float(gamma_max_used - gamma_min_used)

    base["gamma_min_used"] = gamma_min_used
    base["gamma_max_used"] = gamma_max_used
    base["gamma_span"] = gamma_span

    if len(curve) < min_angle_bins:
        base["quadratic_status"] = "too_few_angle_bins"
        return base

    # ------------------------------------------------------------------
    # B0 primary estimator: highest valid binned median normalised power.
    # ------------------------------------------------------------------
    top = curve.loc[curve["power_norm"].idxmax()]
    theta_argmax = float(top["gamma"])

    base["theta_hat"] = theta_argmax
    base["theta_hat_argmax"] = theta_argmax
    base["status"] = "ok"
    base["argmax_observability"] = _peak_observability(
        theta_argmax,
        gamma_min_used,
        gamma_max_used,
    )

    # ------------------------------------------------------------------
    # B0-quadratic diagnostic: smooth the valid empirical curve.
    # ------------------------------------------------------------------
    x = curve["gamma"].to_numpy(dtype=float)
    y = curve["power_norm"].to_numpy(dtype=float)
    n = curve["n"].to_numpy(dtype=float)

    # np.polyfit minimises sum((w * residual)^2), therefore sqrt(n)
    # produces an effective weight proportional to sample count n.
    fit_weights = np.sqrt(n)

    try:
        a, b, c = np.polyfit(
            x,
            y,
            deg=2,
            w=fit_weights,
        )
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        base["quadratic_status"] = "fit_failed"
        return base

    y_hat = a * x**2 + b * x + c
    fit_r2 = weighted_r2(y, y_hat, n)

    base["quadratic_a"] = float(a)
    base["quadratic_b"] = float(b)
    base["quadratic_c"] = float(c)
    base["fit_r2"] = fit_r2
    base["curvature"] = float(-a)

    if not np.isfinite(a) or a >= 0:
        base["quadratic_status"] = "non_concave_curve"
        return base

    theta_quad = float(-b / (2.0 * a))

    if not (gamma_min_used <= theta_quad <= gamma_max_used):
        base["quadratic_status"] = "peak_outside_support"
        return base

    base["theta_hat_quadratic"] = theta_quad
    base["quadratic_status"] = "ok"
    base["quadratic_observability"] = _peak_observability(
        theta_quad,
        gamma_min_used,
        gamma_max_used,
    )
    return base


def build_windows(data, window_mode, label_bin_width, rolling_window_days):
    if window_mode == "label-state":
        out = data.copy()
        out["label_window"] = out["yaw_misalignment_deg"].round(6)
        group_cols = ["turbine_id", "label_window"]

    elif window_mode == "label-bin":
        out = data.copy()
        out["label_window"] = (
            np.round(
                out["yaw_misalignment_deg"] / label_bin_width
            )
            * label_bin_width
        )
        group_cols = ["turbine_id", "label_window"]

    elif window_mode == "daily":
        out = data
        group_cols = ["turbine_id", "date"]

    elif window_mode == "rolling":
        half_before = (rolling_window_days - 1) // 2
        half_after = rolling_window_days - 1 - half_before

        windows = []
        for turbine_id, turbine_data in data.groupby(
            "turbine_id",
            sort=True,
        ):
            unique_dates = (
                turbine_data[
                    ["date", "date_dt", "yaw_misalignment_deg"]
                ]
                .groupby(
                    ["date", "date_dt"],
                    as_index=False,
                )["yaw_misalignment_deg"]
                .median()
                .sort_values("date_dt")
            )

            for _, date_row in unique_dates.iterrows():
                start = (
                    date_row["date_dt"]
                    - pd.Timedelta(days=half_before)
                )
                end = (
                    date_row["date_dt"]
                    + pd.Timedelta(days=half_after)
                )

                group = turbine_data[
                    (turbine_data["date_dt"] >= start)
                    & (turbine_data["date_dt"] <= end)
                ]

                windows.append(
                    {
                        "turbine_id": turbine_id,
                        "target": float(
                            date_row["yaw_misalignment_deg"]
                        ),
                        "date_start": start.date().isoformat(),
                        "date_end": end.date().isoformat(),
                        "date": date_row["date"],
                        "data": group,
                    }
                )

        return windows

    else:
        raise ValueError(
            f"Unknown window mode: {window_mode}"
        )

    windows = []
    for keys, group in out.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = dict(zip(group_cols, keys))
        row["target"] = float(
            group["yaw_misalignment_deg"].median()
        )
        row["date_start"] = group["date"].min()
        row["date_end"] = group["date"].max()
        row["data"] = group
        windows.append(row)

    return windows


def fit_calibration(train_predictions, prediction_column):
    good = train_predictions.dropna(
        subset=[prediction_column, "target"]
    )

    if (
        len(good) < 2
        or good[prediction_column].std() == 0.0
    ):
        return {
            "slope": 1.0,
            "intercept": 0.0,
        }

    x = good[prediction_column].to_numpy(dtype=float)
    y = good["target"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
    }


def apply_calibration(theta_hat, calibration):
    return (
        calibration["slope"] * theta_hat
        + calibration["intercept"]
    )


def estimate_windows(data, args):
    rows = []

    for window in build_windows(
        data,
        args.window_mode,
        args.label_bin_width,
        args.rolling_window_days,
    ):
        result = estimate_peak_angle(
            window["data"],
            args.wind_bin_width,
            args.angle_bin_width,
            args.min_samples_per_angle_bin,
            args.min_angle_bins,
            gamma_min=args.gamma_min,
            gamma_max=args.gamma_max,
            normalizer=getattr(args, "normalizer", None),
        )

        rows.append(
            {
                "turbine_id": window["turbine_id"],
                "window_mode": args.window_mode,
                "target": window["target"],
                "date_start": window["date_start"],
                "date_end": window["date_end"],
                "date": window.get("date", ""),
                **result,
            }
        )

    return pd.DataFrame(rows)


def load_labelled(zip_path, train_turbines):
    frames = []

    for turbine_id in train_turbines:
        data = prepare(
            read_turbine(zip_path, turbine_id)
        )
        data = data[
            data["yaw_misalignment_deg"].notna()
        ].copy()
        frames.append(data)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def _safe_metrics(frame, prediction_column):
    scored = frame.dropna(
        subset=["target", prediction_column]
    )
    if len(scored) == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
        }

    return metrics(
        scored["target"].to_numpy(dtype=float),
        scored[prediction_column].to_numpy(dtype=float),
    )


def run_loto(labelled, train_turbines, args):
    """
    Leave-one-turbine-out evaluation.

    Backward-compatible columns:
      rmse/mae/bias         -> calibrated argmax B0
      raw_rmse/raw_mae/...  -> uncalibrated argmax B0
      coverage              -> argmax B0 coverage

    Additional columns compare the quadratic diagnostic estimator.
    """
    fold_rows = []
    predictions = []

    for holdout in train_turbines:
        train_data = labelled[
            labelled["turbine_id"] != holdout
        ]
        valid_data = labelled[
            labelled["turbine_id"] == holdout
        ]

        train_pred = estimate_windows(
            train_data,
            args,
        )

        argmax_cal = fit_calibration(
            train_pred,
            "theta_hat_argmax",
        )
        quad_cal = fit_calibration(
            train_pred,
            "theta_hat_quadratic",
        )

        valid_pred = estimate_windows(
            valid_data,
            args,
        )

        valid_pred[
            "yaw_misalignment_deg_argmax_calibrated"
        ] = apply_calibration(
            valid_pred["theta_hat_argmax"],
            argmax_cal,
        )

        valid_pred[
            "yaw_misalignment_deg_quadratic_calibrated"
        ] = apply_calibration(
            valid_pred["theta_hat_quadratic"],
            quad_cal,
        )

        # Backward-compatible name: calibrated primary B0 estimator.
        valid_pred["yaw_misalignment_deg"] = (
            valid_pred[
                "yaw_misalignment_deg_argmax_calibrated"
            ]
        )

        valid_pred["calibration_slope"] = (
            argmax_cal["slope"]
        )
        valid_pred["calibration_intercept"] = (
            argmax_cal["intercept"]
        )
        valid_pred["quad_calibration_slope"] = (
            quad_cal["slope"]
        )
        valid_pred["quad_calibration_intercept"] = (
            quad_cal["intercept"]
        )
        valid_pred["holdout_turbine"] = holdout

        predictions.append(valid_pred)

        # Primary B0: argmax
        raw_arg = _safe_metrics(
            valid_pred,
            "theta_hat_argmax",
        )
        cal_arg = _safe_metrics(
            valid_pred,
            "yaw_misalignment_deg_argmax_calibrated",
        )

        # Diagnostic B0-quadratic
        raw_quad = _safe_metrics(
            valid_pred,
            "theta_hat_quadratic",
        )
        cal_quad = _safe_metrics(
            valid_pred,
            "yaw_misalignment_deg_quadratic_calibrated",
        )

        n_total = int(len(valid_pred))
        n_arg = int(
            valid_pred["theta_hat_argmax"]
            .notna()
            .sum()
        )
        n_quad = int(
            valid_pred["theta_hat_quadratic"]
            .notna()
            .sum()
        )

        row = {
            # Backward-compatible calibrated argmax metrics.
            "rmse": cal_arg["rmse"],
            "mae": cal_arg["mae"],
            "bias": cal_arg["bias"],

            # Backward-compatible raw argmax metrics.
            "raw_rmse": raw_arg["rmse"],
            "raw_mae": raw_arg["mae"],
            "raw_bias": raw_arg["bias"],

            # Explicit argmax names.
            "argmax_rmse": cal_arg["rmse"],
            "argmax_mae": cal_arg["mae"],
            "argmax_bias": cal_arg["bias"],
            "raw_argmax_rmse": raw_arg["rmse"],
            "raw_argmax_mae": raw_arg["mae"],
            "raw_argmax_bias": raw_arg["bias"],

            # Quadratic metrics.
            "quad_rmse": cal_quad["rmse"],
            "quad_mae": cal_quad["mae"],
            "quad_bias": cal_quad["bias"],
            "raw_quad_rmse": raw_quad["rmse"],
            "raw_quad_mae": raw_quad["mae"],
            "raw_quad_bias": raw_quad["bias"],

            "holdout_turbine": holdout,

            # Primary B0 coverage.
            "n_windows": n_arg,
            "n_failed_windows": n_total - n_arg,
            "coverage": (
                float(n_arg / n_total)
                if n_total
                else 0.0
            ),

            # Quadratic coverage.
            "quad_n_windows": n_quad,
            "quad_n_failed_windows": n_total - n_quad,
            "quad_coverage": (
                float(n_quad / n_total)
                if n_total
                else 0.0
            ),

            "calibration_slope": argmax_cal["slope"],
            "calibration_intercept": argmax_cal["intercept"],
            "quad_calibration_slope": quad_cal["slope"],
            "quad_calibration_intercept": quad_cal["intercept"],
        }

        fold_rows.append(row)

    return (
        pd.DataFrame(fold_rows),
        pd.concat(predictions, ignore_index=True),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "OpenOA-style B0 power-vs-vane baseline. "
            "The primary estimator is the maximum valid binned median "
            "normalised power (argmax). A weighted quadratic peak is also "
            "reported as a diagnostic variant."
        )
    )

    parser.add_argument(
        "--zip",
        default="turbines_data.zip",
        help="Path to turbines_data.zip.",
    )
    parser.add_argument(
        "--output",
        default=(
            "experiments/power_vane/"
            "openoa_power_vane_scores.csv"
        ),
    )
    parser.add_argument(
        "--prediction-output",
        default=(
            "experiments/power_vane/"
            "openoa_power_vane_predictions.csv"
        ),
    )
    parser.add_argument(
        "--train-turbines",
        nargs="+",
        help=(
            "Optional labelled turbines to use. "
            "By default, labelled turbines are auto-detected."
        ),
    )
    parser.add_argument(
        "--window-mode",
        choices=[
            "label-state",
            "label-bin",
            "daily",
            "rolling",
        ],
        default="rolling",
    )
    parser.add_argument(
        "--label-bin-width",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--rolling-window-days",
        type=int,
        default=21,
        help=(
            "Centered calendar window length for "
            "--window-mode rolling."
        ),
    )
    parser.add_argument(
        "--wind-bin-width",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--angle-bin-width",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--min-samples-per-angle-bin",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--min-angle-bins",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gamma-min",
        type=float,
        default=-30.0,
    )
    parser.add_argument(
        "--gamma-max",
        type=float,
        default=30.0,
    )

    args = parser.parse_args()

    train_turbines = (
        args.train_turbines
        or detect_labelled_turbines(args.zip)
    )

    print(
        "Labelled turbines:",
        ", ".join(train_turbines),
    )
    print(
        f"Window mode: {args.window_mode}"
    )
    if args.window_mode == "rolling":
        print(
            "Rolling window days:",
            args.rolling_window_days,
        )

    labelled = load_labelled(
        args.zip,
        train_turbines,
    )

    scores, predictions = run_loto(
        labelled,
        train_turbines,
        args,
    )

    Path(args.output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(args.prediction_output).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores.to_csv(
        args.output,
        index=False,
    )
    predictions.to_csv(
        args.prediction_output,
        index=False,
    )

    display_cols = [
        "holdout_turbine",
        "raw_argmax_mae",
        "raw_argmax_bias",
        "coverage",
        "raw_quad_mae",
        "raw_quad_bias",
        "quad_coverage",
        "argmax_mae",
        "quad_mae",
    ]

    print()
    print("Fold comparison:")
    print(
        scores[display_cols].to_string(
            index=False
        )
    )

    print()
    print(
        "Average raw argmax:",
        f"MAE={scores['raw_argmax_mae'].mean():.3f}",
        f"bias={scores['raw_argmax_bias'].mean():.3f}",
        f"coverage={scores['coverage'].mean():.3f}",
    )
    print(
        "Average raw quadratic:",
        f"MAE={scores['raw_quad_mae'].mean():.3f}",
        f"bias={scores['raw_quad_bias'].mean():.3f}",
        f"coverage={scores['quad_coverage'].mean():.3f}",
    )
    print(
        "Average calibrated argmax:",
        f"MAE={scores['argmax_mae'].mean():.3f}",
    )
    print(
        "Average calibrated quadratic:",
        f"MAE={scores['quad_mae'].mean():.3f}",
    )

    print()
    print(
        f"Wrote scores to {args.output}"
    )
    print(
        f"Wrote predictions to "
        f"{args.prediction_output}"
    )


if __name__ == "__main__":
    main()

