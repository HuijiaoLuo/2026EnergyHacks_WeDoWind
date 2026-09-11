import argparse
import csv
import io
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


SUBMISSION_COLUMNS = ["turbine_id", "date", "yaw_misalignment_deg", "cluster"]


def wrap_180(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def read_turbine(zip_path, turbine_id):
    with zipfile.ZipFile(zip_path) as z:
        parquet_name = f"{turbine_id}.parquet"
        if parquet_name not in z.namelist():
            raise FileNotFoundError(f"{parquet_name} is not in {zip_path}")
        df = pd.read_parquet(io.BytesIO(z.read(parquet_name)))

    time_col = "session" if "session" in df.columns else "ts"
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.rename(columns={time_col: "timestamp"})
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["turbine_id"] = turbine_id
    return df


def list_turbines(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        return sorted(Path(name).stem for name in z.namelist() if name.endswith(".parquet"))


def detect_labelled_turbines(zip_path):
    labelled = []
    for turbine_id in list_turbines(zip_path):
        df = read_turbine(zip_path, turbine_id)
        if "yaw_misalignment_deg" in df.columns and df["yaw_misalignment_deg"].notna().any():
            labelled.append(turbine_id)
    if len(labelled) < 2:
        raise ValueError(
            "Need at least two labelled turbines for leave-one-turbine-out validation."
        )
    return labelled


def filter_operating_rows(df):
    data = df.copy()
    data = data[
        (data["WindSpeed"] >= 3.0)
        & (data["WindSpeed"] <= 13.0)
        & (data["Power"] > 0.0)
        & (data["RotSpeed"] > 0.0)
        & (data["GenSpeed"] > 0.0)
    ]

    # Drop obvious curtailed/rated points. The exact scaling is anonymised, so use quantiles.
    power_cap = data["Power"].quantile(0.98)
    pitch_cap = data["PitchAngle"].quantile(0.95)
    data = data[(data["Power"] < power_cap) & (data["PitchAngle"] < pitch_cap)]
    return data


def aggregate_daily(df, has_label):
    data = filter_operating_rows(df)
    data["vane_angle"] = wrap_180(data["WindDir"] - data["NacDir"])
    data["vane_sin"] = np.sin(np.deg2rad(data["vane_angle"]))
    data["vane_cos"] = np.cos(np.deg2rad(data["vane_angle"]))
    data["power_per_wind3"] = data["Power"] / np.maximum(data["WindSpeed"], 0.1) ** 3

    features = []
    for signal in ["WindSpeed", "Power", "PitchAngle", "RotSpeed", "GenSpeed", "vane_angle", "power_per_wind3"]:
        features.extend(
            [
                (f"{signal}_mean", (signal, "mean")),
                (f"{signal}_std", (signal, "std")),
                (f"{signal}_q10", (signal, lambda s: s.quantile(0.10))),
                (f"{signal}_q50", (signal, "median")),
                (f"{signal}_q90", (signal, lambda s: s.quantile(0.90))),
            ]
        )

    features.extend(
        [
            ("vane_sin_mean", ("vane_sin", "mean")),
            ("vane_cos_mean", ("vane_cos", "mean")),
            ("n_samples", ("Power", "size")),
        ]
    )

    agg_spec = {name: pd.NamedAgg(column=column, aggfunc=func) for name, (column, func) in features}
    daily = data.groupby(["turbine_id", "date"], as_index=False).agg(**agg_spec)
    daily = daily.fillna(0.0)

    if has_label:
        labels = (
            df.groupby(["turbine_id", "date"], as_index=False)["yaw_misalignment_deg"]
            .median()
            .rename(columns={"yaw_misalignment_deg": "target"})
        )
        daily = daily.merge(labels, on=["turbine_id", "date"], how="inner")
    return daily


def make_design_matrix(frame, feature_columns):
    x = frame[feature_columns].to_numpy(dtype=float)
    x = np.column_stack([x, x**2])
    return x


def fit_ridge(x, y, alpha):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0.0] = 1.0
    xs = (x - mean) / std
    xb = np.column_stack([np.ones(len(xs)), xs])

    penalty = np.eye(xb.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(xb.T @ xb + penalty, xb.T @ y)
    return {"mean": mean, "std": std, "coef": coef}


def predict_ridge(model, x):
    xs = (x - model["mean"]) / model["std"]
    xb = np.column_stack([np.ones(len(xs)), xs])
    return xb @ model["coef"]


def metrics(y_true, y_pred):
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def load_training(zip_path, train_turbines):
    daily_frames = []
    for turbine in train_turbines:
        daily_frames.append(aggregate_daily(read_turbine(zip_path, turbine), has_label=True))
    return pd.concat(daily_frames, ignore_index=True)


def run_loto(train_daily, feature_columns, alpha, train_turbines):
    rows = []
    for holdout in train_turbines:
        train_part = train_daily[train_daily["turbine_id"] != holdout]
        valid_part = train_daily[train_daily["turbine_id"] == holdout]

        model = fit_ridge(
            make_design_matrix(train_part, feature_columns),
            train_part["target"].to_numpy(dtype=float),
            alpha,
        )
        pred = predict_ridge(model, make_design_matrix(valid_part, feature_columns))
        score = metrics(valid_part["target"].to_numpy(dtype=float), pred)
        score["holdout_turbine"] = holdout
        score["n_train_days"] = int(len(train_part))
        score["n_valid_days"] = int(len(valid_part))
        rows.append(score)
    return pd.DataFrame(rows)


def contiguous_clusters(values, tolerance=1.0):
    clusters = []
    current = 0
    previous = None
    for value in values:
        if previous is not None and abs(value - previous) > tolerance:
            current += 1
        clusters.append(current)
        previous = value
    return clusters


def write_submission(path, turbine_id, daily, predictions):
    output = daily[["date"]].copy()
    output["turbine_id"] = turbine_id
    output["yaw_misalignment_deg"] = predictions
    output["cluster"] = contiguous_clusters(predictions)
    output = output[SUBMISSION_COLUMNS].sort_values("date")
    output["yaw_misalignment_deg"] = output["yaw_misalignment_deg"].map(lambda x: f"{x:.6f}")
    output.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def main():
    parser = argparse.ArgumentParser(
        description="Leave-one-turbine-out ridge baseline for labelled train turbines."
    )
    parser.add_argument("--zip", default="turbines_data.zip", help="Path to turbines_data.zip.")
    parser.add_argument("--alpha", type=float, default=10.0, help="Ridge regularisation.")
    parser.add_argument("--cv-output", default="baseline_loto_scores.csv")
    parser.add_argument(
        "--train-turbines",
        nargs="+",
        help="Optional labelled turbines to use. By default, labelled turbines are auto-detected.",
    )
    parser.add_argument("--predict-turbine", help="Optional unlabelled turbine to predict.")
    parser.add_argument("--submission-output", help="Optional CSV path for predictions.")
    args = parser.parse_args()

    train_turbines = args.train_turbines or detect_labelled_turbines(args.zip)
    print("Labelled turbines:", ", ".join(train_turbines))

    train_daily = load_training(args.zip, train_turbines)
    feature_columns = [
        c for c in train_daily.columns if c not in {"turbine_id", "date", "target"}
    ]

    scores = run_loto(train_daily, feature_columns, args.alpha, train_turbines)
    scores.to_csv(args.cv_output, index=False)
    print(scores.to_string(index=False))
    print()
    print(
        "Average:",
        f"RMSE={scores['rmse'].mean():.3f}",
        f"MAE={scores['mae'].mean():.3f}",
        f"bias={scores['bias'].mean():.3f}",
    )

    if args.predict_turbine:
        if not args.submission_output:
            raise ValueError("--submission-output is required with --predict-turbine")

        model = fit_ridge(
            make_design_matrix(train_daily, feature_columns),
            train_daily["target"].to_numpy(dtype=float),
            args.alpha,
        )
        target_daily = aggregate_daily(read_turbine(args.zip, args.predict_turbine), has_label=False)
        pred = predict_ridge(model, make_design_matrix(target_daily, feature_columns))
        pred = np.clip(pred, -20.0, 20.0)
        write_submission(args.submission_output, args.predict_turbine, target_daily, pred)
        print(f"Wrote submission to {args.submission_output}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as exc:
        raise SystemExit(
            "This script needs pandas plus a parquet engine such as pyarrow. "
            "Run it in the same environment used for the notebooks, or install pyarrow."
        ) from exc
