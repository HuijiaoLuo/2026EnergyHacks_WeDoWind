# Static Yaw Misalignment — Physics-Informed Long-Term Vane Calibration

[![CI](https://github.com/HuijiaoLuo/2026EnergyHacks_WeDoWind/actions/workflows/ci.yml/badge.svg)](https://github.com/HuijiaoLuo/2026EnergyHacks_WeDoWind/actions/workflows/ci.yml)

This repository summarizes a physics-informed approach for estimating **static yaw misalignment** from wind-turbine SCADA.

> **Start here:** [`YawMisalignment_Final_Summary.ipynb`](YawMisalignment_Final_Summary.ipynb)

The strongest validated method in this development study is:

$$
\hat{y} = C - \theta^\star
$$

where `theta_star` is the long-term median of rolling power-optimal vane angles from the frozen OpenOA-style B0 estimator. The final shared calibration is:

```text
C_FINAL = -6.0793162066°
```

Raw `WindDir - NacDir` is not treated as true yaw because vane/controller/reference bias can persist.

## Results

| Strict LOTO holdout | MAE (deg) |
|---|---:|
| PPP_WTG12 | 0.354 |
| PPP_WTG13 | 0.997 |
| PPP_WTG14 | 0.207 |
| **Macro** | **0.519** |

The 28-day block-bootstrap macro median is **0.678°**, with a central robustness range of **[0.448°, 1.470°]**. These are sampling-robustness intervals, not formal confidence intervals.

| Frozen target | Prediction | Bootstrap std | 95% robustness range |
|---|---:|---:|---|
| PPP_WTG17 | **-3.596°** | 0.375° | [-4.527°, -2.996°] |
| SSS_WTG06 | **-4.568°** | 1.365° | [-6.644°, -2.280°] |

Only **three labelled turbines** from one site were available. Calibration and sign selection exclude each outer holdout's labels, but the final formula was discovered through development on these same turbines. The reported LOTO result is therefore not independent new-site validation, and transfer to SSS is substantially less certain.

## Method in brief

For each turbine, the frozen B0 estimator uses centered 21-day windows. Within each window, power is normalized by the median power in 1.0-unit wind-speed bins. The normalized power-vs-vane curve is evaluated over

```text
gamma = wrap(WindDir - NacDir)
```

within ±30°, using 1° angle bins with minimum sample and coverage requirements. The vane angle of the maximum valid median normalized-power bin is the daily apparent optimum. Its long-term median is `theta_star`.

The final model then applies the shared cross-turbine calibration:

```text
yaw_hat = C_FINAL - theta_star
```

Geometry, context normalization, daily dynamics, residual ML, and a generic ML power surface were investigated but did not improve strict validation. Geometry is retained only as a domain-shift/risk diagnostic.

## Inspect the final analysis

The executed notebook is the main entry point:

[`YawMisalignment_Final_Summary.ipynb`](YawMisalignment_Final_Summary.ipynb)

For exact methodology, historical comparisons, provenance, limitations, and robustness details, see:

[`FINAL_METHOD_SUMMARY.md`](FINAL_METHOD_SUMMARY.md)

The notebook is committed with its executed outputs so the final analysis can be inspected directly on GitHub.

## Reproduction note

The public repository intentionally excludes the project data directory, raw SCADA, parquet files, and data ZIPs.

As a result, the committed notebook can be **viewed**, but a clean clone cannot fully re-execute the numerical analysis without separately obtained challenge data or the local derived cache used during development.

Install the code dependencies with:

```bash
python -m pip install -r requirements.txt
```

If you have the required local data, the repository code can be used to reproduce the frozen B0 estimates and final calibration. See [`FINAL_METHOD_SUMMARY.md`](FINAL_METHOD_SUMMARY.md) for the exact inputs and provenance.

## Repository layout

```text
README.md
FINAL_METHOD_SUMMARY.md
YawMisalignment_Final_Summary.ipynb
requirements.txt
.gitignore

src/
  baseline_loto_ridge.py
  baseline_openoa_power_vane_updated.py
  make_final_submission_vane_level.py

tests/
  test_final_method.py

slides/
  final.pptx

submissions/
  Results_30_T3_0.csv
  Results_30_T3_final.csv
```

The following are intentionally **not included** in the public repository:

```text
data/
turbines_data.zip
*.parquet
archived experiments
temporary outputs
obsolete notebooks
```

## Final submissions

The frozen exported predictions are:

```text
PPP_WTG17  -> -3.596°
SSS_WTG06  -> -4.568°
```

Challenge-format files are stored under `submissions/`:

```text
Results_30_T3_0.csv
Results_30_T3_final.csv
```

Both use the required schema:

```text
turbine_id,date,yaw_misalignment_deg,cluster
```

with 731 daily rows.

## Limitations

This is a compact hackathon solution rather than a universally validated yaw-calibration model.

The calibration is supported by only three labelled turbines from one site, and the model was selected through repeated development on those same turbines. The shared calibration constant is not fully stable across years, constant predictions cannot follow actual yaw interventions, and cross-site transfer to SSS remains the main unresolved uncertainty.
