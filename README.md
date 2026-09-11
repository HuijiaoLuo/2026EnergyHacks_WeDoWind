# Static Yaw Misalignment — Energy Hackdays / WeDoWind 2026

Participant **33** · T3

This repository documents my work on the NADARA / WeDoWind static yaw misalignment challenge.

The project evolved through three stages:

1. an independent physics-informed long-term yaw model,
2. a team hybrid experiment that combined my long-term calibration with a teammate's state-aware prediction,
3. a new independent state-aware model currently being developed from SCADA only.

The distinction between these stages matters, especially for attribution and interpretation.

## 1. Independent physics model

My original model estimates a long-term yaw level from SCADA using the apparent power-optimal vane angle.

The core relationship is:

`yaw_level = C_FINAL - theta_star`

where:

- `gamma = wrap(WindDir - NacDir)`
- SCADA are filtered to normal operating conditions
- a 21-day centered rolling window is used
- Power is normalized within 1.0-unit WindSpeed bins
- only `gamma` within ±30° is used
- the observed median angle of the highest normalized-power bin is taken as the apparent optimum
- `theta_star` is the long-term median of valid rolling-window optima
- `C_FINAL` is calibrated from the three labelled development turbines

The frozen calibration constant was:

`C_FINAL = -6.0793162066°`

The resulting target-level predictions were:

- `PPP_WTG17`: `-3.596°`
- `SSS_WTG06`: `-4.568°`

This model predicts one constant yaw level for the full 731-day target period.

### Internal validation

Strict leave-one-turbine-out validation across the three labelled turbines gave:

| Holdout | Prediction | MAE |
|---|---:|---:|
| `PPP_WTG12` | -2.744° | 0.354° |
| `PPP_WTG13` | -6.348° | 0.997° |
| `PPP_WTG14` | -1.711° | 0.207° |
| **Macro** | — | **0.519°** |

This was strong internal evidence, but it turned out to be too optimistic as an estimate of hidden-target performance.

### Hidden leaderboard result

The standalone constant model scored:

| Round | RMSE | MAE |
|---|---:|---:|
| Public validation (`PPP_WTG17`) | 5.28° | 5.25° |
| Final (`SSS_WTG06`) | 3.46° | 3.39° |

This showed that a single long-term level was not enough when the target turbine changed yaw state over time.

## 2. Team hybrid experiment

After the standalone model underperformed on the public validation target, I compared it with a teammate's independent state-aware T3 prediction.

The teammate model provided:

- the temporal state boundaries,
- the cluster assignments,
- and the relative yaw differences between states.

I did **not** independently derive those state amplitudes in this experiment.

I then re-centered the teammate trajectory so that its long-term mean matched my independently estimated physics level.

In plain form:

`hybrid(t) = my_physics_level + [teammate_prediction(t) - mean(teammate_prediction)]`

For `PPP_WTG17`:

- my long-term physics level: `-3.596°`
- teammate state levels: `-7.235°` and `+0.258°`
- teammate temporal mean: approximately `-3.535°`
- re-centering shift: approximately `-0.061°`

The resulting hybrid state levels were approximately:

- cluster 0: `-7.296°`
- cluster 1: `+0.197°`

### Public validation result

This team hybrid submission (`T3 Sub #1`) scored:

| RMSE | MAE | Cluster ARI |
|---:|---:|---:|
| **1.58°** | **1.49°** | **1.00** |

Compared with the standalone constant model:

| Model | RMSE | MAE |
|---|---:|---:|
| Independent constant physics model | 5.28° | 5.25° |
| Team hybrid | **1.58°** | **1.49°** |

The hybrid result is important because it demonstrated that temporal regime structure matters.

However, it should not be interpreted as a standalone result of my physics model. The teammate prediction supplied both the change-point structure and the relative state-to-state yaw amplitudes. My contribution in this experiment was the long-term absolute re-centering.

This experiment is kept in the repository because it is an important part of the modelling history and because it motivated the next independent model.

## 3. New independent state-aware model

The next model removes the dependence on teammate yaw predictions.

The target pipeline is:

`SCADA -> apparent-optimum time series -> independent change-point detection -> regime-level physics optimum -> yaw calibration`

The new model is designed to estimate both:

- when the yaw state changes,
- and how large each state-level yaw offset is,

using only the challenge SCADA plus the three labelled training turbines.

The current modelling plan is:

- use a shorter rolling B0 signal for change-point detection,
- detect piecewise-constant regimes directly from SCADA,
- pool SCADA within each detected regime,
- estimate one power-optimal vane angle per regime,
- calibrate state-level yaw amplitude using labelled turbines,
- evaluate the complete pipeline with strict turbine-level validation before submitting another leaderboard entry.

This independent state-aware model is still under development. No leaderboard result is claimed for it yet.

## Why this progression matters

This project became a useful example of model-selection risk.

The first model looked excellent under internal leave-one-turbine-out validation but failed to transfer to the hidden target.

The team hybrid then showed that the missing information was largely temporal state structure.

The current goal is to reproduce that capability independently from SCADA rather than relying on another model's predicted state trajectory.

For a portfolio, the main technical lessons are:

- physics-based feature construction can be more transferable than direct SCADA regression,
- strong internal cross-validation can still be misleading with only three labelled turbines,
- long-term calibration and temporal state detection are separate modelling problems,
- external validation is valuable for identifying which part of the modelling assumption failed,
- attribution matters when combining independently developed models.

## Start here

The executed summary notebook is:

[`YawMisalignment_Final_Summary.ipynb`](YawMisalignment_Final_Summary.ipynb)

The new independent state-aware model is developed separately from the historical summary so that the original evidence and the new modelling work remain distinguishable.

## Repository structure

```text
.
├── README.md
├── FINAL_METHOD_SUMMARY.md
├── YawMisalignment_Final_Summary.ipynb
├── requirements.txt
├── slides/
│   └── final.pptx
├── src/
│   ├── baseline_loto_ridge.py
│   ├── baseline_openoa_power_vane_updated.py
│   └── make_final_submission_vane_level.py
├── submissions/
│   ├── Results_33_T3_0.csv
│   ├── Results_33_T3_1.csv
│   └── Results_33_T3_final.csv
└── tests/
    └── test_final_method.py
```

### Submission history

- `Results_33_T3_0.csv` — standalone constant physics validation submission.
- `Results_33_T3_1.csv` — team hybrid validation submission using teammate state predictions re-centered to my physics level.
- `Results_33_T3_final.csv` — final-round artifact currently stored in this research repository.

The organizer submission repository has its own immutable-submission workflow, so repository artifacts and organizer-side submission history may not always be identical.

## Data and reproducibility

Challenge data are intentionally not included in this public repository.

The repository does not publish:

- raw SCADA,
- parquet challenge files,
- challenge archives,
- or local derived caches that may inherit challenge-data restrictions.

Because of that, a clean clone cannot reproduce all numerical experiments from raw data.

The executed notebook is included so the modelling history and recorded outputs can still be inspected.

## Tests

Run:

```bash
python -m unittest discover -s tests
```

Data-dependent tests are skipped when the local challenge cache is absent.

## Environment

A Python 3.11 environment is recommended:

```bash
conda create -n yaw-baseline python=3.11 -y
conda activate yaw-baseline
pip install -r requirements.txt
```

## Important interpretation

`WindDir - NacDir` is not treated as the true yaw label.

The nacelle/vane reference may itself be biased, so the physics model searches for the apparent angle associated with maximum normalized power and calibrates that relationship using labelled turbines.

---

Repository: `HuijiaoLuo/2026EnergyHacks_WeDoWind`
