# Static Yaw Misalignment — Energy Hackdays / WeDoWind 2026

Participant **33** · Portfolio / modelling record

This repository documents my work on the NADARA / WeDoWind static yaw misalignment challenge.

The modelling objective is to use a **two-year historical dataset to build a general model** that can transfer to previously unseen turbines.

## Challenge data setting

The challenge uses two complementary data settings:

- **T0 — unlabeled dataset:** two years of SCADA data without yaw-misalignment labels.
- **T3 — labeled dataset:** two years of SCADA data for three turbines with yaw-misalignment labels.

In this repository, T0 and T3 refer to the **data setting**, not to submission number.

For example:

- `Results_33_T3_0.csv` = Participant 33, **T3 setting**, validation submission **#0**
- `Results_33_T3_1.csv` = Participant 33, **T3 setting**, validation submission **#1**
- `Results_33_T3_final.csv` = Participant 33, **T3 setting**, final-round submission

The overall modelling idea is:

`two-year SCADA structure + three labeled turbines -> general yaw model -> unseen turbines`

---

## 1. Independent physics-informed general model

My original model was designed as a transferable turbine-level calibration model rather than a standard daily supervised regressor.

### Step 1 — Extract a physics feature from SCADA

For each turbine, use the two-year SCADA history to compute an apparent power-optimal vane angle.

The main derived quantity is:

`gamma = wrap(WindDir - NacDir)`

`gamma` is **not** treated as the true yaw-misalignment label because the nacelle / vane reference itself may be biased.

For each centered 21-day window:

- keep WindSpeed between 3 and 13
- require positive Power, RotSpeed, and GenSpeed
- remove the top 2% of Power
- remove the top 5% of PitchAngle
- normalize Power within 1.0-unit WindSpeed bins
- restrict `gamma` to ±30°
- use 1° angle bins
- require at least 20 samples per angle bin
- require at least 8 valid angle bins
- select the observed median `gamma` in the bin with the highest median normalized Power

This gives a rolling apparent optimum:

`theta_t`

The long-term turbine-level physics feature is:

`theta_star = median(theta_t over valid windows)`

This feature extraction can be applied to **unlabeled SCADA**, so it is naturally compatible with the T0 setting.

### Step 2 — Use the three T3 labeled turbines to learn a shared calibration

The three labeled turbines are:

- `PPP_WTG12`
- `PPP_WTG13`
- `PPP_WTG14`

For each turbine, the two-year yaw labels are summarized to a long-term target level and linked to the SCADA-derived `theta_star`.

The fitted relationship is:

`yaw_level = C - theta_star`

The three calibration values were:

| Turbine | `theta_star` | Mean yaw target | Implied `C` |
|---|---:|---:|---:|
| `PPP_WTG12` | -3.441° | -2.428° | -5.868° |
| `PPP_WTG13` | +0.357° | -6.613° | -6.256° |
| `PPP_WTG14` | -4.351° | -1.762° | -6.114° |

The final shared calibration constant was:

`C_FINAL = -6.0793162066°`

The resulting general model is:

`predicted_yaw = C_FINAL - theta_star`

The key point is that the model is **not fitted separately for each target turbine**.  
The same shared calibration is transferred to unseen turbines after extracting their own `theta_star` from SCADA.

### Strict turbine-level validation

Validation was performed with leave-one-turbine-out folds.

For each fold:

- two labeled turbines were used to estimate the calibration
- the third turbine's labels were hidden
- its own SCADA was used to compute `theta_star`
- the shared calibration was then transferred to that held-out turbine

Results:

| Holdout | Prediction | MAE |
|---|---:|---:|
| `PPP_WTG12` | -2.744° | 0.354° |
| `PPP_WTG13` | -6.348° | 0.997° |
| `PPP_WTG14` | -1.711° | 0.207° |
| **Macro** | — | **0.519°** |

This looked strong internally, but the hidden-target leaderboard later showed that the transfer estimate was too optimistic.

### Hidden-target predictions

Applying the same general model to unseen targets gave:

- `PPP_WTG17`: `-3.596°`
- `SSS_WTG06`: `-4.568°`

Because the original model compressed the full two-year history into one long-term `theta_star`, each target received one constant yaw level for all 731 days.

Leaderboard result:

| Target | RMSE | MAE |
|---|---:|---:|
| Public validation — `PPP_WTG17` | 5.28° | 5.25° |
| Final — `SSS_WTG06` | 3.46° | 3.39° |

This showed that the long-term absolute calibration alone was not sufficient when yaw state changed over time.

---

## 2. Team hybrid experiment with Daniel

After the standalone model underperformed on the public validation target, I compared it with an independent state-aware T3 model developed by my teammate **Daniel**.

Daniel's repository:

[Daniel — NADARA Yaw Misalignment Challenge](https://github.com/d-er/NADARA-Yaw-Misalignment-Challenge)

Daniel's model supplied:

- temporal state boundaries
- cluster assignments
- relative yaw differences between states

I did **not** independently derive those state amplitudes in this experiment.

I re-centered Daniel's predicted state trajectory so that its long-term mean matched my independently estimated physics level.

The hybrid was:

`hybrid(t) = my_physics_level + [Daniel_prediction(t) - mean(Daniel_prediction)]`

For `PPP_WTG17`:

- my long-term physics level: `-3.596°`
- Daniel's two state levels: `-7.235°` and `+0.258°`
- Daniel's temporal mean: approximately `-3.535°`
- re-centering shift: approximately `-0.061°`

The resulting hybrid state levels were approximately:

- cluster 0: `-7.296°`
- cluster 1: `+0.197°`

### Public validation result

`Results_33_T3_1.csv` scored:

| RMSE | MAE | Cluster ARI |
|---:|---:|---:|
| **1.58°** | **1.49°** | **1.00** |

Compared with the independent constant physics model:

| Model | RMSE | MAE |
|---|---:|---:|
| Independent constant physics model | 5.28° | 5.25° |
| Team hybrid | **1.58°** | **1.49°** |

This experiment demonstrated that temporal state structure was important.

However, this result is **not a standalone result of my physics model**.

Daniel's prediction supplied both:

- when the state changed
- how large the relative state-to-state yaw differences were

My contribution in this hybrid was the long-term absolute re-centering using my physics calibration.

The experiment is kept in this repository because it is part of the modelling history and directly motivated the next independent model.

---

## 3. New independent state-aware model

The next model removes the dependence on Daniel's predicted yaw values, cluster labels, and change-point dates.

The new pipeline is:

`SCADA -> apparent-optimum time series -> independent change-point detection -> regime-level physics optimum -> yaw calibration`

The goal is to estimate both:

- **when** yaw state changes
- **how large** each state-level yaw offset is

using only:

- the challenge SCADA history
- the three T3 labeled turbines

The current design is:

- derive a shorter-window physics signal from SCADA for temporal sensitivity
- detect piecewise-constant regimes directly from that signal
- pool SCADA inside each detected regime
- estimate one regime-level apparent power optimum
- learn state-level yaw calibration from the labeled turbines
- evaluate the full pipeline with strict turbine-level validation before making another leaderboard claim

The development notebook is:

[`YawMisalignment_Independent_State_Physics.ipynb`](YawMisalignment_Independent_State_Physics.ipynb)

No leaderboard result is claimed for this new independent model yet.

---

## Model progression

The project evolved through three distinct stages:

| Stage | Absolute yaw level | Temporal state boundaries | Relative state amplitudes |
|---|---|---|---|
| Independent physics model | Mine | None | None |
| Team hybrid with Daniel | Mine | Daniel | Daniel |
| New independent state model | Mine | Mine | Mine |

This distinction is important for both modelling interpretation and attribution.

---

## Main modelling lessons

This project exposed several useful lessons:

- a strong internal leave-one-turbine-out score can still underestimate cross-turbine domain shift
- extracting a transferable physics feature is different from learning temporal state dynamics
- long-term absolute yaw calibration and within-turbine state detection are separate modelling problems
- hidden-target validation can reveal which modelling assumption failed
- model combination can be useful diagnostically, but contribution boundaries should be stated explicitly
- the next model should only claim capabilities that are independently reproduced from SCADA

---

## Repository structure

```text
.
├── README.md
├── FINAL_METHOD_SUMMARY.md
├── YawMisalignment_Final_Summary.ipynb
├── YawMisalignment_Independent_State_Physics.ipynb
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

- `Results_33_T3_0.csv` — independent constant physics validation submission
- `Results_33_T3_1.csv` — team hybrid validation submission using Daniel's state-aware prediction re-centered to my physics level
- `Results_33_T3_final.csv` — final-round artifact currently stored in this research repository

The organizer submission repository has its own immutable-submission workflow, so organizer-side submission history and research-repository artifacts may not always be identical.

---

## Data and reproducibility

Challenge data are intentionally not included in this public repository.

The repository does not publish:

- raw SCADA
- challenge archives
- parquet challenge files
- local derived caches that may inherit challenge-data restrictions

Because of that, a clean clone cannot reproduce all numerical experiments from raw data.

Executed notebooks are included so the modelling history and recorded outputs can still be inspected.

---

## Environment

Python 3.11 is recommended:

```bash
conda create -n yaw-baseline python=3.11 -y
conda activate yaw-baseline
pip install -r requirements.txt
```

Run tests with:

```bash
python -m unittest discover -s tests
```

Data-dependent tests are skipped when the local challenge cache is absent.

---

Repository: `HuijiaoLuo/2026EnergyHacks_WeDoWind`
