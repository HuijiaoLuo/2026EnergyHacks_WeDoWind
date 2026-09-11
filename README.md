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

## 3. Current independent model — V7.5 constant prior + robust relative-heading states

The current independent model removes the dependence on Daniel's predicted yaw values, cluster labels, and change-point dates.

The main release notebook is:

[`YawMisalignment_Independent_Model_Release.ipynb`](YawMisalignment_Independent_Model_Release.ipynb)

The model separates the problem into two parts:

1. a low-variance **absolute yaw prior** from the long-term aerodynamic reference;
2. a sparse **label-free temporal correction** from robust relative-heading states.

The model is:

`B0_i = C - theta_star_i`

followed by

`yaw_hat_i(t) = B0_i - beta * relative_state_correction_i(t)`

The constant prior is the default. The model moves away from it only when SCADA provides sufficiently strong evidence of a persistent state change.

### 3.1 Absolute prior

The same rolling power-vs-vane argmax estimator is retained.

For turbine `i`:

`theta_star_i = median(valid rolling theta_hat_argmax_i(t))`

The labeled training turbines estimate one shared calibration:

`C = mean_i(mean_yaw_i + theta_star_i)`

so the turbine-level default is:

`B0_i = C - theta_star_i`

In the current V7.5 reference run, the final calibration is approximately:

- `C = -6.085°`
- `beta = 1.000`

The physical slope prior is retained rather than fitting a more complex amplitude mapping from very few labeled state transitions.

### 3.2 Label-free relative-heading state detector

The dynamic channel uses only SCADA and same-site turbine context.

For target turbine `i` and neighbour `j`, the pipeline:

1. forms circular nacelle-heading differences;
2. removes pair- and wind-sector-specific baselines;
3. estimates pair reliability from coverage, residual MAD, and distance;
4. robustly aggregates usable pair residuals into a daily target-relative heading signal;
5. constructs a neighbour-only background residual to identify common-mode motion.

Two complementary change detectors operate on the cleaned relative-heading signal:

- a persistent rolling before/after detector;
- a conservative L2 state segmentation for long two-sided regimes.

A candidate state change is rejected when it is better explained by:

- sensor / encoder-like jumps
- the sensor-shadow exclusion window
- local background or common-mode motion
- insufficient pair agreement
- same-site synchronous events
- weak event confidence

The sensor-shadow veto is applied **across detector sources**, preventing a PELT boundary from reintroducing an encoder event already identified by the rolling detector.

Only surviving high-confidence events can create a dynamic correction.

### 3.3 Conservative state correction

The dynamic channel is intentionally sparse.

The model does not integrate every detected step indefinitely. Instead, persistent states are anchored to their relative-heading levels and corrections are shrunk according to label-free event confidence.

Conceptually:

`prediction(t) = constant_prior + sparse_state_correction(t)`

This lets the model behave differently depending on the turbine:

- if no trustworthy temporal state evidence exists, it collapses to the constant prior;
- if persistent state evidence exists, it adds a bounded state-aware correction.

### Strict turbine-level LOTO validation

The current V7.5 reference run gives:

| Holdout | V7.5 MAE | Constant MAE | V7.5 RMSE | Constant RMSE |
|---|---:|---:|---:|---:|
| `PPP_WTG12` | **0.357°** | 0.357° | **0.608°** | 0.608° |
| `PPP_WTG13` | **0.608°** | 1.004° | **0.979°** | 1.387° |
| `PPP_WTG14` | **0.206°** | 0.206° | **0.262°** | 0.262° |
| **Macro** | **0.390°** | **0.522°** | **0.616°** | **0.752°** |

Relative to the constant prior, this is approximately:

- **25% lower macro MAE**
- **18% lower macro RMSE**

The structural behaviour is more important than the aggregate score:

- `PPP_WTG12`: no accepted dynamic boundary → constant prior
- `PPP_WTG13`: two accepted persistent boundaries → dynamic correction
- `PPP_WTG14`: no accepted dynamic boundary → constant prior

In the reference run, the accepted `PPP_WTG13` boundaries were:

- `2023-02-19`
- `2023-07-27`

This is the intended behaviour of the model: **know when to move, and know when not to move**.

The release notebook reports MAE/RMSE for model comparison. Older micro-step boundary-recall diagnostics are not used for model selection because they do not represent the persistent macro-state objective of the current detector.

### Unlabeled target diagnostics

After fitting the final calibration on all three labeled turbines, the same model can be applied to the unlabeled targets.

These are **deployment diagnostics, not validation scores** because the target labels are unavailable locally.

Reference V7.5 diagnostics:

| Target | Accepted boundaries | Correction range | Prediction range |
|---|---|---:|---:|
| `PPP_WTG17` | 2023-06-25, 2023-10-15, 2024-01-07 | 5.258° | -6.680° to -1.422° |
| `SSS_WTG06` | 2023-05-28, 2023-09-24, 2024-10-13 | 4.434° | -8.162° to -3.729° |

No leaderboard result is claimed for V7.5 in this repository at this stage.

The larger target correction ranges relative to the labeled training turbines are treated as a generalisation risk rather than as evidence of better performance.

---

## Model progression

The project evolved through three main modelling stages:

| Stage | Absolute yaw level | Temporal state boundaries | Relative state amplitudes |
|---|---|---|---|
| Original independent constant model | Independent B0 calibration | None | None |
| Team hybrid with Daniel | Independent B0 calibration | Daniel | Daniel |
| Current independent V7.5 | Independent B0 calibration | Label-free SCADA detector | Relative-heading state correction with physical `beta` prior |

The hybrid is a team result. The current V7.5 model does not consume Daniel's predictions, state labels, or change-point dates.

---

## Main modelling lessons

This project exposed several useful lessons:

- a strong internal leave-one-turbine-out score can still underestimate hidden-target domain shift
- absolute yaw calibration and temporal state detection are distinct problems
- the long-term B0 estimator is useful as a low-variance **prior**, even when it is insufficient as a complete model
- neighbours are most useful as a robust local reference and stability context, not as a direct target-yaw regressor
- persistent relative-heading structure can recover useful temporal information without target labels
- sensor / encoder re-references can mimic yaw state changes and must be explicitly vetoed
- a state detector should be allowed to return **no correction** when the evidence is weak
- additional complexity should only be introduced when a simpler observable fails for a demonstrated reason
- unlabeled target state magnitude remains a key uncertainty, so robustness analysis is more valuable than further optimisation of the three training turbines
- team-model improvements are useful diagnostically, but contribution boundaries should remain explicit

---

## Repository structure

```text
.
├── README.md
├── FINAL_METHOD_SUMMARY.md
├── YawMisalignment_Independent_Model_Release.ipynb
├── YawMisalignment_StateAware_Update_executed.ipynb
├── requirements.txt
├── slides/
│   └── final.pptx
├├── src/
│   ├── baseline_loto_ridge.py
│   ├── baseline_openoa_power_vane_updated.py
│   ├── fleet_context_v6.py
│   ├── yaw_relative_state_v6.py
│   ├── yaw_model_v6.py
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
