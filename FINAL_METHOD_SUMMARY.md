# Final Method Summary — NADARA Static Yaw Misalignment

**Energydata Hackdays / WeDoWind & NADARA 2026 · Participant 33**

This document contains the detailed modelling record behind the recruiter-facing [`README.md`](README.md).

It preserves the full methodological progression, calibration details, strict leave-one-turbine-out validation, model-selection audit, public-target diagnostics, ownership boundaries, and reproducibility notes.

The retained model is named **Physics-Anchored Relative-State (PARS)**. It combines a turbine-specific physics prior for the absolute level with reliability-weighted cross-turbine relative-heading states for persistent changes.

---

## 1. Challenge setting

The modelling objective is to use a **two-year historical SCADA dataset to build a general yaw-misalignment model that transfers to previously unseen turbines**.

Two challenge data settings are used:

- **T0 — unlabeled dataset:** two years of SCADA data without yaw-misalignment labels.
- **T3 — labeled dataset:** two years of SCADA data for three turbines with yaw-misalignment labels.

In this repository, `T0` and `T3` refer to the **data setting**, not to submission number.

The three labeled turbines are:

- `PPP_WTG12`
- `PPP_WTG13`
- `PPP_WTG14`

The main unseen targets discussed in this repository are:

- `PPP_WTG17` — public validation target
- `SSS_WTG06` — final target / deployment target in the challenge workflow

The overall modelling problem is:

```text
two-year SCADA structure
        +
three labeled turbines
        ↓
transferable calibration + label-free state estimator
        ↓
previously unseen turbine
```

---

# 2. Modelling principles

The final model is built around five principles.

### 2.1 Separate absolute calibration from temporal state estimation

A single long-term yaw value is too restrictive when the turbine changes state over time.

The problem is therefore decomposed into:

```text
absolute yaw prior
        +
persistent relative state correction
        ↓
daily yaw trajectory
```

### 2.2 Use physical observables before adding model complexity

The absolute prior comes from a power-vs-vane aerodynamic reference extracted from SCADA.

The dynamic channel comes from relative nacelle-heading structure across turbines.

### 2.3 Treat the wind farm as a local measurement field

Neighbouring turbines are not used as direct target-yaw regressors.

Instead, pairwise measurements are used to construct a robust local reference field, with pair reliability determined by data quality and spatial context.

### 2.4 Restrict model selection to development turbines

Sensitivity analysis, detector design, pair weighting, and amplitude blending are developed only using the labeled training turbines under strict turbine-level leave-one-out validation.

### 2.5 Keep public validation blind

Public-target metrics are used only for final evaluation and post-hoc error diagnosis.

They are not used to tune state boundaries, amplitude parameters, or absolute intercepts.

---

# 3. Stage I — Independent constant physics model

The first model was designed as a transferable turbine-level calibration model rather than a daily supervised regressor.

## 3.1 Physics feature from SCADA

For each turbine, the two-year SCADA history is used to estimate an apparent power-optimal vane angle.

The main derived variable is:

```text
gamma = wrap(WindDir - NacDir)
```

`gamma` is **not** assumed to be the true yaw-misalignment label because the nacelle / vane reference itself may be biased.

For each centered 21-day window:

- keep `WindSpeed` between 3 and 13;
- require positive `Power`, `RotSpeed`, and `GenSpeed`;
- remove the top 2% of `Power`;
- remove the top 5% of `PitchAngle`;
- normalize `Power` within 1.0-unit `WindSpeed` bins;
- restrict `gamma` to ±30°;
- use 1° angle bins;
- require at least 20 samples per angle bin;
- require at least 8 valid angle bins;
- select the observed median `gamma` in the bin with the highest median normalized `Power`.

This gives a rolling apparent optimum:

```text
theta_t
```

The long-term turbine feature is:

```text
theta_star = median(theta_t over valid windows)
```

Because this feature uses only SCADA, it can also be computed for unlabeled turbines.

---

## 3.2 Shared absolute calibration

For each labeled turbine, the two-year yaw labels are summarized to a long-term target level and linked to the SCADA-derived `theta_star`.

The fitted physical relationship is:

```text
yaw_level = C - theta_star
```

The original calibration values were:

| Turbine | `theta_star` | Mean yaw target | Implied `C` |
|---|---:|---:|---:|
| `PPP_WTG12` | -3.441° | -2.428° | -5.868° |
| `PPP_WTG13` | +0.357° | -6.613° | -6.256° |
| `PPP_WTG14` | -4.351° | -1.762° | -6.114° |

The original shared calibration constant was:

```text
C_FINAL = -6.0793162066°
```


This exact value belongs to the historical Stage-I constant model. It is not the later current-reference estimate reported in Section 6.
and the resulting model was:

```text
predicted_yaw = C_FINAL - theta_star
```

The model is not fitted separately for each target turbine.

The same shared calibration is transferred to unseen turbines after extracting their own `theta_star`.

---

## 3.3 Strict turbine-level validation of the constant model

For each leave-one-turbine-out fold:

1. two labeled turbines estimate the shared calibration;
2. the third turbine's labels are hidden;
3. the held-out turbine's SCADA is used to estimate `theta_star`;
4. the shared calibration is transferred to that turbine.

Results:

| Holdout | Prediction | MAE |
|---|---:|---:|
| `PPP_WTG12` | -2.744° | 0.354° |
| `PPP_WTG13` | -6.348° | 0.997° |
| `PPP_WTG14` | -1.711° | 0.207° |
| **Macro** | — | **0.519°** |

The internal result appeared strong, but hidden-target evaluation exposed a major limitation.

---

## 3.4 Hidden-target behavior of the constant model

Applying the constant model to unseen targets gave:

- `PPP_WTG17`: `-3.596°`
- `SSS_WTG06`: `-4.568°`

Because the full two-year history was compressed into one `theta_star`, each target received one constant yaw level for all 731 days.

Leaderboard result:

| Target | RMSE | MAE |
|---|---:|---:|
| Public validation — `PPP_WTG17` | 5.28° | 5.25° |
| Final — `SSS_WTG06` | 3.46° | 3.39° |

This established that the long-term absolute calibration was useful as a **prior**, but insufficient as a complete daily yaw model when persistent yaw states change over time.

---

# 4. Stage II — Team-derived hybrid as a diagnostic experiment

After the constant model underperformed on the public target, it was compared with an independent state-aware T3 model developed by teammate **Daniel**. The related reference repository is [Daniel's NADARA Yaw Misalignment Challenge repository](https://github.com/d-er/NADARA-Yaw-Misalignment-Challenge.git).

Daniel's model supplied:

- temporal state boundaries;
- relative state amplitudes.

My contribution in this hybrid experiment was only to re-center that trajectory using my independently estimated long-term physics level.

The hybrid reached:

```text
PPP_WTG17
RMSE = 1.58°
MAE  = 1.49°
ARI  = 1.00
```

This experiment demonstrated that persistent temporal state structure mattered.

However, it is **not** treated as a standalone result of my independent model.

Its role was diagnostic: it motivated the development of a fully independent relative-heading state estimator.

PARS does **not** consume Daniel's:

- predictions;
- state labels;
- state amplitudes;
- change-point dates.

---

# 5. Stage III — Physics-Anchored Relative-State (PARS) model

PARS removes dependence on teammate state predictions and separates the problem into:

1. a low-variance **absolute yaw prior**;
2. a sparse, label-free **relative-heading state correction**.

The main release notebook is:

[`YawMisalignment_Independent_Model_Release.ipynb`](YawMisalignment_Independent_Model_Release.ipynb)

The model is:

$$
\hat y_i(d)=B_{0,i}-c_{i,\lambda}^{\mathrm{soft}}(d)
$$

with

$$
B_{0,i}=C-\theta_i^*
$$

and

$$
c_{i,\lambda}(d)=c_{i,\mathrm{stable}}(d)+\lambda\left[c_{i,\mathrm{full}}(d)-c_{i,\mathrm{stable}}(d)\right].
$$

The retained candidate uses:

```text
lambda = 0.75
beta   = 1.0
soft pair spread scale = 2.5°
```

The constant prior is the default.

The prediction moves away from that prior only when SCADA provides sufficiently strong evidence of a persistent state change.

---

# 6. Absolute prior in PARS

The rolling power-vs-vane argmax estimator is retained.

For turbine `i`:

```text
theta_star_i = median(valid rolling theta_hat_argmax_i(t))
```

The labeled training turbines estimate one shared calibration:

```text
C = mean_i(mean_yaw_i + theta_star_i)
```

and therefore:

```text
B0_i = C - theta_star_i
```

In the current reference run:

```text
C ≈ -6.085°
beta = 1.000
```

This is a later current-reference estimate and should not be conflated with the historical `C_FINAL` above.

The physical slope prior is fixed rather than fitting a more complex amplitude relationship from a very small number of labeled state transitions.

This keeps the absolute channel low variance.

---

# 7. Farm-level relative-heading state estimator

The dynamic channel uses only SCADA and same-site turbine context.

For target turbine `i` and neighbour `j`, the pipeline:

1. forms circular nacelle-heading differences;
2. removes pair-specific and wind-sector-specific baselines;
3. estimates pair reliability from:
   - coverage;
   - residual MAD;
   - distance;
4. robustly aggregates usable pair residuals into a daily target-relative heading signal;
5. constructs a neighbour-only background residual to detect common-mode motion.

This implements a **farm-level background-field idea**:

```text
target-relative pair evidence
        -
local common-mode field
        ↓
target-specific relative heading state
```

Neighbouring turbines are therefore treated as distributed local references rather than as direct predictors of target yaw.

The neighbour contribution is reliability-weighted rather than fixed.

---

# 8. Persistent-state detection

Two complementary state detectors operate on the cleaned relative-heading signal:

- a persistent rolling before/after detector;
- a conservative L2 segmentation for long two-sided regimes.

Candidate state changes are rejected when they are better explained by:

- sensor / encoder-like jumps;
- sensor-shadow exclusion windows;
- local background or common-mode motion;
- insufficient pair agreement;
- same-site synchronous events;
- weak event confidence.

The sensor-shadow veto is applied across detector sources so that a segmentation method cannot reintroduce an encoder-like event already rejected by another detector.

Only surviving high-confidence events can create a dynamic correction.

The detector is explicitly allowed to return:

```text
no dynamic correction
```

when the evidence is weak.

---

# 9. State construction and amplitude correction

The dynamic model is intentionally sparse.

It does not integrate every detected local increment indefinitely.

Instead, persistent states are anchored to relative-heading levels.

Two state-amplitude constructions are used:

### Stable correction

The stable correction uses label-free event-confidence shrinkage.

This reduces sensitivity to noisy state estimates but can suppress the true state amplitude.

### Full correction

The full correction removes that amplitude shrinkage while keeping the **same accepted state boundaries**.

### Partial-amplitude blend

The retained candidate interpolates the two:

$$
c_{i,\lambda}=c_{i,\mathrm{stable}}+\lambda(c_{i,\mathrm{full}}-c_{i,\mathrm{stable}})
$$

with:

```text
lambda = 0.75
```

State-level soft weighting reduces the influence of days with large cross-pair disagreement.

It does not:

- create new boundaries;
- move accepted boundaries;
- change the absolute `B0` anchor.

---

# 10. Strict development-turbine LOTO validation

The fixed-boundary interaction audit gives the following strict turbine-level LOTO result for the retained:

```text
soft_2.5deg + lambda=0.75
```

candidate.

| Holdout | Current candidate MAE | Constant MAE | Current candidate RMSE | Constant RMSE |
|---|---:|---:|---:|---:|
| `PPP_WTG12` | **0.357°** | 0.357° | **0.608°** | 0.608° |
| `PPP_WTG13` | **0.510°** | 1.004° | **0.702°** | 1.387° |
| `PPP_WTG14` | **0.206°** | 0.206° | **0.262°** | 0.262° |
| **Macro** | **0.358°** | **0.522°** | **0.524°** | **0.752°** |

Relative to the constant prior:

```text
macro MAE  ↓ ~31%
macro RMSE ↓ ~30%
```

The structural behavior is more important than the aggregate score:

- `PPP_WTG12`: no accepted dynamic boundary → constant prior;
- `PPP_WTG13`: two accepted persistent boundaries → dynamic correction;
- `PPP_WTG14`: no accepted dynamic boundary → constant prior.

In the reference run, the accepted `PPP_WTG13` boundaries were:

- `2023-02-19`
- `2023-07-27`

This is the intended behavior:

> **Know when to move, and know when not to move.**

Older micro-step boundary-recall diagnostics are not used for model selection because they do not represent the persistent macro-state objective of the current detector.

---

# 11. Unlabeled-target diagnostics

After fitting the final calibration on all three labeled turbines, the same model can be applied to unlabeled targets.

These target outputs are **deployment diagnostics, not local validation scores**, because the target labels are unavailable locally.

Reference diagnostics for the earlier stable all-days stage:

| Target | Accepted boundaries | Correction range | Prediction range |
|---|---|---:|---:|
| `PPP_WTG17` | 2023-06-25, 2023-10-15, 2024-01-07 | 5.258° | -6.680° to -1.422° |
| `SSS_WTG06` | 2023-05-28, 2023-09-24, 2024-10-13 | 4.434° | -8.162° to -3.729° |

The larger correction ranges relative to the labeled development turbines are treated as a **generalization risk**, not as evidence of better performance.

---

# 12. Public held-out validation

The externally reported public-validation results were:

| Model stage | RMSE | MAE | SHAPE | BIAS | ARI |
|---|---:|---:|---:|---:|---:|
| Stability-filtered RRS | 2.58° | 2.53° | 2.26° | -1.24° | 1.00 |
| Physics-Anchored Relative-State (PARS, `lambda=0.75`) | **1.44°** | **1.27°** | **0.99°** | -1.05° | **1.00** |

The important result is not only the lower RMSE.

The state segmentation is unchanged and:

```text
ARI = 1.00
```

in both cases.

The dominant improvement is therefore:

```text
SHAPE: 2.26° → 0.99°
```

while state structure remains correct.

This supports the development-stage diagnosis that confidence shrinkage was suppressing true state amplitude.

---

# 13. Error decomposition

The remaining held-out error is approximately decomposable as:

$$
\mathrm{RMSE}
\approx
\sqrt{\mathrm{SHAPE}^2+\mathrm{BIAS}^2}
$$

and numerically:

$$
\sqrt{0.99^2+1.05^2}\approx1.44^\circ.
$$

This gives a useful diagnostic separation:

```text
residual trajectory-shape error ≈ 0.99°
absolute-level bias            ≈ -1.05°
```

The dominant unresolved problem is therefore now the **absolute anchor**, rather than state segmentation.

The diagnostic intercept shift implied by the public bias is **not** applied to the model.

Doing so would use held-out leaderboard feedback as a calibration label.

---

# 14. Frozen-model protocol after public validation

The dynamic estimator is treated as frozen:

```text
fixed sector baseline
+
frozen accepted boundaries
+
soft pair-quality weighting
+
lambda = 0.75
+
beta = 1
```

Further model development is restricted to the absolute-anchor term:

$$
B0=C-\theta^*
$$

Any new absolute-anchor method should:

1. be developed only on the labeled development turbines;
2. preferably have physical or independently measurable justification;
3. not use the public target's `-1.05°` bias to select parameters;
4. be evaluated later as a new blind hypothesis.

This preserves the independence of the public validation result.

---

# 15. Model-selection and sensitivity audit

Several alternatives were evaluated before freezing the retained candidate.

| Candidate | Finding | Decision |
|---|---|---|
| Constant prior | Very stable, but cannot represent persistent state changes. | Retained as fallback baseline. |
| Multiscale state / sign veto | Did not provide a consistent strict-LOTO gain and added complexity. | Not used in final prediction path. |
| De-stepped PELT / local-increment accumulation | Exposed nonstationary transitions, but could amplify noise and accumulate level error. | Diagnostic only. |
| Power-guided Ridge/ML | Power changes were not sufficiently identifiable for reliable dynamic amplitude calibration. | Diagnostic only. |
| Dynamic pair baselines | Could absorb meaningful state structure and did not add stable evidence. | Not retained. |
| Soft pair-quality weighting | Improved robustness to disagreement without changing boundaries or absolute center. | Retained. |
| Partial amplitude restoration | Corrected confidence shrinkage while preserving state segmentation. | Retained with `lambda=0.75`. |

---

## 15.1 Why multiscale detection was tested

The motivation was that dynamic changes can occur over different characteristic time scales.

Weather, operating-state changes, sensor transitions, and other disturbances do not necessarily share one fixed window length.

A multiscale formulation was therefore tested to avoid assuming that all relevant state changes should appear at one temporal scale.

However, strict development-turbine evaluation did not show a consistent improvement over the simpler rolling / segmentation formulation.

The multiscale idea was therefore rejected from the production path rather than retained for complexity alone.

---

## 15.2 Why PELT was not used as the full state model

PELT-type segmentation was useful for exposing nonstationary periods and long regime changes.

However, direct accumulation of local increments can:

- amplify noisy local steps;
- accumulate level error;
- convert uncertain local changes into a drifting global trajectory.

PELT therefore remains a useful diagnostic / complementary detector, but not the sole amplitude-construction mechanism.

---

## 15.3 Why Power-guided ML was not retained

Power was investigated as a possible signal for dynamic state amplitude.

The difficulty was identifiability:

```text
Power variation
=
yaw effect
+
wind variation
+
operating-state variation
+
control effects
+
other environmental effects
```

The available development data did not support a sufficiently reliable decomposition for dynamic amplitude calibration.

Power-based Ridge / ML experiments were therefore kept as diagnostics rather than added to the final estimator.

---

# 16. Why PARS remains deliberately small

The retained prediction path contains only components with a demonstrated role:

```text
fixed wind-sector baselines
        ↓
relative-heading pair residuals
        ↓
reliability-weighted farm context
        ↓
persistent state detector
        ↓
soft state-level weighting
        ↓
partial amplitude restoration
        +
absolute B0 prior
```

The modelling rule is:

> **Additional complexity is introduced only when a simpler observable fails for a demonstrated reason and the replacement survives strict turbine-level validation.**

---

# 17. Model progression and ownership

The project evolved through three main modelling stages:

| Stage | Absolute yaw level | Temporal state boundaries | Relative state amplitudes |
|---|---|---|---|
| Original independent constant model | Independent `B0` calibration | None | None |
| Team hybrid with Daniel | Independent `B0` calibration | Daniel | Daniel |
| Physics-Anchored Relative-State (PARS) | Independent `B0` calibration | Label-free SCADA detector | Soft pair-quality relative-heading correction with partial amplitude |

The **FarmAnchor** variant is a post-release absolute-anchor exploration of
PARS, not a replacement stage. It changes only the shared `C` estimate and
keeps the PARS state detector, state correction, `lambda=0.75`, and `beta=1`
fixed.

The hybrid is a **team-derived historical comparison**.

PARS does not consume Daniel's:

- predictions;
- state labels;
- change-point dates;
- amplitude estimates.

This distinction is intentionally retained in the repository.

---

# 18. Submission history

The validation-submission history is summarized below. Filenames are preserved as repository artifacts; organizer-side submissions may follow a separate immutable workflow.

| Entry | Submission | Model / note |
|---|---|---|
| #20 | Results_33_T3_4.csv | Latest Stability-filtered RRS: frozen label-free boundaries, soft pair-quality weighting with a 2.5° scale, partial amplitude lambda=0.75, B0_i=C-theta_star_i, and fixed beta=1. Contains all 731 PPP_WTG17 dates; no target labels were used. Strict LOTO reference: MAE 0.358°, RMSE 0.524°. |
| Previous RRS update | Results_33_T3_3.csv | Stability-filtered RRS validation submission. Strict LOTO reference: MAE 0.362°, RMSE 0.610°. |
| #14 | Results_33_T3_2.csv | Independent same-site relative-power-efficiency state model. Strict nested turbine-level LOTO: macro MAE 0.420°, RMSE 0.564°; 731 PPP_WTG17 rows and two predicted clusters. |
| #13 | Results_33_T3_1.csv | Team-derived state-aware hybrid: my physics-based long-term level plus Daniel's piecewise state residual. The previously merged final submission was left unchanged. |
| #12 | Results_33_T3_0.csv | Initial independent constant-line physics submission. Results_33_T3_final.csv was also created for the final target and was later removed by request from the organizer repository. |

---

# 19. Main modelling lessons

The project produced several broader conclusions.

1. Strong internal LOTO performance can still underestimate hidden-target domain shift.
2. Absolute yaw calibration and temporal state detection are distinct estimation problems.
3. A simple long-term aerodynamic estimator can remain valuable as a low-variance **prior** even after it fails as a complete model.
4. Neighbouring turbines are more useful as a robust local reference field than as direct target-yaw regressors.
5. Persistent relative-heading structure can recover useful state information without target labels.
6. Sensor / encoder re-references can mimic physical yaw-state changes and require explicit veto logic.
7. A good state detector must be allowed to return **no correction**.
8. Confidence weighting can improve robustness while simultaneously shrinking amplitude too strongly.
9. Sensitivity analysis is more useful than adding complexity without validation evidence.
10. A failed modelling hypothesis can still be informative if it isolates which part of the problem is not identifiable.
11. Public validation should remain a test, not silently become an additional training label.
12. Contribution boundaries should remain explicit when team-derived diagnostics influence later independent modelling.

---

# 20. Data availability and reproducibility

Challenge SCADA data are intentionally not included in this public repository.

The repository does not publish:

- raw SCADA;
- challenge archives;
- parquet challenge files;
- local derived caches that may inherit challenge-data restrictions.

Data access must be requested separately through WeDoWind / the challenge organizers.

Because the source SCADA are restricted, a clean clone cannot reproduce all numerical experiments from raw data.

Executed notebooks are included so that:

- modelling history;
- recorded outputs;
- diagnostics;
- method progression

can still be inspected.

---

# 21. CI strategy

CI validates code-level behavior without redistributing challenge data.

The test suite is intended to cover data-independent components such as:

- model utilities;
- circular-angle handling;
- deterministic calculations;
- import / execution behavior;
- regression behavior of code paths that do not require restricted SCADA.

Data-dependent tests are skipped when the local challenge cache is unavailable.

Run the test suite with:

```bash
python -m unittest discover -s tests
```

---

# 22. Environment

Python 3.11 is recommended.

```bash
conda create -n yaw-baseline python=3.11 -y
conda activate yaw-baseline
pip install -r requirements.txt
```

---

# 23. Main implementation files

```text
README.md
FINAL_METHOD_SUMMARY.md
YawMisalignment_Independent_Model_Release.ipynb
YawMisalignment_Independent_Model_FarmAnchor_Exploration.ipynb
requirements.txt

src/
├── baseline_loto_ridge.py
├── baseline_openoa_power_vane_updated.py
├── fleet_context.py
├── yaw_relative_state.py
├── yaw_model.py
├── yaw_farm_anchor.py
├── yaw_self_response.py
└── make_final_submission_vane_level.py

submissions/
├── Results_33_T3_0.csv
├── Results_33_T3_1.csv
└── Results_33_T3_final.csv

tests/
└── test_final_method.py
```

The Daniel-derived notebook is retained as historical provenance, not as a dependency of PARS.

---

# 24. Current research state

The dynamic-state problem is considered sufficiently validated for the current project stage:

```text
state partition: correct on public validation (ARI = 1)
shape error:     reduced to ~0.99°
```

The main unresolved research question is now:

> **How can the absolute yaw anchor be transferred more reliably to an unseen turbine without using held-out target feedback?**

This is deliberately treated as a separate calibration problem rather than as a reason to reopen every component of the dynamic-state estimator.

### Farm-anchor exploration

The exploratory notebook `YawMisalignment_Independent_Model_FarmAnchor_Exploration.ipynb`
tests one isolated change to the absolute calibration. Instead of estimating
`C` from the release long-run turbine means, it first uses the state-consistent
daily support

$$
a_i(d)=y_i(d)+c_{i,0.75}^{\mathrm{soft}}(d)+\theta_i^*.
$$

to form a quality-weighted turbine-level anchor summary `A_i`:

$$
A_i=\mathrm{RobustCenter}_d\left(a_i(d);q_i(d)\right).
$$

A robust Huber centre is then fitted across those turbine summaries:

$$
C_{\mathrm{corrected}}=\mathrm{HuberCenter}_i(A_i),
\qquad
B_{0,i}=C_{\mathrm{corrected}}-\theta_i^*.
$$

$$
\hat y_i(d)=B_{0,i}-c_{i,0.75}^{\mathrm{soft}}(d).
$$

If labelled turbines from the target farm are available, the corresponding farm
centre is preferred; otherwise the global centre is used. The current PPP
development set contains only one farm, so the full-fit value
`C_corrected = -6.155553°` is primarily a robust corrected global/farm-anchor
estimate, not evidence of cross-farm transfer.

The existing frozen relative-state detector, fixed pair/sector baselines, soft
pair-quality weighting, `lambda=0.75`, `beta=1`, and global-median
`theta_star` estimator are unchanged. Power does not directly enter the final
yaw prediction. Strict development-turbine LOTO improved from
`0.358° / 0.524°` MAE/RMSE for the release anchor to `0.320° / 0.510°` for the
corrected anchor. This remains an exploratory anchor variant;
`YawMisalignment_Independent_Model_Release.ipynb` and the original PARS model
are preserved as the release reference. No new external leaderboard validation
has been performed.

---

## Summary

PARS is best understood as a **physics-informed relative-state estimator with a low-variance absolute prior**.

Its development path is:

```text
constant physical prior
        ↓
hidden-target failure diagnosis
        ↓
temporal-state hypothesis
        ↓
independent cross-turbine relative-heading model
        ↓
strict LOTO sensitivity analysis
        ↓
confidence-shrinkage diagnosis
        ↓
partial amplitude restoration
        ↓
blind validation
        ↓
absolute-anchor problem isolated
```

The project therefore emphasizes not only predictive performance, but also:

- physical interpretability;
- domain-shift awareness;
- turbine-level validation;
- sensitivity analysis;
- model-selection discipline;
- explicit contribution ownership;
- reproducible software practices under restricted-data constraints.
