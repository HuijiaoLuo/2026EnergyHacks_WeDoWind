# Static Yaw Misalignment — Energy Hackdays / WeDoWind & NADARA 2026

**Participant 33 · Independent modelling / portfolio release**

A physics-informed, label-free SCADA workflow for estimating persistent wind-turbine yaw states and transferring the model to unseen turbines.

## Highlights

- Built an independent model from two years of SCADA history and three labeled training turbines.
- Separated **absolute calibration** from **dynamic state estimation**.
- Used neighbouring turbines as a reliability-weighted **farm-level reference field**, not as direct target-yaw regressors.
- Performed model selection and sensitivity analysis only on the labeled development turbines using strict turbine-level leave-one-turbine-out (LOTO) validation.
- Compared rolling and conservative PELT/L2 segmentation with multiscale, dynamic-baseline, and Power-guided alternatives; retained only components with stable validation gains.
- Improved development-turbine macro MAE from **0.522° to 0.358°** and RMSE from **0.752° to 0.524°** versus the constant prior.
- On the public held-out turbine, recovered yaw-state structure with **ARI = 1.00** and reduced **SHAPE from 2.26° to 0.99°**.
- Final public-validation result: **RMSE 1.44° / MAE 1.27°**. The remaining error is dominated by an approximately **−1.05° absolute bias**.
- Public validation was used only for blind evaluation and post-hoc diagnosis, not for tuning.
- CI validates code behavior without redistributing restricted SCADA data.

## Problem

The challenge provides:
- **T0:** two years of unlabeled SCADA;
- **T3:** two years of SCADA for three turbines with yaw-misalignment labels.

The objective is to build a model that transfers to unseen turbines.

The final model treats this as two coupled problems:

```text
absolute yaw prior
        +
label-free persistent state correction
        ↓
daily yaw trajectory
```

## Current independent model

The retained model is

$$
\hat y_i(d)=B_{0,i}-c_{i,\lambda}^{\mathrm{soft}}(d),
\qquad
B_{0,i}=C-\theta_i^*
$$

with

$$
c_{i,\lambda}(d)=c_{i,\mathrm{stable}}(d)+\lambda[c_{i,\mathrm{full}}(d)-c_{i,\mathrm{stable}}(d)],\qquad\lambda=0.75.
$$

The physical slope is fixed at \($\beta=1$\).

The model falls back to the constant prior when SCADA does not provide sufficiently strong evidence for a persistent state change.

## 1. Absolute yaw prior

For each turbine,

```text
gamma = wrap(WindDir - NacDir)
```

is evaluated over rolling SCADA windows to estimate an apparent power-optimal vane angle.

The long-term turbine feature is

```text
theta_star = median(valid rolling apparent optima)
```

and the three labeled development turbines estimate one shared calibration:

```text
B0_i = C - theta_star_i
```

This gives a low-variance physics-based prior without fitting a separate model to each target turbine.

## 2. Farm-level relative-heading states

The dynamic channel uses only SCADA and same-site turbine context.

For target turbine `i` and neighbour `j`, the pipeline:

1. forms circular nacelle-heading differences;
2. removes pair- and wind-sector-specific baselines;
3. estimates pair reliability from coverage, residual MAD, and distance;
4. robustly aggregates usable pair residuals into a daily target-relative heading signal;
5. builds a neighbour-only background residual to identify common-mode motion.

Nearby turbines therefore act as a **local measurement field**.
### Label-free state segmentation

Segmentation is performed before supervised calibration:

1. smooth the daily robust relative-heading signal;
2. nominate persistent before/after changes with a rolling detector, supplemented by conservative L2/PELT segmentation for long two-sided regimes;
3. merge near-duplicate candidates and reject sensor-like jumps, local common-mode motion, weak persistence, and insufficient pair agreement;
4. freeze the surviving boundary dates before estimating state amplitudes or fitting the shared calibration.

The accepted boundaries partition the daily relative-heading trajectory into persistent states. Soft pair-quality weighting and partial-amplitude blending can change state levels, but cannot create or move boundaries.

Candidate state changes are rejected when they are better explained by:
- sensor / encoder-like jumps;
- sensor-shadow periods;
- local background or common-mode motion;
- insufficient pair agreement;
- same-site synchronous events;
- weak event confidence.

Only surviving persistent events can create a dynamic correction.

## 3. State amplitude

The stable estimator intentionally shrinks uncertain state amplitudes.

A full-amplitude estimate keeps the same accepted boundaries but restores the measured relative-heading level. The retained model interpolates between them with

```text
lambda = 0.75
```

selected from development-turbine LOTO before public validation.

Soft pair-quality weighting reduces the influence of days with large cross-pair disagreement without creating new state boundaries or changing the absolute anchor.

## Validation protocol

Development and sensitivity analysis use only:

- `PPP_WTG12`
- `PPP_WTG13`
- `PPP_WTG14`

For each LOTO fold:

1. two turbines estimate the shared calibration;
2. the third turbine's yaw labels are hidden;
3. its own SCADA estimates the physical prior and relative-heading states;
4. the full model is transferred to the held-out turbine.

No public-target metric is used to select state boundaries, pair weighting, multiscale parameters, `lambda`, or an intercept correction.

## Development-turbine LOTO

| Holdout | Current MAE | Constant MAE | Current RMSE | Constant RMSE |
|---|---:|---:|---:|---:|
| `PPP_WTG12` | **0.357°** | 0.357° | **0.608°** | 0.608° |
| `PPP_WTG13` | **0.510°** | 1.004° | **0.702°** | 1.387° |
| `PPP_WTG14` | **0.206°** | 0.206° | **0.262°** | 0.262° |
| **Macro** | **0.358°** | **0.522°** | **0.524°** | **0.752°** |

Relative to the constant prior:
- macro MAE improves by approximately **31%**;
- macro RMSE improves by approximately **30%**.

The intended behavior is visible in the folds:
- `PPP_WTG12`: no accepted persistent boundary → constant prior;
- `PPP_WTG13`: persistent state changes → dynamic correction;
- `PPP_WTG14`: no accepted persistent boundary → constant prior.

> **Move when persistent evidence exists; otherwise stay at the low-variance prior.**

## Blind public-validation result

| Model stage | RMSE | MAE | SHAPE | BIAS | ARI |
|---|---:|---:|---:|---:|---:|
| Stability-filtered RRS | 2.58° | 2.53° | 2.26° | -1.24° | 1.00 |
| Amplitude-corrected RRS (`lambda=0.75`) | **1.44°** | **1.27°** | **0.99°** | -1.05° | **1.00** |

ARI remains 1.00, so the gain comes primarily from **state-amplitude correction rather than re-segmentation**.

The remaining error is approximately

$$
\sqrt{0.99^2+1.05^2}\approx1.44^\circ,
$$

suggesting that the dominant unresolved problem is now **absolute-anchor transfer**, not state detection.

The observed public-target bias is diagnostic only and is **not** fed back into the model as a calibration correction.

## What was tested and not retained

Several alternatives were evaluated before freezing the current model:

- **Multiscale state detection:** motivated by variable-duration operating and environmental changes, but no consistent strict-LOTO gain.
- **De-stepped PELT / accumulated local increments:** useful diagnostically, but prone to level-error accumulation. A conservative PELT/L2 supplement remains in the final detector; unrestricted PELT is not the model.
- **Power-guided ML and dynamic pair baselines:** insufficiently identifiable for reliable dynamic amplitude calibration.

The final prediction path is deliberately small: fixed sector baselines, a persistent rolling detector with a conservative PELT/L2 supplement, reliability-weighted relative-heading states, the physics-based `B0` prior, and partial amplitude restoration.

Full model-development history is documented in [`FINAL_METHOD_SUMMARY.md`](FINAL_METHOD_SUMMARY.md).

## Ownership and model progression

The project evolved through three stages:

| Stage | Absolute level | Temporal states | Relative amplitude |
|---|---|---|---|
| Initial constant model | Independent physics calibration | None | None |
| Team hybrid | Independent calibration | Teammate state model | Teammate amplitudes |
| Current independent model | Independent physics calibration | Label-free SCADA detector | Independent relative-heading correction |

The hybrid was retained only as historical evidence that temporal state structure matters.

The current independent model does **not** consume teammate predictions, state labels, or change-point dates.

## Data, CI, and reproducibility

Challenge SCADA data are not redistributed here. Data access must be requested separately through WeDoWind / the challenge organizers.

The repository does not publish raw SCADA, challenge archives, parquet challenge files, or derived caches that may inherit challenge-data restrictions.

Executed notebooks preserve the modelling history and recorded outputs.

CI focuses on code-level reproducibility. Data-dependent tests are skipped when the local challenge cache is unavailable.

```bash
conda create -n yaw-baseline python=3.11 -y
conda activate yaw-baseline
pip install -r requirements.txt
python -m unittest discover -s tests
```

## Main files

```text
README.md
FINAL_METHOD_SUMMARY.md
YawMisalignment_Independent_Model_Release.ipynb

src/
tests/
slides/
submissions/
```

## Main modelling takeaway

The strongest lesson from this project is that yaw estimation is not one monolithic regression problem.

A more robust decomposition is:

```text
absolute calibration
        +
persistent relative state structure
        +
farm-level measurement context
```

The current model therefore prioritizes **transferability, interpretability, sensitivity analysis, and validation discipline** over additional model complexity.

Repository: `HuijiaoLuo/2026EnergyHacks_WeDoWind`
