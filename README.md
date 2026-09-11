# Static Yaw Misalignment — Energy Hackdays / WeDoWind 2026

Participant **33** · T3 submission

This repository contains our solution for the NADARA / WeDoWind static yaw misalignment challenge.

The project started from a physics-informed power-vs-vane calibration model and later evolved into a **state-aware hybrid** after public leaderboard feedback showed that a constant-per-turbine prediction did not transfer well to the hidden target turbines.

## Current method

The original physics model estimates one long-term yaw level per turbine:

$$
\hat y_{\mathrm{level}} = C - \theta^\star
$$

where:

- $\gamma = \mathrm{wrap}(WindDir - NacDir)$
- a 21-day centered window is used
- power is normalized within 1.0-unit WindSpeed bins
- the apparent optimum is the observed median $\gamma$ of the highest normalized-power angle bin
- $\theta^\star$ is the long-term median apparent optimum
- $C$ is calibrated from the three labelled development turbines

The frozen calibration gave:

- `PPP_WTG17`: long-term level **-3.596°**
- `SSS_WTG06`: long-term level **-4.568°**

### Why the model changed

Strict leave-one-turbine-out validation on the three labelled turbines gave a strong internal result:

- macro MAE: **0.519°**

However, the original constant deployment did not transfer well to the hidden targets:

| Round | RMSE | MAE |
|---|---:|---:|
| Public validation (`PPP_WTG17`) | 5.28° | 5.25° |
| Final (`SSS_WTG06`) | 3.46° | 3.39° |

This showed that the long-term level alone was not sufficient.

## State-aware hybrid

The current model preserves our physics-based absolute yaw level and adds a **piecewise-constant temporal state residual** derived from a teammate's independent state model.

For each target:

$$
\delta(t)
=
\hat y_{\mathrm{state}}(t)
-
\mathrm{mean}_t\left[\hat y_{\mathrm{state}}(t)\right]
$$

and the hybrid prediction is

$$
\boxed{
\hat y_{\mathrm{hybrid}}(t)
=
L_{\mathrm{physics}} + \delta(t)
}
$$

This means:

- the **absolute long-term center remains our physics calibration**
- the temporal model contributes only **zero-mean state changes**
- the output is piecewise constant rather than noisy daily variation
- cluster IDs represent the detected temporal regimes

For the current hybrid:

- validation target: **2 temporal clusters**
- final target: **13 temporal clusters**
- validation trajectory shift: **-0.061373°**
- final trajectory shift: **-2.083073°**

The state component is a team contribution. This repository does not claim the teammate's independent T0/T3 method as our own; we use its temporal regime structure only after zero-mean centering.

## Public validation update

The state-aware hybrid substantially improved the public validation result:

| Submission | RMSE | MAE | Cluster ARI |
|---|---:|---:|---:|
| Original constant T3 (`Sub #0`) | 5.28° | 5.25° | — |
| State-aware hybrid T3 (`Sub #1`) | **1.58°** | **1.49°** | **1.00** |

The perfect **Cluster ARI = 1.00** shows that the temporal regime segmentation matched the hidden validation clusters exactly.

The remaining error is therefore mainly associated with **cluster-level yaw calibration**, not with the timing of state changes.

## Start here

Open the executed notebook:

[`YawMisalignment_Final_Summary.ipynb`](YawMisalignment_Final_Summary.ipynb)

It contains:

- the physical framing
- method evolution
- strict leave-one-turbine-out validation
- robustness analysis
- domain-shift diagnostics
- hidden leaderboard feedback
- the state-aware hybrid update
- the final state tables used to construct the updated predictions

The notebook is intentionally self-contained for inspection. The compact temporal state tables used by the hybrid are embedded directly in the notebook.

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

### Submission files

- **`Results_33_T3_0.csv`** — historical validation submission from the original constant-per-turbine model.
- **`Results_33_T3_1.csv`** — updated validation submission using the state-aware hybrid.
- **`Results_33_T3_final.csv`** — state-aware hybrid final-round artifact maintained in this research repository.

The organizer submission repository enforces immutable merged submissions, so files in this research repository may not always match the currently merged organizer-side final entry.

## Original physics estimator

The frozen B0 estimator uses:

- operating WindSpeed range: 3–13
- positive Power, RotSpeed, and GenSpeed
- Power below its 98th percentile
- PitchAngle below its 95th percentile
- $\gamma$ restricted to ±30°
- 1° angle bins
- at least 20 samples per angle bin
- at least 8 valid angle bins
- 21-day centered rolling windows
- power normalized by median power within WindSpeed bins
- the observed median angle of the highest-power valid bin

Across the three labelled turbines, the long-term apparent optima were combined with a shared calibration constant:

$$
C_{\mathrm{FINAL}} = -6.0793162066^\circ
$$

The strict outer leave-one-turbine-out constant predictions gave:

| Holdout | Prediction | MAE |
|---|---:|---:|
| `PPP_WTG12` | -2.744° | 0.354° |
| `PPP_WTG13` | -6.348° | 0.997° |
| `PPP_WTG14` | -1.711° | 0.207° |
| **Macro** | — | **0.519°** |

These numbers are retained as **internal model-selection evidence**, not as external generalization performance.

## Data and reproducibility

Challenge data are intentionally **not included** in this public repository.

In particular, the repository does not publish:

- raw SCADA
- the original `data/` directory
- parquet datasets
- challenge archives
- local derived caches that may inherit challenge-data restrictions

Because of that, a clean clone cannot fully reproduce every numerical experiment from raw data.

The executed notebook is included so the analysis and recorded outputs can still be inspected. The state-aware deployment section embeds the compact state tables needed to reconstruct the current hybrid trajectories.

## Tests

The public-repository test suite can be run with:

```bash
python -m unittest discover -s tests
```

Data-dependent integration tests are skipped when the local challenge cache is absent. This is intentional for the public release.

## Environment

A Python 3.11 environment is recommended:

```bash
conda create -n yaw-baseline python=3.11 -y
conda activate yaw-baseline
pip install -r requirements.txt
```

## Important interpretation

`WindDir - NacDir` is **not treated as the true yaw label**.

The vane/controller reference can itself be biased, so the method looks for the apparent angle associated with maximum normalized power and calibrates that relationship using labelled turbines.

The main lesson from the project is that two distinct components matter:

1. **absolute yaw level calibration**
2. **temporal yaw-state detection**

The original physics model captured the first component well on the development turbines. Hidden-set results showed that the second component was necessary for deployment, motivating the current state-aware hybrid.

## Limitations

The labelled development set contains only three turbines, so cross-turbine validation has high variance and can give overly optimistic model-selection results.

The final SSS turbine also represents a stronger domain shift than the PPP validation turbine. Geometry diagnostics showed that cross-site transfer is substantially harder.

The current temporal state component is therefore best interpreted as a pragmatic hackathon solution rather than a fully validated general-purpose yaw estimator.

---

Repository: `HuijiaoLuo/2026EnergyHacks_WeDoWind`
