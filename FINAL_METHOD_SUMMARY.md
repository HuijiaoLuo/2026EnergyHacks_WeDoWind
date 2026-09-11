# Static Yaw Misalignment — Final Method

## 1. Problem

Estimate persistent yaw misalignment from cleaned 15-minute wind-turbine SCADA.
`gamma = wrap(WindDir - NacDir)` in `[-180°, 180°)` is a vane/controller-frame
measurement, not the true yaw label. Reference and aerodynamic biases can remain.
The qualitative yaw-loss relation `P ~ P0*cos(theta)^p` motivates looking for
peak relative power; the final method does not estimate `p` or invert this law.

## 2. Data split

- Labelled development: `PPP_WTG12`, `PPP_WTG13`, `PPP_WTG14`.
- Public validation target: `PPP_WTG17`; final target: `SSS_WTG06`.
- Other PPP/SSS turbines provide unlabeled context, used only in controlled diagnostics.
- Input: private `turbines_data.zip`; site-relative layouts: the two location CSVs.
- Submission calendar: every day from 2023-01-01 through 2024-12-31 (731 days).

## 3. Validation philosophy

Use strict outer leave-one-turbine-out (LOTO), never a random row split. Each
fold estimates its calibration from the other two labelled turbines. The held-out
turbine supplies only its unlabeled long-term vane estimate. Full-history windows
make this offline/transductive validation, not prospective time forecasting.

The sign check in original `TrainValidation.ipynb` cell 26 (zero-based) selects
between ±theta using the two training-turbine means, with an intercept fitted
on those same means. All three outer folds select minus. It is **training-only
sign selection inside outer LOTO**, not an explicit additional inner-CV loop.
The formula itself was discovered after examining these three turbines; repeated
development on them limits the independence of the final performance estimate.

## 4. Frozen B0 estimator

Reuse `baseline_openoa_power_vane_updated.py` and the operating filter from
`baseline_loto_ridge.py`: WindSpeed in [3,13], positive Power/RotSpeed/GenSpeed,
then Power below its 98th percentile and PitchAngle below its 95th percentile.
Signals are anonymized/scaled; these are not physical m/s or kW cutoffs.

Each target date uses 10 previous days, that date, and 10 following days.
Within the window, normalize `Power / median(Power | WindSpeed bin)` using
1.0-unit bins. Restrict gamma to [-30,+30] degrees, then use 1-degree angle
bins, at least 20 samples/bin and at least 8 valid bins. The selected angle is
the **observed median gamma** in the valid bin with maximum median normalized
power. This is the formal raw argmax B0, not its quadratic diagnostic.

`theta_star = median(valid rolling theta_hat_argmax)` over the full period.
Failed B0 windows are omitted from this median; the final constant still covers
all dates. Labelled evaluation dates are those surviving the original operating
filter: 725 / 727 / 727 days, not the full submission calendar.

## 5. Discovery and frozen equation

`C_i = mean(target_i) + theta_star_i`; estimate C as the equally weighted mean
of training-turbine C_i, not a day-count-weighted mean.

| Turbine | Target mean | theta_star | C_i |
|---|---:|---:|---:|
| PPP_WTG12 | -2.428 | -3.441 | -5.868 |
| PPP_WTG13 | -6.613 | +0.357 | -6.256 |
| PPP_WTG14 | -1.762 | -4.351 | -6.114 |

**Final model: `yaw_hat = C_FINAL - theta_star`.**

`C_FINAL = -6.079316206588366°` (displayed as -6.079°).
No geometry correction, site shift, residual ML or daily dynamics enter it.

## 6. Strict LOTO results

| Holdout | Predicted constant | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| PPP_WTG12 | -2.744 | 0.354 | 0.601 | -0.316 |
| PPP_WTG13 | -6.348 | 0.997 | 1.374 | +0.265 |
| PPP_WTG14 | -1.711 | 0.207 | 0.263 | +0.051 |

**Macro MAE: 0.519°**, versus 2.647° for the prior residual-Ridge model.
The summary notebook recomputes the winner from cached daily B0 estimates.

## 7. Robustness

- MAE-optimal constant oracle: **0.462° macro MAE**. It uses the holdout's own
  target **median**, so it is a non-deployable diagnostic, not a target-mean baseline.
- Shared 28-day block bootstrap, 500 replicates, seed 42: macro median **0.678°**,
  mean **0.738°**, central 95% robustness range **[0.448°, 1.470°]**.
- Blocks resample cached daily B0 estimates and targets; vane medians,
  calibration and errors are recomputed. Raw SCADA and rolling B0 fits are
  **not** refitted in each replicate. Overlapping windows remain dependent.
- These are sampling-robustness intervals, not formal confidence intervals or
  calibrated bounds on unseen-site prediction error.

## 8. Domain shift and context

Original notebook cells 39–41 use layout-only directional exposure. PPP train
means are 0.274/0.332/0.280. PPP_WTG17 is 0.382, with a 0.674 profile correlation
to WTG12. SSS_WTG06 is 0.529; correlations with PPP train are
-0.217/-0.174/-0.180. PPP17 has moderate shift; SSS06 is the riskier transfer.

Geometry-matched calibration worsened all folds (0.533° vs 0.519°); geometry
therefore remains a risk diagnostic, not a model input or correction.

SSS theta_star: WTG04 +1.432°, WTG05 -1.553°, WTG06 -1.511°, WTG07 +1.388°,
WTG16 -0.620°. SSS06 differs from the context median by -1.896°, with robustness
range [-3.869°, 0.544°]; it is not a clearly established site-relative outlier.
PPP and SSS context medians are -1.526° and +0.384°. Their +1.910° shift is
**not** used for yaw calibration: context normalization failed validation.

## 9. Rejected and superseded methods

| Method | Macro MAE (deg) | Evidence/status |
|---|---:|---|
| Daily Ridge | 2.933 | Root `baseline_loto_scores.csv`; legacy |
| Tuned RF | 2.783 | `experiments/ensemble/rf_tuned_loto_scores.csv`; development-selected configuration |
| B0 raw argmax / quadratic | 3.669 / 2.644 | 97.2% / 77.3% coverage; unequal subsets |
| B1 sequential cubic / argmax | 4.118 | Worse than B0 also on common dates |
| B2a joint cubic/log-yaw | 2.902 | 63.5% coverage; rejected, not a comparable full-coverage gain |
| Physics v2 | 2.829 | Frozen daily physics; superseded |
| Physics + residual Ridge | 2.647 | Previous strongest reference |
| Enhanced residual | 2.642 | Only 0.005° gain, not meaningful |
| TV denoising | 2.826 | Negligible improvement over physics |
| Level + full dynamics / selected dynamics | 0.930 / 0.580 | Worse than constant winner |
| Geometry-matched reference | 0.533 | Every fold worsened |
| Context-normalized block reference | 2.142 | Strong degradation |
| Generic ML power surface | 8.363 | Plateau/boundary instability in argmax |

GP/paper-inspired notebooks were abandoned for low hackathon return; no winning
GP metric is asserted. RF direct blends also failed to improve residual Ridge.
The archived v3 ~2.524° result reused a deployment physics choice selected on all
three turbines in later calibration, so it is **not comparable strict evidence**.
Older README/audit claims that residual Ridge is the current winner are superseded
by later executed `TrainValidation` cells, not competing final results.

Predictive power accuracy does not imply identifiability of the power-optimal
yaw angle. There is no validated improvement over the frozen final mapping
from the additional methods tested here.

## 10. Final target predictions

| Target | theta_star | Frozen export | Bootstrap mean | Std | 95% robustness range |
|---|---:|---:|---:|---:|---|
| PPP_WTG17 | -2.484 | **-3.596°** | -3.578 | 0.375 | [-4.527, -2.996] |
| SSS_WTG06 | -1.511 | **-4.568°** | -4.438 | 1.365 | [-6.644, -2.280] |

The unrounded full-data calculations are -3.5955834011° and -4.5679000871°.
Exports preserve the explicitly frozen three-decimal predictions, formatted
with six decimals. No bootstrap median replaces them. Target robustness uses
1,000 replicates with target blocks and independently sampled training blocks,
matching original cell 51. Each submission contains all 731 dates and cluster 0.
Schema matches the existing project submissions; no official separate sample
file was found, so prior submissions and challenge date metadata are the references.

## 11. Limitations

Only three labelled turbines from one site support calibration. Formula discovery
and repeated model comparisons used that same development set. C is not fully
stable across years. Constant predictions cannot follow actual yaw interventions.
Geometry diagnostics are simplified exposure proxies, not validated wake physics.
Cross-site transfer to SSS is the principal unresolved uncertainty. Obtain new
labelled-site evidence before changing the frozen winner.

## 12. Reproduction

Run inside `github_release/` (or the root of its standalone GitHub repository):

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python -m nbconvert --to notebook --execute --inplace YawMisalignment_Final_Summary.ipynb
python src/make_final_submission_vane_level.py
```

The notebook writes no experiment CSV/JSON/figure files. Its plots and tables are
embedded, and its default execution uses the derived cache. It recomputes final
LOTO, sign selection and bootstrap results; historical model scores are documented
executed evidence, not expensive reruns. Optional verification using separately
obtained cleaned SCADA:

```bash
python src/make_final_submission_vane_level.py --verify-scada /path/to/turbines_data.zip
```

The copied B0 implementation changes only its import-root path for the `src/`
layout; estimator calculations are unchanged. Original files remain in place.

## 13. Exact release dependencies and provenance

- `src/baseline_openoa_power_vane_updated.py`: frozen B0 implementation.
- `src/baseline_loto_ridge.py`: existing reading, filtering and wrapping helpers.
- `src/make_final_submission_vane_level.py`: final calibration, LOTO, bootstrap,
  optional B0 regeneration and checked submission export.
- `data/vane_daily.csv`: 3,615 derived rows (239 KB), five turbines, only dates,
  target labels and B0 angle estimates. Required to reproduce final validation
  and robustness without distributing private SCADA.
- `data/turbine_locations_PPP.csv`, `data/turbine_locations_SSS.csv`: layout plots.
- `requirements.txt`, `tests/test_final_method.py`, the final notebook and docs.
- `slides/final.pptx` and the two CSVs in `submissions/`: final presentation/export.

Derived training rows come from `experiments/power_vane/b0_21d_updated_predictions.csv`;
target rows come from `_archive/outputs/hackathon_fasttrack/test_b0_predictions.csv`.
Their target medians/counts match the later final notebook exactly; sampled
SCADA windows were also checked. The old cache's later temporal processing is
not used: only its original frozen 21-day B0 columns are extracted.

Original historical notebooks, experiments, slides and data are retained locally
but are not release dependencies. The final slide copy updates the latest
`yaw_misalignment_physics_vane_calibration_update.pptx` narrative.

SHA256 evidence at consolidation:

```text
TrainValidation.ipynb
25e71cf5ca2fc2980e63f5da215e2f07a46b787d2d3af90a74968ebfad4494e8
b0_21d_updated_predictions.csv
102d3d176ec9eb25b6869c50e27a640d32ddbfa3ba69ec00323de27f0e535987
archived test_b0_predictions.csv
ab65da9b4de38088f5d3d068085f5d4fd1ef9ead1d9a30808f81619e65f60db2
release data/vane_daily.csv
9d10383e1ae305d057b799fb139830e7ac4fa7ed124f6f97a72de4226e06e14a
```
