import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from make_final_submission_vane_level import (
    TRAIN, TARGETS, FROZEN_POINTS, load_daily, turbine_summary, validate_loto,
    predict_targets, bootstrap_loto, submission, validate_submission,
)
from baseline_openoa_power_vane_updated import estimate_peak_angle
from baseline_loto_ridge import wrap_180


class FinalMethodTests(unittest.TestCase):
    # @classmethod
    # def setUpClass(cls):
    #     cls.daily = load_daily(ROOT / "data/vane_daily.csv")
    @classmethod
    def setUpClass(cls):
        cache = ROOT / "data" / "vane_daily.csv"
        if not cache.exists():
            raise unittest.SkipTest(
            "data/vane_daily.csv is intentionally excluded from the public repository"
            )
        cls.daily = load_daily(cache)

    def test_frozen_scores_and_sign(self):
        scores = validate_loto(self.daily)
        np.testing.assert_allclose(scores.mae, [0.354, 0.997, 0.207], atol=0.0005)
        self.assertTrue(scores.selected_sign.eq(-1).all())
        self.assertAlmostEqual(scores.mae.mean(), .519, delta=.0005)
        self.assertAlmostEqual(scores.oracle_mae.mean(), .462, delta=.0005)

    def test_holdout_labels_do_not_change_its_prediction(self):
        original = validate_loto(self.daily)
        for holdout in TRAIN:
            modified = self.daily.copy()
            modified.loc[modified.turbine_id.eq(holdout), "target"] = 999.
            altered = validate_loto(modified)
            self.assertEqual(original.loc[holdout, "prediction"], altered.loc[holdout, "prediction"])
            self.assertEqual(original.loc[holdout, "selected_sign"], altered.loc[holdout, "selected_sign"])

    def test_final_points_and_calibration(self):
        self.assertAlmostEqual(turbine_summary(self.daily).C_i.mean(), -6.079316206588366)
        points = predict_targets(self.daily)
        for turbine in TARGETS:
            self.assertEqual(round(points.loc[turbine, "yaw_full_precision"], 3), FROZEN_POINTS[turbine])

    def test_submission_calendar_and_schema(self):
        for turbine in TARGETS:
            frame = submission(turbine)
            validate_submission(frame, turbine)
            self.assertEqual(len(frame), 731)
            self.assertIn("2024-02-29", frame.date.to_list())
            self.assertTrue(frame.cluster.eq(0).all())

    def test_submission_rejects_corruption(self):
        frame = submission(TARGETS[0])
        for broken in [frame.iloc[:-1], frame.iloc[::-1], pd.concat([frame.iloc[:-1], frame.iloc[:1]])]:
            with self.assertRaises(ValueError):
                validate_submission(broken, TARGETS[0])

    def test_bootstrap_is_deterministic(self):
        a = bootstrap_loto(self.daily, n_boot=8)
        b = bootstrap_loto(self.daily, n_boot=8)
        pd.testing.assert_frame_equal(a, b)

    def test_b0_label_independence_and_sign(self):
        rng = np.random.default_rng(42)
        gamma = np.repeat(np.arange(-9.5, 10., 1.), 30)
        wind = rng.uniform(5., 10., len(gamma))
        data = pd.DataFrame({"gamma": gamma, "WindSpeed": wind,
                             "Power": np.exp(.2 * wind - (gamma - 2.)**2 / 100.),
                             "yaw_misalignment_deg": 0.})
        a = estimate_peak_angle(data, 1., 1., 20, 8)
        data["yaw_misalignment_deg"] = -100.
        b = estimate_peak_angle(data, 1., 1., 20, 8)
        self.assertEqual(a["theta_hat_argmax"], b["theta_hat_argmax"])
        np.testing.assert_equal(wrap_180(np.array([5.-355., 355.-5.])), [10., -10.])


if __name__ == "__main__":
    unittest.main()
