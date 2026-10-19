"""Regression checks for scientific data integrity and held-out causality.

Run: python -m unittest discover -s code/tests -v
"""

import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not (Path(__file__).resolve().parents[1] / "Interpolation.py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from Interpolation import seasonal_kalman_gapfill


class GapFillTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2009-01-01", periods=60, freq="MS")
        t = np.arange(60)
        rng = np.random.default_rng(24)
        self.frame = pd.DataFrame({
            "well_id": "A", "date": self.dates,
            "level": 100 + .07*t + 1.2*np.sin(2*np.pi*t/12) + rng.normal(0, .05, 60),
        })
        self.cutoff = self.dates[35]

    def fill(self, frame, **kwargs):
        return seasonal_kalman_gapfill(frame, train_end=self.cutoff, **kwargs)

    def test_observations_preserved_and_masks_correct(self):
        frame = self.frame.copy()
        frame.loc[[14, 15, 40, 41], "level"] = np.nan
        result = self.fill(frame)
        obs = frame["level"].notna()
        np.testing.assert_array_equal(result.loc[obs, "level_filled"], frame.loc[obs, "level"])
        np.testing.assert_array_equal(result["level"], frame["level"])
        self.assertEqual(int(result["is_imputed"].sum()), 4)
        self.assertTrue(result["level_filled"].notna().all())
        np.testing.assert_array_equal(result["is_observed"], obs)

    def test_absent_rows_are_returned_and_shuffled_input_aligns(self):
        frame = self.frame.drop(index=[14, 40]).sample(frac=1, random_state=2)
        result = self.fill(frame)
        self.assertEqual(len(result), 60)
        self.assertTrue(result.loc[[14, 40], "is_imputed"].all())
        pd.testing.assert_series_equal(result["date"], pd.Series(self.dates, name="date"))
        actual = result.set_index("date")["level"].dropna()
        expected = self.frame.drop(index=[14, 40]).set_index("date")["level"]
        pd.testing.assert_series_equal(actual, expected)

    def test_future_holdout_changes_cannot_change_earlier_fills(self):
        frame = self.frame.copy()
        frame.loc[[15, 40, 41, 42, 49], "level"] = np.nan
        original = self.fill(frame)
        changed = frame.copy()
        changed.loc[changed["date"] >= self.dates[45], "level"] += 500
        repeated = self.fill(changed)
        np.testing.assert_allclose(original.loc[:44, "level_filled"],
                                   repeated.loc[:44, "level_filled"], rtol=0, atol=1e-10, equal_nan=True)

    def test_leading_and_long_gaps_remain_flagged(self):
        frame = self.frame.copy()
        frame.loc[[0, 1, 40, 41, 42, 43, 44], "level"] = np.nan
        result = self.fill(frame, max_gap=3)
        self.assertTrue(result.loc[[0, 1, 43, 44], "level_filled"].isna().all())
        self.assertTrue(result.loc[[40, 41, 42], "is_imputed"].all())
        self.assertTrue(result.loc[[0, 1], "imputation_method"].eq("unfilled_leading").all())
        self.assertTrue(result.loc[[43, 44], "imputation_method"].eq("unfilled_gap_limit").all())

    def test_different_wells_do_not_share_values(self):
        other = self.frame.copy()
        other["well_id"] = "B"
        other["level"] += 1000
        frame = pd.concat([self.frame, other], ignore_index=True)
        frame.loc[frame["date"].isin(self.dates[[14, 40]]), "level"] = np.nan
        result = self.fill(frame)
        a = result.loc[result.well_id == "A", "level_filled"].to_numpy()
        b = result.loc[result.well_id == "B", "level_filled"].to_numpy()
        np.testing.assert_allclose(b-a, 1000, atol=1e-4)

    def test_insufficient_training_keeps_observations(self):
        frame = self.frame.copy()
        frame.loc[:29, "level"] = np.nan
        frame.loc[40, "level"] = np.nan
        with self.assertWarnsRegex(RuntimeWarning, "insufficient training"):
            result = self.fill(frame)
        observed = frame["level"].notna()
        np.testing.assert_array_equal(result.loc[observed, "level_filled"], frame.loc[observed, "level"])
        self.assertTrue(pd.isna(result.loc[40, "level_filled"]))

    def test_failed_and_nonconverged_fits_are_visible(self):
        frame = self.frame.copy()
        frame.loc[40, "level"] = np.nan
        for failed_fit in (RuntimeError("test failure"), SimpleNamespace(mle_retvals={"converged": False})):
            with self.subTest(failed_fit=failed_fit):
                with patch("Interpolation._make_model") as model:
                    if isinstance(failed_fit, Exception):
                        model.return_value.fit.side_effect = failed_fit
                    else:
                        model.return_value.fit.return_value = failed_fit
                    with self.assertWarnsRegex(RuntimeWarning, "gap filling failed"):
                        result = self.fill(frame)
                self.assertEqual(result.loc[40, "imputation_method"], "unfilled_model_failure")
                np.testing.assert_array_equal(result.loc[:39, "level_filled"], frame.loc[:39, "level"])

    def test_invalid_monthly_keys_and_levels_are_rejected(self):
        duplicate = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        off_month = self.frame.copy()
        off_month.loc[0, "date"] += pd.Timedelta(days=3)
        infinite = self.frame.copy()
        infinite.loc[0, "level"] = np.inf
        missing_id = self.frame.copy()
        missing_id.loc[0, "well_id"] = None
        for frame in [duplicate, off_month, infinite, missing_id, self.frame.iloc[:0]]:
            with self.subTest(frame_size=len(frame)):
                with self.assertRaises(ValueError):
                    self.fill(frame)

    def test_np_column_is_accepted_without_replacing_measurements(self):
        result = self.fill(self.frame.rename(columns={"level": "NP (m)"}))
        np.testing.assert_array_equal(result["level_filled"], self.frame["level"])


if __name__ == "__main__":
    unittest.main()
