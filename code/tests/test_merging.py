"""Run from the project root: python -m unittest discover -s code/tests -p test_merging.py -v"""

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not (Path(__file__).resolve().parents[1] / "merging_gwl_with_features.py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from merging_gwl_with_features import merge_groundwater_features, missingness_report


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2008-01-01", periods=36, freq="MS")
        frames = []
        for well, offset in [("A", 0), ("4123/53", 100)]:
            frames.append(pd.DataFrame({
                "Well_ID": well, "date": self.dates.strftime("%Y-%m"),
                "precipitation_mm": np.arange(36, dtype=float)+offset,
                "Snowfall_mm": 2.0, "ET_MODIS_mm": 3.0,
                "Runoff_SubSurface_mm": 4.0, "Runoff_Total_mm": 5.0,
                "NDVI_MODIS": .4, "NDVI_Landsat": .5, "LST_C": 20.0,
                "ET_MODIS_mm_coverage": 1.0, "LST_C_coverage": 1.0,
                "NDVI_MODIS_coverage": 1.0, "system:index": "export_id", ".geo": "{}",
            }))
        self.features = pd.concat(frames, ignore_index=True)
        panels = []
        for well in ["A", "4123/53"]:
            panels.append(pd.DataFrame({
                "well_id": well, "date": self.dates[12:],
                "level": [100.0, np.nan, np.nan]+[101.0]*21,
                "level_filled": [100.0, 100.5, np.nan]+[101.0]*21,
                "is_observed": [True, False, False]+[True]*21,
                "is_imputed": [False, True, False]+[False]*21,
                "imputation_method": ["observed", "kalman_prediction", "unfilled_gap_limit"]+["observed"]*21,
            }))
        self.groundwater = pd.concat(panels, ignore_index=True)

    def merge(self, groundwater=None, features=None):
        return merge_groundwater_features(self.groundwater if groundwater is None else groundwater,
                                          self.features if features is None else features)

    def test_groundwater_panel_and_provenance_are_preserved(self):
        result = self.merge()
        expected = self.groundwater.sort_values(["well_id", "date"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(result[expected.columns], expected)
        self.assertEqual(set(result.well_id), {"A", "4123/53"})
        self.assertNotIn("system:index", result)
        self.assertNotIn(".geo", result)
        self.assertNotIn("et_groundwater", result)

    def test_extra_feature_wells_and_dates_cannot_add_target_rows(self):
        extra = self.features.loc[self.features.Well_ID == "A"].copy()
        extra["Well_ID"] = "UNSELECTED"
        result = self.merge(features=pd.concat([self.features, extra], ignore_index=True))
        self.assertEqual(len(result), len(self.groundwater))
        self.assertNotIn("UNSELECTED", set(result.well_id))
        self.assertEqual(result.date.min(), self.groundwater.date.min())

    def test_rolling_uses_preceding_history_and_stays_within_well(self):
        result = self.merge()
        first = result.loc[result.date == pd.Timestamp("2009-01-01")].set_index("well_id")
        self.assertEqual(first.loc["A", "precipitation_sum_2m_mm"], 23)
        self.assertEqual(first.loc["4123/53", "precipitation_sum_2m_mm"], 223)
        self.assertEqual(first.loc["A", "ET_MODIS_sum_12m_mm"], 36)
        self.assertAlmostEqual(first.loc["A", "NDVI_MODIS_mean_12m"], .4)
        self.assertEqual(first.loc["A", "LST_mean_12m_C"], 20)

    def test_missing_history_month_is_not_skipped_by_rolling(self):
        features = self.features.drop(index=[11])  # Dec 2008, before the target panel
        result = self.merge(features=features)
        first = result.loc[(result.well_id == "A") & (result.date == self.dates[12])].iloc[0]
        self.assertTrue(pd.isna(first.precipitation_sum_2m_mm))
        self.assertTrue(pd.isna(first.ET_MODIS_sum_12m_mm))
        self.assertTrue(first.feature_row_available)

    def test_future_changes_cannot_change_earlier_features(self):
        original = self.merge()
        changed = self.features.copy()
        future = changed.date >= "2010-01"
        changed.loc[future, ["precipitation_mm", "LST_C", "ET_MODIS_mm"]] += 1000
        repeated = self.merge(features=changed)
        pd.testing.assert_frame_equal(original.loc[original.date < "2010-01-01"],
                                      repeated.loc[repeated.date < "2010-01-01"])

    def test_ndvi_fallback_keeps_raw_values_and_marks_source(self):
        features = self.features.copy()
        features.loc[12, "NDVI_Landsat"] = np.nan
        result = self.merge(features=features)
        first = result.loc[(result.well_id == "A") & (result.date == self.dates[12])].iloc[0]
        self.assertTrue(pd.isna(first.NDVI_Landsat))
        self.assertEqual(first.NDVI_combined, .4)
        self.assertEqual(first.NDVI_source, "MODIS")
        self.assertTrue(result.loc[result.NDVI_Landsat.notna(), "NDVI_source"].eq("Landsat").all())

    def test_missing_feature_values_are_retained_and_reported(self):
        features = self.features.copy()
        features.loc[features.Well_ID == "A", "ET_MODIS_mm"] = np.nan
        result = self.merge(features=features)
        self.assertTrue(result.loc[result.well_id == "A", "ET_MODIS_mm"].isna().all())
        report = missingness_report(result)
        missing = report.loc[(report.well_id == "A") & (report.column == "ET_MODIS_mm")].iloc[0]
        self.assertEqual(missing.missing_fraction, 1)

    def test_unmatched_keys_and_duplicate_keys_are_rejected(self):
        duplicate = pd.concat([self.features, self.features.iloc[[0]]], ignore_index=True)
        missing = self.features.drop(index=[12])
        duplicate_gwl = pd.concat([self.groundwater, self.groundwater.iloc[[0]]], ignore_index=True)
        for groundwater, features in [(self.groundwater, duplicate), (self.groundwater, missing),
                                      (duplicate_gwl, self.features)]:
            with self.subTest(rows=len(features)):
                with self.assertRaises(ValueError):
                    self.merge(groundwater, features)

    def test_ambiguous_names_collisions_and_nonmonthly_dates_are_rejected(self):
        ambiguous = self.features.assign(well_id=self.features.Well_ID)
        collision = self.features.assign(level=123)
        off_month = self.features.copy()
        off_month.loc[0, "date"] = "2008-01-15"
        for features in [ambiguous, collision, off_month]:
            with self.subTest(columns=len(features.columns)):
                with self.assertRaises(ValueError):
                    self.merge(features=features)

    def test_invalid_quality_flags_and_changed_observations_are_rejected(self):
        bad_coverage = self.features.copy()
        bad_coverage.loc[12, "LST_C_coverage"] = 5
        bad_mask = self.groundwater.copy()
        bad_mask.loc[0, "is_observed"] = False
        changed_observation = self.groundwater.copy()
        changed_observation.loc[0, "level_filled"] = 0
        for groundwater, features in [(self.groundwater, bad_coverage), (bad_mask, self.features),
                                      (changed_observation, self.features)]:
            with self.subTest(rows=len(groundwater)):
                with self.assertRaises(ValueError):
                    self.merge(groundwater, features)

    def test_shuffled_input_has_identical_results(self):
        original = self.merge()
        shuffled = self.merge(self.groundwater.sample(frac=1, random_state=3),
                              self.features.sample(frac=1, random_state=4))
        pd.testing.assert_frame_equal(original, shuffled)


if __name__ == "__main__":
    unittest.main()
