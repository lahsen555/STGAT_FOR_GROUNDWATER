"""Data-integrity and forecasting-boundary regression checks (no torch needed)."""

from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not (Path(__file__).resolve().parents[1] / "stgnn_data.py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from stgnn_data import prepare_data, resolve_imputation_cutoff, spatial_graph


def fixture(nodes=2, months=60):
    dates = pd.date_range("2000-01-01", periods=months, freq="MS")
    frames = []
    for node in range(nodes):
        level = 100 + node*100 + np.arange(months)*.1 + np.sin(np.arange(months)*2*np.pi/12)
        observed = np.ones(months, dtype=bool)
        observed[[15, 25, 40, 50]] = False
        raw = np.where(observed, level, np.nan)
        filled = level.copy()
        filled[25] = np.nan
        frames.append(pd.DataFrame({"well_id": f"W{node}", "date": dates,
            "level": raw, "level_filled": filled, "is_observed": observed,
            "is_imputed": ~observed & np.isfinite(filled), "precipitation_mm": np.arange(months)+node,
            "ET_MODIS_mm": 5.0 if node == 0 else np.nan, "ET_MODIS_mm_coverage": 1.0,
            "LST_C": np.nan, "NDVI_MODIS": .4, "Runoff_Total_mm": .2}))
    meta = pd.DataFrame({"Well_ID": [f"W{i}" for i in range(nodes)],
                         "latitude": 31+np.arange(nodes)*.2, "longitude": -8-np.arange(nodes)*.1})
    return pd.concat(frames, ignore_index=True), meta


class DataTests(unittest.TestCase):
    def prepare(self, frame=None, meta=None, **kwargs):
        source, metadata = fixture()
        return prepare_data(source if frame is None else frame, metadata if meta is None else meta,
                            features=["precipitation_mm", "ET_MODIS_mm", "LST_C"],
                            min_feature_coverage=.4, **kwargs)

    def test_missing_labels_masked_and_inputs_finite_with_availability(self):
        data = self.prepare()
        self.assertFalse(data.target_mask[15, 0])
        self.assertFalse(data.target_mask[25, 0])
        self.assertTrue(np.isfinite(data.inputs).all())
        channel = data.channels.index("level_filled_available")
        self.assertEqual(data.inputs[15, 0, channel], 1)
        self.assertEqual(data.inputs[25, 0, channel], 0)
        self.assertEqual(data.inputs[25, 0, data.channels.index("level_filled")], 0)
        self.assertNotIn("LST_C", data.features)
        self.assertEqual(data.inputs[30, 1, data.channels.index("ET_MODIS_mm_available")], 0)

    def test_holdout_changes_cannot_change_training_stats_or_earlier_inputs(self):
        frame, meta = fixture()
        original = self.prepare(frame, meta)
        changed = frame.copy()
        late = changed.date >= original.dates[45]
        changed.loc[late, ["level", "level_filled", "precipitation_mm", "ET_MODIS_mm", "LST_C"]] += 1000
        repeated = self.prepare(changed, meta)
        np.testing.assert_array_equal(original.value_mean, repeated.value_mean)
        np.testing.assert_array_equal(original.value_scale, repeated.value_scale)
        np.testing.assert_array_equal(original.inputs[:45], repeated.inputs[:45])
        self.assertEqual(original.features, repeated.features)
        self.assertEqual(original.feature_report, repeated.feature_report)

    def test_well_with_no_test_labels_keeps_training_and_graph_node(self):
        frame, meta = fixture()
        late = (frame.well_id == "W1") & (frame.date >= pd.Timestamp("2004-01-01"))
        frame.loc[late, ["level", "level_filled"]] = np.nan
        frame.loc[late, ["is_observed", "is_imputed"]] = False
        data = self.prepare(frame, meta)
        coverage = data.coverage().set_index(["split", "well_id"])
        self.assertGreater(coverage.loc[("train", "W1"), "measured_targets"], 0)
        self.assertEqual(coverage.loc[("test", "W1"), "measured_targets"], 0)
        self.assertIn("W1", data.wells)

    def test_multihorizon_targets_never_cross_split_boundaries(self):
        data = self.prepare(horizon=3)
        for split, lo, hi in [("train", 0, data.train_stop), ("val", data.train_stop, data.val_stop),
                               ("test", data.val_stop, len(data.dates))]:
            indices = data.target_indices(split)
            self.assertTrue((indices >= lo).all())
            self.assertTrue((indices < hi).all())
            self.assertTrue((data.starts[split]-data.lookback >= 0).all())

    def test_missing_calendar_row_is_reinserted_with_false_mask(self):
        frame, meta = fixture()
        frame = frame.loc[~((frame.well_id == "W0") & (frame.date == pd.Timestamp("2001-01-01")))]
        data = self.prepare(frame, meta)
        self.assertEqual(len(data.dates), 60)
        self.assertFalse(data.target_mask[12, 0])
        self.assertEqual(data.inputs[12, 0, data.channels.index("level_filled_available")], 0)

    def test_shuffled_keys_and_metadata_keep_identical_node_order(self):
        frame, meta = fixture()
        original = self.prepare(frame, meta)
        changed = self.prepare(frame.sample(frac=1, random_state=2), meta.iloc[::-1])
        self.assertEqual(original.wells, changed.wells)
        np.testing.assert_array_equal(original.inputs, changed.inputs)
        np.testing.assert_array_equal(original.coordinates, changed.coordinates)

    def test_bad_keys_flags_and_missing_coordinates_fail(self):
        frame, meta = fixture()
        duplicate = pd.concat([frame, frame.iloc[[0]]])
        bad_mask = frame.copy()
        bad_mask.loc[0, "is_observed"] = False
        bad_measurement = frame.copy()
        bad_measurement.loc[0, "level_filled"] += 10
        for candidate, coordinates in [(duplicate, meta), (bad_mask, meta), (bad_measurement, meta), (frame, meta.iloc[:1])]:
            with self.subTest(rows=len(candidate)):
                with self.assertRaises(ValueError):
                    self.prepare(candidate, coordinates)

    def test_geographic_graph_uses_km_symmetry_and_handles_single_node(self):
        coords = np.array([[0, 0], [0, 1], [0, 3]])
        edges, attr, distance = spatial_graph(coords, 1)
        pairs = set(map(tuple, edges.T))
        self.assertTrue(all((j, i) in pairs for i, j in pairs))
        self.assertTrue(np.isclose(distance, 111.195, atol=.01).any())
        self.assertTrue(((attr > 0) & (attr <= 1)).all())
        single_edges, single_attr, _ = spatial_graph(coords[:1], 1)
        self.assertEqual(single_edges.shape, (2, 0))
        self.assertEqual(single_attr.shape, (0, 1))

    def test_imputation_cutoff_aligns_default_and_rejects_earlier_model_split(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            summary = root / "summary.csv"
            pd.DataFrame({"well_id": ["W0"], "train_end": ["2015-08-01"]}).to_csv(summary, index=False)
            cutoff, details = resolve_imputation_cutoff(root/"merged.csv", summary, None)
            self.assertEqual(cutoff, "2015-08-01")
            self.assertTrue(details["imputation_summary_verified"])
            with self.assertRaisesRegex(ValueError, "Imputation used"):
                resolve_imputation_cutoff(root/"merged.csv", summary, "2015-07-01")


if __name__ == "__main__":
    unittest.main()
