"""Optimization regressions use synthetic data, never the existing benchmark."""
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimize_stgat import (OptimConfig, DataFactory, CompactForecast, causal_slopes, fit_compact,
                            forecast, persistence, calibration_quantiles, rows_for_predictions,
                            metrics_tables, grouped_importance, training_interval_scales, calibration_targets)
from test_stgnn_data import fixture

torch.set_num_threads(2)


class OptimizationTests(unittest.TestCase):
    def data(self, nodes=2):
        frame, meta = fixture(nodes=nodes)
        cfg = OptimConfig(hidden=8, heads=1, dropout=0., feature_set="core")
        factory = DataFactory(frame, meta)
        return factory.get(cfg, "2002-12-01", "2003-12-01"), cfg

    def test_robust_trends_use_only_past_measurements(self):
        values = np.arange(40, dtype=float)[:, None]*.2
        values[[3, 6, 10, 20]] = np.nan
        slopes, mask = causal_slopes(values, 12)
        changed = values.copy()
        changed[30:] += 1000
        repeated, _ = causal_slopes(changed, 12)
        np.testing.assert_allclose(slopes[:30], repeated[:30])
        self.assertAlmostEqual(slopes[25, 0], .2)
        self.assertFalse(mask[0, 0])

    def test_cv_inputs_ignore_precomputed_interpolation_values(self):
        frame, meta = fixture()
        changed = frame.copy()
        changed.loc[~changed.is_observed, "level_filled"] = 99999
        cfg = OptimConfig()
        original = DataFactory(frame, meta).get(cfg, "2002-12-01", "2003-12-01")
        repeated = DataFactory(changed, meta).get(cfg, "2002-12-01", "2003-12-01")
        np.testing.assert_array_equal(original.inputs, repeated.inputs)

    def test_features_and_statistics_do_not_learn_later_fold_measurements(self):
        frame, meta = fixture()
        cfg = OptimConfig()
        original = DataFactory(frame, meta).get(cfg, "2002-12-01", "2003-12-01")
        changed = frame.copy()
        late = changed.date >= "2003-06-01"
        changed.loc[late, ["level", "level_filled", "precipitation_mm"]] += 100
        repeated = DataFactory(changed, meta).get(cfg, "2002-12-01", "2003-12-01")
        np.testing.assert_array_equal(original.inputs[:41], repeated.inputs[:41])
        np.testing.assert_array_equal(original.target_mean, repeated.target_mean)
        np.testing.assert_array_equal(original.value_scale, repeated.value_scale)

    def test_compact_singleton_shapes_and_zero_residual_reference(self):
        data, cfg = self.data(nodes=1)
        cfg.trend_gain = 0
        x = torch.from_numpy(data.inputs[24:36][None])
        anchor = torch.tensor([[.4]])
        edges, attr = torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
        for kind in ["STGAT", "GRU", "LSTM"]:
            model = CompactForecast(kind, data, cfg)
            result = model(x, edges, attr, anchor)
            self.assertEqual(result.shape, (1, 6, 1))
            torch.testing.assert_close(result, anchor[:, None].expand(1, 6, 1))

    def test_tiny_optimization_training_and_joint_importance(self):
        data, cfg = self.data()
        model, edges, attr, history, best = fit_compact("STGAT", data, cfg, epochs=3, patience=3,
                                                       seed=13, device=torch.device("cpu"))
        self.assertLessEqual(best["score"], history.iloc[0].validation_skill_ratio)
        prediction = forecast(model, data, data.starts["val"], edges, attr, torch.device("cpu"))
        self.assertEqual(prediction.shape, (len(data.starts["val"]), 6, 2))
        importance = grouped_importance([(model, edges, attr)], data, torch.device("cpu"), repeats=2)
        self.assertIn("Groundwater history", set(importance.feature))
        self.assertTrue(np.isfinite(importance.increase_validation_skill_ratio).all())

    def test_metrics_mask_missing_labels_at_each_horizon(self):
        data, cfg = self.data()
        starts = data.starts["val"]
        predictions = persistence(data, starts)
        table = rows_for_predictions(data, starts, {"model": predictions}, period="val", common=True)
        first, _ = metrics_tables(table, ["model"])
        changed = table.copy()
        changed.loc[~changed.is_observed, "model"] = 1e9
        repeated, _ = metrics_tables(changed, ["model"])
        np.testing.assert_allclose(first.mae_m, repeated.mae_m)

    def test_interval_quantiles_use_only_provided_calibration_labels(self):
        truth = np.ones((12, 6, 2))
        prediction = np.zeros_like(truth)
        mask = np.ones_like(truth, dtype=bool)
        q, count = calibration_quantiles(truth, prediction, mask, np.ones((6, 2)))
        np.testing.assert_allclose(q, 1)
        self.assertEqual(count, [24]*6)
        truth[0] = np.nan
        mask[0] = False
        q, _ = calibration_quantiles(truth, prediction, mask, np.ones((6, 2)))
        np.testing.assert_allclose(q, 1)
        with self.assertRaises(ValueError):
            calibration_quantiles(truth, prediction, np.zeros_like(mask), np.ones((6, 2)))

    def test_interval_scale_has_finite_training_only_fallback(self):
        data, _ = self.data()
        data.measured = data.measured.copy()
        data.measured[:data.train_stop, 0] = np.nan
        scale = training_interval_scales(data)
        self.assertTrue(np.isfinite(scale).all())
        self.assertTrue((scale > 0).all())
        data.measured[data.train_stop:] = 1e6
        np.testing.assert_array_equal(scale, training_interval_scales(data))

    def test_direct_horizon_windows_stay_within_fold(self):
        frame, meta = fixture()
        for horizon in [1, 3, 6]:
            data = DataFactory(frame, meta, horizon=horizon).get(OptimConfig(), "2002-12-01", "2003-12-01")
            self.assertTrue((data.starts["train"]+horizon <= data.train_stop).all())
            self.assertTrue((data.starts["val"] >= data.train_stop).all())
            self.assertTrue((data.starts["val"]+horizon <= data.val_stop).all())

    def test_calibration_uses_late_measurements_without_crossing_into_benchmark(self):
        data = SimpleNamespace(measured=np.ones((30, 4)), target_mask=np.ones((30, 4), dtype=bool), horizon=6)
        data.target_mask[10:14] = False
        starts, truth, mask = calibration_targets(data, 10, 18)
        q, count = calibration_quantiles(truth, np.zeros_like(truth), mask, np.ones((6, 4)))
        self.assertEqual(count, [16, 16, 16, 16, 16, 12])
        data.measured[18:] = 1e6
        _, changed, changed_mask = calibration_targets(data, 10, 18)
        repeated, _ = calibration_quantiles(changed, np.zeros_like(changed), changed_mask, np.ones((6, 4)))
        np.testing.assert_array_equal(q, repeated)


if __name__ == "__main__":
    unittest.main()
