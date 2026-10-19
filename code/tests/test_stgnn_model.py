"""Regression checks for masked training, graph batching, and evaluation."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if not (Path(__file__).resolve().parents[1] / "STGNN_model.py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from STGNN_model import (ModelConfig, STGNN_GAT_GRU, BaselineModel, STBlock, expand_edges,
                         masked_mse, fit_model, evaluate_predictions, predict, make_model,
                         save_checkpoint, baseline_predictions, run_epoch)
from stgnn_data import prepare_data
from test_stgnn_data import fixture

torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def test_masked_loss_ignores_missing_nan_labels_and_has_zero_gradient_there(self):
        prediction = torch.tensor([[[1., 100., 3.]]], requires_grad=True)
        truth = torch.tensor([[[2., float("nan"), 5.]]])
        mask = torch.tensor([[[True, False, True]]])
        loss = masked_mse(prediction, truth, mask)
        self.assertEqual(loss.item(), 2.5)
        loss.backward()
        self.assertEqual(prediction.grad[0, 0, 1].item(), 0)
        empty = masked_mse(prediction, truth, torch.zeros_like(mask))
        self.assertEqual(empty.item(), 0)
        with self.assertRaises(ValueError):
            masked_mse(prediction, truth.squeeze(0), mask)

    def test_expanded_edges_keep_graphs_isolated_and_attributes_aligned(self):
        edges = torch.tensor([[0, 1], [1, 0]])
        attr = torch.tensor([[.2], [.8]])
        expanded, attributes = expand_edges(edges, attr, copies=3, nodes=2)
        np.testing.assert_array_equal(expanded.numpy(), [[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]])
        np.testing.assert_array_equal(attributes.numpy(), attr.repeat(3, 1).numpy())
        self.assertTrue(torch.equal(expanded[0]//2, expanded[1]//2))

    def test_epoch_loss_weights_measured_points_and_handles_last_batch(self):
        class Reference(torch.nn.Module):
            def forward(self, x, edges, attributes, anchor):
                return anchor[:, None, :]
        dataset = torch.utils.data.TensorDataset(torch.zeros(3, 1, 1, 1),
            torch.tensor([[[1.]], [[2.]], [[3.]]]), torch.ones(3, 1, 1, dtype=torch.bool), torch.zeros(3, 1))
        loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        loss = run_epoch(Reference(), loader, torch.zeros(2, 0, dtype=torch.long),
                          torch.zeros(0, 1), torch.device("cpu"))
        self.assertAlmostEqual(loss, 14/3)

    def test_singleton_batch_node_and_multihorizon_shapes(self):
        cfg = ModelConfig(hidden_channels=8, node_emb_dim=4, gat_heads=1, dropout=0)
        for nodes in [1, 3]:
            for horizon in [1, 3]:
                x, anchor = torch.randn(1, 5, nodes, 4), torch.randn(1, nodes)
                edges, attr = torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
                models = [STGNN_GAT_GRU(nodes, 4, 5, horizon, cfg),
                          BaselineModel("GRU", nodes, 4, horizon, cfg),
                          BaselineModel("LSTM", nodes, 4, horizon, cfg)]
                for model in models:
                    result = model(x, edges, attr, anchor)
                    self.assertEqual(result.shape, (1, horizon, nodes))
                    torch.testing.assert_close(result, anchor[:, None, :].expand(-1, horizon, -1))

    def test_proximity_attributes_reach_attention_and_change_messages(self):
        torch.manual_seed(3)
        block = STBlock(8, ModelConfig(hidden_channels=8, gat_heads=2, dropout=0))
        block.eval()
        x = torch.randn(1, 3, 3, 8)
        edges = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]])
        a = block(x, edges, torch.tensor([[.1], [.5], [.9], [.2]]))
        b = block(x, edges, torch.zeros((4, 1)))
        self.assertGreater(torch.max(torch.abs(a-b)).item(), 1e-6)

    def test_metrics_include_only_measured_labels_and_keep_unscored_well(self):
        frame, meta = fixture()
        late = (frame.well_id == "W1") & (frame.date >= "2004-01-01")
        frame.loc[late, ["level", "level_filled"]] = np.nan
        frame.loc[late, ["is_observed", "is_imputed"]] = False
        data = prepare_data(frame, meta, features=["precipitation_mm"])
        indices = data.target_indices("test")
        prediction = np.nan_to_num(data.measured[indices], nan=-1e6)
        metrics, wells = evaluate_predictions(data, "test", "test", prediction)
        self.assertEqual(metrics["mae_m"], 0)
        self.assertEqual(wells[1]["n"], 0)
        self.assertTrue(np.isnan(wells[1]["mae_m"]))
        self.assertNotIn("r2", metrics)

    def test_tiny_training_checkpoint_reload_and_dated_horizon_predictions(self):
        frame, meta = fixture()
        data = prepare_data(frame, meta, features=["precipitation_mm"], horizon=2)
        cfg = ModelConfig(hidden_channels=8, node_emb_dim=4, gat_heads=1, dropout=0,
                           batch_size=7, k_neighbors=1)
        model, edges, attr, history, best = fit_model("STGAT", data, cfg, epochs=3, patience=3,
                                                     seed=12, device=torch.device("cpu"), verbose=False)
        self.assertLessEqual(best["best_val_scaled_mse"], history.iloc[0].val_scaled_mse)
        original = predict(model, data, "test", edges, attr, torch.device("cpu"))
        self.assertEqual(original.shape, (len(data.starts["test"]), 2, 2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"checkpoint.pt"
            save_checkpoint(path, model, "STGAT", cfg, data, edges, attr, best)
            saved = torch.load(path, weights_only=True)
            restored = make_model("STGAT", data, ModelConfig(**saved["config"]))
            restored.load_state_dict(saved["state_dict"])
            repeated = predict(restored, data, "test", saved["edge_index"], saved["edge_attr"], torch.device("cpu"))
            np.testing.assert_allclose(original, repeated)
        baselines = baseline_predictions(data, "test")
        self.assertEqual(baselines["Persistence"].shape, original.shape)


if __name__ == "__main__":
    unittest.main()
