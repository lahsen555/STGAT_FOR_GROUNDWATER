"""Masked STGAT forecasting, validation-only model selection, and measured-label evaluation.

Run from any directory. Defaults forecast one month ahead using 12 historical
months. Existing interpolation provenance sets the training cutoff when found.
All preprocessing statistics and feature-coverage decisions use training only.
Missing labels are never supervised/scored. This is a rolling-origin forecast:
earlier validation/test measurements become available for later forecast inputs.

Examples:
    python code/STGNN_model.py
    python code/STGNN_model.py --trials 20 --epochs 200
    python code/STGNN_model.py --prepare-only
    python code/STGNN_model.py --optimize --trials 24

The optional search tunes each neural comparator on validation, never test.
No arbitrary input-noise band is reported as a confidence/prediction interval.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.nn import GATv2Conv

from stgnn_data import DEFAULT_FEATURES, ForecastData, prepare_data, resolve_imputation_cutoff, spatial_graph


@dataclass
class ModelConfig:
    hidden_channels: int = 32
    num_blocks: int = 1
    num_gru_layers: int = 1
    gat_heads: int = 2
    dropout: float = .2
    node_emb_dim: int = 8
    k_neighbors: int = 5
    lr: float = .003
    weight_decay: float = .0005
    batch_size: int = 16


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


class ForecastDataset(Dataset):
    def __init__(self, data: ForecastData, split: str):
        self.data, self.starts = data, data.starts[split]

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = int(self.starts[index])
        d = self.data
        return (
            torch.from_numpy(d.inputs[start-d.lookback:start]),
            torch.from_numpy(d.targets[start:start+d.horizon]),
            torch.from_numpy(d.target_mask[start:start+d.horizon]),
            torch.from_numpy(d.anchor_z[start-1].astype(np.float32)),
        )


def loaders(data: ForecastData, batch_size: int, seed: int) -> dict[str, DataLoader]:
    generator = torch.Generator().manual_seed(seed)
    return {name: DataLoader(ForecastDataset(data, name), batch_size=batch_size,
                            shuffle=name == "train", generator=generator if name == "train" else None,
                            num_workers=0)
            for name in ["train", "val", "test"]}


def expand_edges(edge_index: torch.Tensor, edge_attr: torch.Tensor, copies: int,
                 nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph-major edge ordering matches flattened (batch, time, node) values."""
    offsets = torch.arange(copies, device=edge_index.device) * nodes
    edges = (edge_index[None, :, :] + offsets[:, None, None]).permute(1, 0, 2).reshape(2, -1)
    return edges, edge_attr.repeat(copies, 1)


class STBlock(nn.Module):
    def __init__(self, hidden: int, cfg: ModelConfig):
        super().__init__()
        self.norm_temporal = nn.LayerNorm(hidden)
        self.temporal = nn.GRU(hidden, hidden, num_layers=cfg.num_gru_layers, batch_first=True,
                               dropout=cfg.dropout if cfg.num_gru_layers > 1 else 0)
        self.norm_spatial = nn.LayerNorm(hidden)
        self.spatial = GATv2Conv(hidden, hidden, heads=cfg.gat_heads, concat=False,
                                dropout=cfg.dropout, edge_dim=1, add_self_loops=True, fill_value=1.0)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, edge_index, edge_attr):
        batch, steps, nodes, hidden = x.shape
        temporal = self.norm_temporal(x).permute(0, 2, 1, 3).reshape(batch*nodes, steps, hidden)
        temporal, _ = self.temporal(temporal)
        temporal = temporal.reshape(batch, nodes, steps, hidden).permute(0, 2, 1, 3)
        edges, attributes = expand_edges(edge_index, edge_attr, batch*steps, nodes)
        spatial = self.spatial(self.norm_spatial(temporal).reshape(batch*steps*nodes, hidden),
                               edges, edge_attr=attributes)
        spatial = spatial.reshape(batch, steps, nodes, hidden)
        return x + self.dropout(torch.nn.functional.gelu(spatial))


class STGNN_GAT_GRU(nn.Module):
    """Temporal GRU + spatial GATv2 blocks, with a residual persistence head."""
    def __init__(self, num_nodes: int, num_features: int, in_len: int, out_len: int, cfg: ModelConfig):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, cfg.node_emb_dim) * .05)
        self.input_proj = nn.Linear(num_features + cfg.node_emb_dim, cfg.hidden_channels)
        self.blocks = nn.ModuleList([STBlock(cfg.hidden_channels, cfg) for _ in range(cfg.num_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(cfg.hidden_channels),
                                  nn.Linear(cfg.hidden_channels, cfg.hidden_channels), nn.GELU(),
                                  nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_channels, out_len))
        # Epoch zero is an explicit persistence candidate. Learning estimates
        # corrections to the last measured head rather than a large absolute level.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, edge_index, edge_attr, anchor):
        batch, steps, nodes, _ = x.shape
        emb = self.node_emb[None, None, :, :].expand(batch, steps, nodes, -1)
        x = self.input_proj(torch.cat([x, emb], dim=-1))
        for block in self.blocks:
            x = block(x, edge_index, edge_attr)
        correction = self.head(x[:, -1]).transpose(1, 2)   # (B, horizon, nodes)
        return anchor[:, None, :] + correction


class BaselineModel(nn.Module):
    """Shared per-well GRU/LSTM with the same inputs, IDs and residual reference."""
    def __init__(self, kind: str, num_nodes: int, num_features: int, out_len: int, cfg: ModelConfig):
        super().__init__()
        self.node_emb = nn.Parameter(torch.randn(num_nodes, cfg.node_emb_dim) * .05)
        rnn = nn.LSTM if kind == "LSTM" else nn.GRU
        self.rnn = rnn(num_features + cfg.node_emb_dim, cfg.hidden_channels,
                       num_layers=cfg.num_gru_layers, batch_first=True,
                       dropout=cfg.dropout if cfg.num_gru_layers > 1 else 0)
        self.head = nn.Linear(cfg.hidden_channels, out_len)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, edge_index, edge_attr, anchor):
        batch, steps, nodes, channels = x.shape
        emb = self.node_emb[None, None].expand(batch, steps, nodes, -1)
        sequence = torch.cat([x, emb], dim=-1).permute(0, 2, 1, 3).reshape(batch*nodes, steps, -1)
        result, _ = self.rnn(sequence)
        correction = self.head(result[:, -1]).reshape(batch, nodes, -1).transpose(1, 2)
        return anchor[:, None, :] + correction


def make_model(kind: str, data: ForecastData, cfg: ModelConfig) -> nn.Module:
    if kind == "STGAT":
        return STGNN_GAT_GRU(len(data.wells), len(data.channels), data.lookback, data.horizon, cfg)
    if kind in {"GRU", "LSTM"}:
        return BaselineModel(kind, len(data.wells), len(data.channels), data.horizon, cfg)
    raise ValueError(f"Unknown model {kind}.")


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, target and mask shapes must match exactly.")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Model produced non-finite predictions.")
    if not bool(mask.any()):
        return prediction.sum() * 0
    error = prediction[mask] - target[mask]
    if not torch.isfinite(error).all():
        raise FloatingPointError("Non-finite measured targets.")
    return error.square().mean()


def run_epoch(model, loader, edges, attributes, device, optimizer=None) -> float:
    model.train(optimizer is not None)
    squared, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, target, mask, anchor in loader:
            x, target, mask, anchor = (item.to(device) for item in [x, target, mask, anchor])
            n = int(mask.sum().item())
            if not n:
                continue
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            output = model(x, edges, attributes, anchor)
            loss = masked_mse(output, target, mask)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
                optimizer.step()
            squared += loss.item() * n
            count += n
    if not count:
        raise ValueError("No measured targets in this loader.")
    return squared / count


def fit_model(kind: str, data: ForecastData, cfg: ModelConfig, *, epochs: int, patience: int,
              seed: int, device: torch.device, trial=None, verbose: bool = True):
    seed_everything(seed)
    model = make_model(kind, data, cfg).to(device)
    graph, attributes, _ = spatial_graph(data.coordinates, cfg.k_neighbors)
    edges = torch.as_tensor(graph, dtype=torch.long, device=device)
    attributes = torch.as_tensor(attributes, dtype=torch.float32, device=device)
    batches = loaders(data, cfg.batch_size, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=8)
    best = run_epoch(model, batches["val"], edges, attributes, device)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch, bad_epochs = 0, 0
    history = [{"epoch": 0, "train_scaled_mse": np.nan, "val_scaled_mse": best}]
    for epoch in range(1, epochs + 1):
        training = run_epoch(model, batches["train"], edges, attributes, device, optimizer)
        validation = run_epoch(model, batches["val"], edges, attributes, device)
        scheduler.step(validation)
        history.append({"epoch": epoch, "train_scaled_mse": training, "val_scaled_mse": validation})
        if validation < best - 1e-8:
            best, best_epoch, bad_epochs = validation, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        if verbose and (epoch == 1 or epoch % 10 == 0):
            print(f"{kind}: epoch {epoch:03d}, train={training:.5f}, val={validation:.5f}, best_epoch={best_epoch}", flush=True)
        if trial is not None:
            import optuna
            trial.report(best, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if bad_epochs >= patience:
            break
    model.load_state_dict(best_state)
    return model, edges, attributes, pd.DataFrame(history), {"best_epoch": best_epoch, "best_val_scaled_mse": best}


def tune_config(kind, data, base, *, trials, epochs, patience, seed, device, output) -> ModelConfig:
    if trials == 0:
        return base
    import optuna
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))

    def objective(trial):
        params = {
            "hidden_channels": trial.suggest_categorical("hidden_channels", [16, 32]),
            "num_gru_layers": trial.suggest_int("num_gru_layers", 1, 2),
            "dropout": trial.suggest_categorical("dropout", [0.0, .1, .2]),
            "node_emb_dim": trial.suggest_categorical("node_emb_dim", [4, 8]),
            "lr": trial.suggest_float("lr", .0003, .005, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, .001, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
        }
        if kind == "STGAT":
            params.update({"num_blocks": trial.suggest_int("num_blocks", 1, 2),
                           "gat_heads": trial.suggest_categorical("gat_heads", [1, 2]),
                           "k_neighbors": trial.suggest_int("k_neighbors", 2, min(8, max(2, len(data.wells)-1)))})
        cfg = replace(base, **params)
        _, _, _, _, best = fit_model(kind, data, cfg, epochs=epochs, patience=patience,
                                     seed=seed, device=device, trial=trial, verbose=False)
        return best["best_val_scaled_mse"]

    study.optimize(objective, n_trials=trials)
    study.trials_dataframe().to_csv(output / f"{kind}_search.csv", index=False)
    return replace(base, **study.best_params)


def predict(model, data, split, edges, attributes, device, batch_size=16) -> np.ndarray:
    model.eval()
    batches = DataLoader(ForecastDataset(data, split), batch_size=batch_size, shuffle=False)
    predictions = []
    with torch.no_grad():
        for x, _, _, anchor in batches:
            output = model(x.to(device), edges, attributes, anchor.to(device))
            predictions.append(output.cpu().numpy())
    z = np.concatenate(predictions, axis=0)
    return z * data.target_scale[None, None, :] + data.target_mean[None, None, :]


def baseline_predictions(data: ForecastData, split: str) -> dict[str, np.ndarray]:
    starts = data.starts[split]
    anchor = np.where(np.isfinite(data.anchor_m[starts-1]), data.anchor_m[starts-1], data.target_mean)
    # Only before a well's first measurement, use its training mean as the
    # explicitly documented fallback. No validation/test observation is borrowed.
    persistence = np.repeat(anchor[:, None, :], data.horizon, axis=1)
    seasonal = persistence.copy()
    for row, start in enumerate(starts):
        for lead in range(data.horizon):
            prior = start + lead - 12
            if 0 <= prior < start:
                reference = data.measured[prior]
                seasonal[row, lead] = np.where(np.isfinite(reference), reference, persistence[row, lead])
    return {"Persistence": persistence, "Seasonal_or_persistence": seasonal}


def _point_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict:
    if not len(truth):
        return {"n": 0, "mae_m": np.nan, "rmse_m": np.nan, "bias_m": np.nan, "r2": np.nan}
    errors = prediction - truth
    variance = float(np.square(truth - truth.mean()).sum())
    return {"n": len(truth), "mae_m": float(np.abs(errors).mean()),
            "rmse_m": float(np.sqrt(np.square(errors).mean())), "bias_m": float(errors.mean()),
            "r2": 1-float(np.square(errors).sum())/variance if len(truth) >= 2 and variance > 1e-10 else np.nan}


def evaluate_predictions(data, split, name, predictions) -> tuple[dict, list[dict]]:
    indices = data.target_indices(split)
    truth, mask = data.measured[indices], data.target_mask[indices]
    if predictions.shape != truth.shape:
        raise ValueError("Prediction/target shapes differ.")
    if not np.isfinite(predictions[mask]).all():
        raise FloatingPointError("Non-finite forecasts at measured targets; do not silently exclude failed forecasts.")
    per_well = []
    for node, well in enumerate(data.wells):
        valid = mask[:, :, node]
        metrics = _point_metrics(truth[:, :, node][valid], predictions[:, :, node][valid])
        per_well.append({"split": split, "model": name, "well_id": well, **metrics})
    pooled = _point_metrics(truth[mask], predictions[mask])
    pooled.pop("r2")  # Absolute elevation differences can make pooled R2 misleading.
    valid_wells = pd.DataFrame(per_well).loc[lambda rows: rows.n > 0]
    overall = {"split": split, "model": name, **pooled, "scored_wells": len(valid_wells),
               "macro_mae_m": float(valid_wells.mae_m.mean()), "macro_rmse_m": float(valid_wells.rmse_m.mean()),
               "macro_r2": float(valid_wells.r2.mean()) if valid_wells.r2.notna().any() else np.nan}
    return overall, per_well


def prediction_table(data, split, predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for i, start in enumerate(data.starts[split]):
        for lead in range(data.horizon):
            target = start + lead
            for node, well in enumerate(data.wells):
                row = {"split": split, "origin_month": data.dates[start-1].strftime("%Y-%m-%d"),
                       "target_month": data.dates[target].strftime("%Y-%m-%d"), "lead_months": lead+1,
                       "well_id": well, "is_observed_target": bool(data.target_mask[target, node]),
                       "observed_level_m": data.measured[target, node],
                       "last_measurement_age_at_origin_months": data.age[start-1, node]}
                for name, values in predictions.items():
                    row[name + "_prediction_m"] = values[i, lead, node]
                rows.append(row)
    return pd.DataFrame(rows)


def save_checkpoint(path, model, kind, cfg, data, edges, attributes, best):
    torch.save({"state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                "model_kind": kind, "config": asdict(cfg), "preprocessing": data.preprocessing(),
                "edge_index": edges.detach().cpu(), "edge_attr": attributes.detach().cpu(), "selection": best}, path)


def plot_results(output, histories, predictions: pd.DataFrame, data, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for kind, history in histories.items():
        axes[0].plot(history.epoch, history.val_scaled_mse, label=kind)
    axes[0].set(xlabel="Epoch", ylabel="Validation MSE (standardized)")
    axes[0].legend()
    test = metrics.loc[metrics.split == "test"]
    axes[1].bar(test.model, test.mae_m, color=["#546270"]*2 + ["#2176ae", "#3b9d77", "#dc933a"][:len(test)-2])
    axes[1].set(ylabel="Test MAE (m), measured targets")
    axes[1].tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output / "training_and_metrics.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(int(np.ceil(len(data.wells)/4)), 4, figsize=(18, 3*np.ceil(len(data.wells)/4)), squeeze=False)
    for ax, well in zip(axes.flat, data.wells):
        rows = predictions.loc[(predictions.well_id == well) & (predictions.lead_months == 1)]
        dates = pd.to_datetime(rows.target_month)
        ax.plot(dates, rows.STGAT_prediction_m, color="#2176ae", linewidth=1, label="STGAT")
        ax.plot(dates, rows.Persistence_prediction_m, color="#d28d37", linewidth=.8, alpha=.8, label="Persistence")
        ax.plot(dates, rows.observed_level_m, ".", color="black", markersize=3, label="Measured targets")
        ax.axvline(data.dates[data.train_stop], color="#555555", linestyle="--", linewidth=.7)
        ax.axvline(data.dates[data.val_stop], color="#a43558", linestyle="--", linewidth=.7)
        ax.set_title(well, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(data.wells):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(output / "forecast_trajectories.png", dpi=150)
    plt.close(fig)


def main():
    if "--optimize" in sys.argv[1:]:
        from optimize_stgat import main as optimize_main
        optimize_main([arg for arg in sys.argv[1:] if arg != "--optimize"])
        return
    code_dir = Path(__file__).resolve().parent
    data_dir = code_dir.parent / "data"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--optimize", action="store_true", help="Use compact residual models, chronological CV, and all notebook plots; see --optimize --help.")
    parser.add_argument("--input", type=Path, default=data_dir / "gwl_and_features_advanced_sumed.csv")
    parser.add_argument("--metadata", type=Path, default=data_dir / "wells_meta.csv")
    parser.add_argument("--imputation-summary", type=Path, default=data_dir / "final_filled_summary.csv")
    parser.add_argument("--output", type=Path, help="A new directory; existing directories are never overwritten.")
    parser.add_argument("--train-end", help="Defaults to verified interpolation fit end, otherwise the first 60%.")
    parser.add_argument("--lookback", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--trials", type=int, default=0, help="Optional validation-only Optuna trials per neural model.")
    parser.add_argument("--hpo-epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--min-feature-coverage", type=float, default=.8)
    parser.add_argument("--features", nargs="+", default=DEFAULT_FEATURES)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.hpo_epochs, args.threads) < 1 or args.trials < 0:
        parser.error("Epochs, patience, HPO epochs and threads must be positive; trials cannot be negative.")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                           "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this environment.")
    cutoff, provenance = resolve_imputation_cutoff(args.input, args.imputation_summary, args.train_end)
    data = prepare_data(pd.read_csv(args.input), pd.read_csv(args.metadata), lookback=args.lookback,
                        horizon=args.horizon, train_end=cutoff, features=args.features,
                        min_feature_coverage=args.min_feature_coverage)
    if not provenance["imputation_summary_verified"]:
        print("No interpolation summary verified: check that imputation did not use held-out months.", flush=True)
    output = args.output or code_dir / "model_runs" / datetime.now(timezone.utc).strftime("stgat_%Y%m%d_%H%M%S_%f")
    if output.exists():
        parser.error(f"Output directory already exists; choose a new path: {output}")
    output.mkdir(parents=True)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for filename in ["STGNN_model.py", "stgnn_data.py"]:
        (snapshot / filename).write_bytes((code_dir / filename).read_bytes())
    preprocessing = data.preprocessing()
    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    data.coverage().to_csv(output / "label_coverage.csv", index=False)
    pd.DataFrame(data.feature_report).to_csv(output / "feature_coverage.csv", index=False)
    print(f"{len(data.wells)} wells; train through {preprocessing['train_end']}; validation through "
          f"{preprocessing['validation_end']}; test starts {data.dates[data.val_stop]:%Y-%m-%d}", flush=True)
    print(pd.DataFrame(data.feature_report).to_string(index=False), flush=True)
    print(f"Measured labels: {data.coverage().groupby('split').measured_targets.sum().to_dict()}", flush=True)
    print(f"Device: {device}; output: {output}", flush=True)
    manifest = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                "preprocessing": preprocessing, "provenance": provenance,
                "input_sha256": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [args.input, args.metadata]},
                "code_sha256": {name: hashlib.sha256((code_dir/name).read_bytes()).hexdigest()
                                for name in ["STGNN_model.py", "stgnn_data.py"]},
                "source_snapshot_directory": "source_snapshot",
                "runtime_versions": {package: version(package) for package in ["torch", "torch-geometric", "numpy", "pandas"]},
                "device": str(device), "selection_criterion": "measured-label validation standardized MSE",
                "test_used_for_selection": False, "models": {}}
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.prepare_only:
        return
    models, histories = {}, {}
    for kind in ["STGAT", "GRU", "LSTM"]:
        cfg = tune_config(kind, data, ModelConfig(), trials=args.trials, epochs=args.hpo_epochs,
                          patience=min(args.patience, 15), seed=args.seed, device=device, output=output)
        model, edges, attributes, history, best = fit_model(
            kind, data, cfg, epochs=args.epochs, patience=args.patience, seed=args.seed, device=device)
        checkpoint = output / f"{kind}.pt"
        save_checkpoint(checkpoint, model, kind, cfg, data, edges, attributes, best)
        # Test checkpoint reconstruction before generating any test performance.
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        restored = make_model(kind, data, ModelConfig(**saved["config"])).to(device)
        restored.load_state_dict(saved["state_dict"])
        expected = predict(model, data, "val", edges, attributes, device)
        actual = predict(restored, data, "val", saved["edge_index"], saved["edge_attr"], device)
        np.testing.assert_allclose(expected, actual, rtol=1e-6, atol=1e-6)
        models[kind] = (restored, saved["edge_index"], saved["edge_attr"])
        histories[kind] = history
        history.to_csv(output / f"{kind}_history.csv", index=False)
        manifest["models"][kind] = {"config": asdict(cfg), **best}
        print(f"{kind}: restored checkpoint from epoch {best['best_epoch']}.", flush=True)
    # Selection is complete for every model before this final evaluation block.
    overall, per_well, tables = [], [], []
    for split in ["train", "val", "test"]:
        predictions = baseline_predictions(data, split)
        for kind, (model, edges, attributes) in models.items():
            predictions[kind] = predict(model, data, split, edges, attributes, device)
        for kind, values in predictions.items():
            aggregate, wells = evaluate_predictions(data, split, kind, values)
            overall.append(aggregate)
            per_well.extend(wells)
        tables.append(prediction_table(data, split, predictions))
    metrics = pd.DataFrame(overall)
    all_predictions = pd.concat(tables, ignore_index=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(per_well).to_csv(output / "metrics_per_well.csv", index=False)
    all_predictions.to_csv(output / "predictions.csv", index=False)
    if not args.no_plots:
        plot_results(output, histories, all_predictions, data, metrics)
    manifest["complete"] = True
    (output / "run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.loc[metrics.split == "test"].to_string(index=False), flush=True)
    print(f"Saved checkpoints, measured-label metrics and dated predictions in {output}", flush=True)


if __name__ == "__main__":
    main()
