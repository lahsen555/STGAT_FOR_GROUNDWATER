"""Compact STGAT optimization, measured-label rolling forecasts, and full diagnostics.

Uses observed-only groundwater history for honest inner chronological folds.
The previously examined Nov 2017-Jan 2020 period is an EXISTING benchmark,
not a new untouched test set. No benchmark scores are used in model selection.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
import optuna

from stgnn_data import prepare_data, spatial_graph, resolve_imputation_cutoff
from STGNN_model import seed_everything, _point_metrics


FEATURE_SETS = {
    "levels": [],
    "core": ["ET_MODIS_mm", "NDVI_MODIS", "Runoff_Total_mm", "precipitation_mm"],
    "hydro": ["ET_MODIS_mm", "NDVI_MODIS", "precipitation_mm", "precipitation_sum_2m_mm",
              "ET_MODIS_sum_12m_mm", "Runoff_SubSurface_sum_12m_mm", "SoilMoisture_kgm2"],
    "expanded": ["ET_MODIS_mm", "NDVI_MODIS", "Runoff_Total_mm", "precipitation_mm",
                 "precipitation_sum_2m_mm", "SoilMoisture_kgm2", "SoilTemp_K", "PDSI", "TotalEvaporation_mm"],
}
LEADS = [1, 3, 6]


@dataclass
class OptimConfig:
    lookback: int = 12
    hidden: int = 16
    layers: int = 1
    heads: int = 2
    dropout: float = .1
    embedding: int = 4
    k: int = 4
    lr: float = .001
    weight_decay: float = .001
    batch_size: int = 16
    loss: str = "huber"
    feature_set: str = "core"
    trend_window: int = 12
    trend_gain: float = .25
    residual_gain: float = .25
    graph_gate: float = .2


def observed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["level_filled"] = result["level"]
    result["is_imputed"] = False
    return result


def causal_slopes(measured: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Median pairwise slope of measured heads in a trailing calendar window."""
    values, valid = np.zeros_like(measured), np.zeros_like(measured, dtype=bool)
    for t in range(len(measured)):
        lo = max(0, t-window+1)
        for node in range(measured.shape[1]):
            times = np.flatnonzero(np.isfinite(measured[lo:t+1, node])) + lo
            if len(times) < 2:
                continue
            i, j = np.triu_indices(len(times), 1)
            slope = (measured[times[j], node]-measured[times[i], node]) / (times[j]-times[i])
            values[t, node] = np.median(slope)
            valid[t, node] = True
    return values, valid


class DataFactory:
    def __init__(self, frame, metadata, horizon=6):
        self.frame, self.metadata, self.horizon = observed_frame(frame), metadata, horizon
        self.cache = {}
        self.slopes = None

    def get(self, cfg: OptimConfig, train_end: str, val_end: str):
        key = (cfg.lookback, cfg.feature_set, train_end, val_end)
        if key in self.cache:
            return self.cache[key]
        data = prepare_data(self.frame, self.metadata, lookback=cfg.lookback, horizon=self.horizon,
                            train_end=train_end, features=FEATURE_SETS[cfg.feature_set])
        if self.slopes is None:
            self.slopes = {window: causal_slopes(data.measured, window) for window in [6, 12, 24]}
        extra, names = [], []
        for window, (slope, valid) in self.slopes.items():
            extra += [(slope/data.target_scale)[..., None], valid[..., None].astype(float)]
            names += [f"gwl_slope_{window}m_z", f"gwl_slope_{window}m_available"]
        dense = np.where(np.isfinite(data.anchor_m), data.anchor_m, data.target_mean)
        for lag in [1, 3, 6, 12]:
            delta = np.zeros_like(dense)
            delta[lag:] = (dense[lag:]-dense[:-lag]) / lag / data.target_scale
            extra.append(delta[..., None])
            names.append(f"gwl_change_rate_{lag}m_z")
        data.inputs = np.concatenate([data.inputs, *extra], axis=2).astype(np.float32)
        data.channels += names
        val_stop = data.dates.get_loc(pd.Timestamp(val_end)) + 1
        if not data.train_stop < val_stop <= len(data.dates):
            raise ValueError("Invalid chronological validation end.")
        data.val_stop = val_stop
        # Equal training dates across lookbacks make their validation comparison fair.
        data.starts = {
            "train": np.arange(max(24, cfg.lookback), data.train_stop-self.horizon+1),
            "val": np.arange(data.train_stop, val_stop-self.horizon+1),
        }
        for split in ["train", "val"]:
            if not len(data.starts[split]):
                raise ValueError("Insufficient windows for this fold/horizon.")
        self.cache[key] = data
        return data


def batch_inputs(data, starts):
    x = np.stack([data.inputs[s-data.lookback:s] for s in starts])
    anchor = data.anchor_z[starts-1].astype(np.float32)
    indices = starts[:, None] + np.arange(data.horizon)
    target = data.targets[indices]
    mask = data.target_mask[indices]
    return [torch.as_tensor(a) for a in [x, target, mask, anchor]]


class CompactForecast(nn.Module):
    """Small temporal encoder, gated spatial attention, and a causal trend skip."""
    def __init__(self, kind, data, cfg: OptimConfig):
        super().__init__()
        self.cfg, self.kind, self.horizon = cfg, kind, data.horizon
        self.slope_index = data.channels.index(f"gwl_slope_{cfg.trend_window}m_z")
        self.age_index = data.channels.index("gwl_measurement_age_capped_48m")
        self.node_emb = nn.Parameter(torch.randn(len(data.wells), cfg.embedding)*.02)
        recurrent = nn.LSTM if kind == "LSTM" else nn.GRU
        self.temporal = recurrent(len(data.channels)+cfg.embedding, cfg.hidden, num_layers=cfg.layers,
                                 batch_first=True, dropout=cfg.dropout if cfg.layers > 1 else 0)
        self.norm = nn.LayerNorm(cfg.hidden)
        if kind == "STGAT":
            self.spatial = GATv2Conv(cfg.hidden, cfg.hidden, heads=cfg.heads, concat=False,
                                     edge_dim=1, fill_value=1., dropout=cfg.dropout)
            gate = np.clip(cfg.graph_gate, .001, .999)
            self.gate_logit = nn.Parameter(torch.tensor(float(np.log(gate/(1-gate)))))
        self.readout = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden, data.horizon))
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def forward(self, x, edge_index, edge_attr, anchor):
        batch, steps, nodes, _ = x.shape
        emb = self.node_emb[None, None].expand(batch, steps, nodes, -1)
        sequence = torch.cat([x, emb], -1).permute(0, 2, 1, 3).reshape(batch*nodes, steps, -1)
        encoded, _ = self.temporal(sequence)
        hidden = self.norm(encoded[:, -1])
        if self.kind == "STGAT":
            offsets = torch.arange(batch, device=x.device)*nodes
            edges = (edge_index[None]+offsets[:, None, None]).permute(1, 0, 2).reshape(2, -1)
            spatial = self.spatial(hidden, edges, edge_attr=edge_attr.repeat(batch, 1))
            hidden = hidden + torch.sigmoid(self.gate_logit)*torch.tanh(spatial)
        correction = self.readout(hidden).reshape(batch, nodes, self.horizon).transpose(1, 2)
        slope = x[:, -1, :, self.slope_index]
        age = x[:, -1, :, self.age_index]*48
        lead = torch.arange(1, self.horizon+1, device=x.device, dtype=x.dtype)
        drift = self.cfg.trend_gain
        trend = drift*slope[:, None, :]*(lead[None, :, None]+age[:, None, :].clamp(max=12))
        return anchor[:, None, :] + trend + self.cfg.residual_gain*correction


def persistence(data, starts):
    anchor = np.where(np.isfinite(data.anchor_m[starts-1]), data.anchor_m[starts-1], data.target_mean)
    return np.repeat(anchor[:, None, :], data.horizon, axis=1)


def forecast(model, data, starts, edges, attributes, device, batch=32):
    model.eval()
    values = []
    with torch.no_grad():
        for lo in range(0, len(starts), batch):
            ss = starts[lo:lo+batch]
            x = torch.from_numpy(np.stack([data.inputs[s-data.lookback:s] for s in ss])).to(device)
            anchor = torch.as_tensor(data.anchor_z[ss-1], dtype=torch.float32, device=device)
            pred = model(x, edges, attributes, anchor).cpu().numpy()
            values.append(pred*data.target_scale[None, None]+data.target_mean[None, None])
    return np.concatenate(values)


def skill_score(data, starts, predictions):
    indices = starts[:, None]+np.arange(data.horizon)
    truth, mask = data.measured[indices], data.target_mask[indices]
    reference = persistence(data, starts)
    ratios, metrics = [], []
    for lead in LEADS:
        if lead > data.horizon:
            continue
        valid = mask[:, lead-1]
        if not valid.any():
            raise ValueError(f"No measured validation labels at lead {lead}.")
        if not np.isfinite(predictions[:, lead-1][valid]).all():
            raise FloatingPointError("Non-finite forecast at a measured validation label.")
        mae = np.abs(predictions[:, lead-1][valid]-truth[:, lead-1][valid]).mean()
        base = np.abs(reference[:, lead-1][valid]-truth[:, lead-1][valid]).mean()
        ratios.append(mae/max(base, .01))
        metrics.append({"lead": lead, "mae_m": float(mae), "persistence_mae_m": float(base)})
    return float(np.mean(ratios)), metrics


def fit_compact(kind, data, cfg, *, epochs, patience, seed, device):
    seed_everything(seed)
    model = CompactForecast(kind, data, cfg).to(device)
    edges, attrs, _ = spatial_graph(data.coordinates, cfg.k)
    edges = torch.as_tensor(edges, device=device)
    attrs = torch.as_tensor(attrs, device=device)
    train = batch_inputs(data, data.starts["train"])
    train = [value.to(device) for value in train]
    idx = data.starts["train"][:, None]+np.arange(data.horizon)
    error = np.abs(data.measured[idx]-persistence(data, data.starts["train"]))
    reference = np.maximum(.05, np.nanmean(error, axis=(0, 2))).astype(np.float32)
    reference = torch.as_tensor(reference[None, :, None], device=device)
    scale = torch.as_tensor(data.target_scale[None, None], dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=6)
    starts = data.starts["val"]
    initial = forecast(model, data, starts, edges, attrs, device)
    best, _ = skill_score(data, starts, initial)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = [{"epoch": 0, "training_loss": np.nan, "validation_skill_ratio": best}]
    bad, best_epoch = 0, 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs+1):
        model.train()
        order = torch.randperm(len(train[0]), generator=generator).to(device)
        total, count = 0., 0
        for lo in range(0, len(order), cfg.batch_size):
            rows = order[lo:lo+cfg.batch_size]
            x, y, mask, anchor = [value[rows] for value in train]
            if not mask.any():
                continue
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, edges, attrs, anchor)
            error = ((pred-y)*scale/reference)[mask]
            if not torch.isfinite(error).all():
                raise FloatingPointError("Non-finite training error.")
            if cfg.loss == "l1":
                loss = error.abs().mean()
            elif cfg.loss == "mse":
                loss = error.square().mean()
            else:
                loss = nn.functional.smooth_l1_loss(error, torch.zeros_like(error), beta=.5)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2., error_if_nonfinite=True)
            optimizer.step()
            total += loss.item()*int(mask.sum())
            count += int(mask.sum())
        pred = forecast(model, data, starts, edges, attrs, device)
        score, _ = skill_score(data, starts, pred)
        scheduler.step(score)
        history.append({"epoch": epoch, "training_loss": total/max(count, 1), "validation_skill_ratio": score})
        if score < best-1e-5:
            best, best_epoch, bad = score, epoch, 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(state)
    return model, edges, attrs, pd.DataFrame(history), {"score": best, "epoch": best_epoch}


def suggest(trial, kind):
    cfg = OptimConfig(
        lookback=trial.suggest_categorical("lookback", [6, 12, 24]),
        hidden=trial.suggest_categorical("hidden", [8, 16, 24, 32]),
        layers=trial.suggest_int("layers", 1, 2),
        dropout=trial.suggest_categorical("dropout", [0., .1, .25]),
        embedding=trial.suggest_categorical("embedding", [2, 4, 8]),
        lr=trial.suggest_float("lr", .0002, .006, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, .05, log=True),
        batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
        loss=trial.suggest_categorical("loss", ["l1", "huber", "mse"]),
        feature_set=trial.suggest_categorical("feature_set", list(FEATURE_SETS)),
        trend_window=trial.suggest_categorical("trend_window", [6, 12, 24]),
        trend_gain=trial.suggest_categorical("trend_gain", [0., .25, .5, 1.]),
        residual_gain=trial.suggest_categorical("residual_gain", [.1, .25, 1.]),
    )
    if kind == "STGAT":
        cfg.heads = trial.suggest_categorical("heads", [1, 2, 4])
        cfg.k = trial.suggest_int("k", 2, 8)
        cfg.graph_gate = trial.suggest_categorical("graph_gate", [.05, .2, .5])
    return cfg


def search(kind, factory, folds, *, trials, epochs, seed, device, output):
    database = (output/"studies.sqlite3").resolve().as_posix()
    study = optuna.create_study(study_name=kind, storage=f"sqlite:///{database}", direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    for gain in [0., .25, .5, 1.]:
        initial = asdict(OptimConfig(hidden=8, dropout=.1, trend_gain=gain, residual_gain=.1, loss="l1"))
        for key in (["heads", "k", "graph_gate"] if kind != "STGAT" else []):
            initial.pop(key)
        study.enqueue_trial(initial)
    fold_rows = []

    def objective(trial):
        cfg = suggest(trial, kind)
        values = []
        for fold, (train_end, val_end) in enumerate(folds):
            data = factory.get(cfg, train_end, val_end)
            _, _, _, _, best = fit_compact(kind, data, cfg, epochs=epochs, patience=12,
                                           seed=seed, device=device)
            values.append(best["score"])
            fold_rows.append({"model": kind, "trial": trial.number, "fold": fold,
                              "train_end": train_end, "val_end": val_end, **best})
        trial.set_user_attr("config", asdict(cfg))
        trial.set_user_attr("fold_scores", values)
        print(f"{kind} trial {trial.number+1}/{trials}: CV skill ratio={np.mean(values):.4f}", flush=True)
        return float(np.mean(values))

    study.optimize(objective, n_trials=trials)
    study.trials_dataframe().to_csv(output/f"{kind}_search.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output/f"{kind}_folds.csv", index=False)
    return OptimConfig(**study.best_trial.user_attrs["config"]), study


def select_trend_baseline(factory, folds, output):
    rows = []
    cfg = OptimConfig(feature_set="levels")
    for window in [6, 12, 24]:
        for gain in [0., .1, .25, .5, .75, 1.]:
            scores = []
            for train_end, val_end in folds:
                data = factory.get(cfg, train_end, val_end)
                starts = data.starts["val"]
                slope = data.inputs[starts-1, :, data.channels.index(f"gwl_slope_{window}m_z")]*data.target_scale
                age = np.nan_to_num(data.age[starts-1], nan=0).clip(max=12)
                pred = persistence(data, starts)+gain*slope[:, None]*(np.arange(1, data.horizon+1)[None, :, None]+age[:, None])
                scores.append(skill_score(data, starts, pred)[0])
            rows.append({"trend_window": window, "trend_gain": gain, "cv_ratio": float(np.mean(scores))})
    table = pd.DataFrame(rows).sort_values("cv_ratio")
    table.to_csv(output/"trend_baseline_search.csv", index=False)
    return table.iloc[0].to_dict()


def rows_for_predictions(data, starts, predictions, *, period="all", common=False):
    rows = []
    for row, start in enumerate(starts):
        for lead in range(1, data.horizon+1):
            target = start+lead-1
            if target >= len(data.dates):
                continue
            for node, well in enumerate(data.wells):
                item = {"period": period, "common_origins": common, "origin_month": str(data.dates[start-1].date()),
                        "target_month": str(data.dates[target].date()), "lead_months": lead, "well_id": well,
                        "is_observed": bool(data.target_mask[target, node]), "observed_m": data.measured[target, node]}
                for name, values in predictions.items():
                    item[name] = values[row, lead-1, node]
                rows.append(item)
    return pd.DataFrame(rows)


def metrics_tables(predictions, models):
    overall, per_well = [], []
    for (period, common, lead), group in predictions.groupby(["period", "common_origins", "lead_months"]):
        valid = group.loc[group.is_observed]
        for model in models:
            local = []
            for well, rows in group.groupby("well_id"):
                rows = rows.loc[rows.is_observed]
                if len(rows) and not np.isfinite(rows[model]).all():
                    raise FloatingPointError("Nonfinite forecasts at measured labels.")
                record = {"period": period, "common_origins": common, "lead_months": lead,
                          "model": model, "well_id": well,
                          **_point_metrics(rows.observed_m.to_numpy(), rows[model].to_numpy())}
                local.append(record)
                per_well.append(record)
            metric = _point_metrics(valid.observed_m.to_numpy(), valid[model].to_numpy())
            metric.pop("r2")
            local = pd.DataFrame(local)
            overall.append({"period": period, "common_origins": common, "lead_months": lead, "model": model,
                            **metric, "scored_wells": int((local.n > 0).sum()),
                            "macro_r2": float(local.r2.mean()), "macro_mae_m": float(local.mae_m.mean())})
    return pd.DataFrame(overall), pd.DataFrame(per_well)


def calibration_quantiles(truth, prediction, mask, base_scale, alpha=.1):
    if not np.isfinite(base_scale).all() or (base_scale <= 0).any():
        raise ValueError("Interval scales must be finite and positive.")
    if not np.isfinite(truth[mask]).all() or not np.isfinite(prediction[mask]).all():
        raise ValueError("Calibration requires finite measured values and predictions.")
    quantiles, counts = [], []
    for h in range(prediction.shape[1]):
        score = np.abs(truth[:, h]-prediction[:, h])/base_scale[h][None]
        values = score[mask[:, h]]
        values = values[np.isfinite(values)]
        if len(values) < 10:
            raise ValueError("Insufficient held-out calibration-period measured forecasts.")
        rank = min(len(values), int(np.ceil((len(values)+1)*(1-alpha))))
        quantiles.append(float(np.sort(values)[rank-1]))
        counts.append(len(values))
    return np.asarray(quantiles), counts


def training_interval_scales(data):
    """Training-only persistence errors; a lead-wide fallback for sparse nodes."""
    starts = data.starts["train"]
    idx = starts[:, None]+np.arange(data.horizon)
    error = np.abs(data.measured[idx]-persistence(data, starts))
    valid = np.isfinite(error)
    count = valid.sum(axis=0)
    sums = np.where(valid, error, 0).sum(axis=0)
    scale = sums/np.maximum(count, 1)
    lead_scale = sums.sum(axis=1)/np.maximum(count.sum(axis=1), 1)
    return np.maximum(.05, np.where(count > 0, scale, lead_scale[:, None]))


def calibration_targets(data, lo, hi):
    """Use all eligible origins per lead; every scored target remains in [lo, hi)."""
    starts = np.arange(lo, hi)
    indices = starts[:, None]+np.arange(data.horizon)
    clipped = np.minimum(indices, len(data.measured)-1)
    mask = data.target_mask[clipped] & (indices < hi)[..., None] & (indices < len(data.measured))[..., None]
    return starts, data.measured[clipped], mask


def main(argv=None):
    code = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=code.parent/"data/gwl_and_features_advanced_sumed.csv")
    parser.add_argument("--metadata", type=Path, default=code.parent/"data/wells_meta.csv")
    parser.add_argument("--imputation-summary", type=Path, default=code.parent/"data/final_filled_summary.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int, default=24, help="Equal CV trial budget for each neural model.")
    parser.add_argument("--cv-epochs", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=140)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 123])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--horizon", type=int, choices=[1, 3, 6], default=6,
                        help="Direct multi-output forecast length; report available 1/3/6-month leads.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if min(args.trials, args.cv_epochs, args.epochs, args.threads) < 1 or not args.seeds:
        parser.error("Positive trial/epoch/thread counts and at least one seed are required.")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = args.output or code/"model_runs"/datetime.now(timezone.utc).strftime("optimized_%Y%m%d_%H%M%S")
    if output.exists():
        parser.error("Output exists; choose a new directory to preserve previous runs.")
    output.mkdir(parents=True)
    (output/"source_snapshot").mkdir()
    for name in ["optimize_stgat.py", "stgat_plots.py", "STGNN_model.py", "stgnn_data.py"]:
        if (code/name).exists():
            (output/"source_snapshot"/name).write_bytes((code/name).read_bytes())
    frame, metadata = pd.read_csv(args.input), pd.read_csv(args.metadata)
    cutoff, provenance = resolve_imputation_cutoff(args.input, args.imputation_summary, None)
    if cutoff != "2015-08-01":
        raise ValueError("This experiment's calendar is defined for the supplied August 2015 training cutoff.")
    factory = DataFactory(frame, metadata, horizon=args.horizon)
    folds = [("2012-08-01", "2013-08-01"), ("2013-08-01", "2014-08-01"), ("2014-08-01", cutoff)]
    manifest = {"complete": False, "benchmark_previously_examined": True, "benchmark_used_for_selection": False,
                "history_policy": "observed_only_with_causal_summaries", "folds": folds,
                "training_end": cutoff, "selection_end": "2017-02-01", "calibration_end": "2017-10-01",
                "benchmark_start": "2017-11-01", "leads_reported": [lead for lead in LEADS if lead <= args.horizon],
                "forecast_horizon": args.horizon,
                "trials_per_model": args.trials, "ensemble_seeds": args.seeds,
                "input_sha256": {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.input, args.metadata]},
                "versions": {p: version(p) for p in ["numpy", "pandas", "torch", "torch-geometric", "optuna", "matplotlib", "scipy"]},
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (output/"source_snapshot").iterdir()},
                "input_path": str(args.input.resolve()), "metadata_path": str(args.metadata.resolve()),
                "provenance": provenance, "models": {}}
    (output/"run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Optimization output: {output}; device {device}; {args.trials} trials/model, three chronological folds.", flush=True)
    trend_baseline = select_trend_baseline(factory, folds, output)
    manifest["trend_baseline"] = trend_baseline
    studies, configs, ensembles, outer_data, histories = {}, {}, {}, {}, {}
    for kind in ["STGAT", "GRU", "LSTM"]:
        cfg, study = search(kind, factory, folds, trials=args.trials, epochs=args.cv_epochs,
                            seed=42, device=device, output=output)
        configs[kind], studies[kind] = cfg, study
        data = factory.get(cfg, cutoff, "2017-02-01")
        outer_data[kind] = data
        members = []
        for seed in args.seeds:
            model, edges, attrs, history, best = fit_compact(kind, data, cfg, epochs=args.epochs,
                                                           patience=22, seed=seed, device=device)
            history.to_csv(output/f"{kind}_seed{seed}_history.csv", index=False)
            checkpoint = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "config": asdict(cfg), "kind": kind, "preprocessing": data.preprocessing(),
                          "edge_index": edges.cpu(), "edge_attr": attrs.cpu(), "seed": seed, "selection": best}
            torch.save(checkpoint, output/f"{kind}_seed{seed}.pt")
            restored = CompactForecast(kind, data, cfg).to(device)
            restored.load_state_dict(torch.load(output/f"{kind}_seed{seed}.pt", weights_only=True, map_location=device)["state_dict"])
            np.testing.assert_allclose(forecast(model, data, data.starts["val"], edges, attrs, device),
                                       forecast(restored, data, data.starts["val"], edges, attrs, device), atol=1e-6)
            members.append((restored, edges, attrs))
            if seed == args.seeds[0]:
                histories[kind] = history
            print(f"{kind} seed {seed}: selected epoch {best['epoch']}, validation ratio {best['score']:.4f}", flush=True)
        ensembles[kind] = members
        manifest["models"][kind] = {"config": asdict(cfg), "cv_skill_ratio": study.best_value}
        (output/"run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Freeze every configuration and checkpoint before computing benchmark scores.
    manifest["selection_frozen"] = True
    (output/"run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    data = outer_data["STGAT"]
    all_starts = np.arange(24, len(data.dates))
    predictions = {"Persistence": persistence(data, all_starts)}
    seasonal = predictions["Persistence"].copy()
    for i, start in enumerate(all_starts):
        for h in range(data.horizon):
            prior = start+h-12
            if 0 <= prior < start:
                seasonal[i, h] = np.where(np.isfinite(data.measured[prior]), data.measured[prior], seasonal[i, h])
    predictions["Seasonal"] = seasonal
    spreads = {}
    for kind, members in ensembles.items():
        p = np.stack([forecast(model, outer_data[kind], all_starts, edges, attrs, device) for model, edges, attrs in members])
        predictions[kind], spreads[kind] = p.mean(axis=0), p.std(axis=0)
    slope = data.inputs[all_starts-1, :, data.channels.index(f"gwl_slope_{int(trend_baseline['trend_window'])}m_z")]*data.target_scale
    age = np.nan_to_num(data.age[all_starts-1], nan=0).clip(max=12)
    predictions["DampedTrend"] = predictions["Persistence"]+trend_baseline["trend_gain"]*slope[:, None]*(np.arange(1, data.horizon+1)[None, :, None]+age[:, None])
    periods = {"train": (24, data.train_stop), "validation": (data.train_stop, data.dates.get_loc("2017-03-01")),
               "calibration": (data.dates.get_loc("2017-03-01"), data.dates.get_loc("2017-11-01")),
               "benchmark": (data.dates.get_loc("2017-11-01"), len(data.dates))}
    tables = []
    for period, (lo, hi) in periods.items():
        common = all_starts[(all_starts >= lo) & (all_starts+data.horizon <= hi)]
        table = rows_for_predictions(data, common, {k: v[np.isin(all_starts, common)] for k, v in predictions.items()}, period=period, common=True)
        tables.append(table)
        # All available one-month origins preserve comparability with the earlier run.
        starts = all_starts[(all_starts >= lo) & (all_starts < hi)]
        table = rows_for_predictions(data, starts, {k: v[np.isin(all_starts, starts)] for k, v in predictions.items()}, period=period, common=False)
        tables.append(table.loc[table.lead_months == 1])
    model_names = list(predictions)
    table = pd.concat(tables, ignore_index=True)
    metrics, wells = metrics_tables(table, model_names)
    table.to_csv(output/"predictions.csv", index=False)
    metrics.to_csv(output/"metrics.csv", index=False)
    wells.to_csv(output/"metrics_per_well.csv", index=False)
    trajectory = rows_for_predictions(data, all_starts, predictions)
    trajectory.to_csv(output/"all_forecasts.csv", index=False)
    cal_lo, cal_hi = periods["calibration"]
    cal_starts, cal_truth, cal_mask = calibration_targets(data, cal_lo, cal_hi)
    base_scale = training_interval_scales(data)
    selected = np.isin(all_starts, cal_starts)
    q, counts = calibration_quantiles(cal_truth, predictions["STGAT"][selected], cal_mask, base_scale)
    pd.DataFrame(base_scale, index=np.arange(1, data.horizon+1), columns=data.wells).rename_axis("lead_months").to_csv(output/"interval_training_scales.csv")
    uncertainty = []
    for i, start in enumerate(all_starts):
        for h in range(data.horizon):
            target = start+h
            if target >= len(data.dates):
                continue
            for node, well in enumerate(data.wells):
                mu, half = predictions["STGAT"][i, h, node], q[h]*base_scale[h, node]
                uncertainty.append({"origin_month": str(data.dates[start-1].date()), "target_month": str(data.dates[target].date()),
                    "lead_months": h+1, "well_id": well, "prediction_m": mu, "lower_m": mu-half, "upper_m": mu+half,
                    "half_width_m": half, "ensemble_std_m": spreads["STGAT"][i, h, node],
                    "observed_m": data.measured[target, node], "is_observed": bool(data.target_mask[target, node])})
    uncertainty = pd.DataFrame(uncertainty)
    uncertainty.to_csv(output/"prediction_intervals.csv", index=False)
    manifest["interval_calibration"] = {"nominal_coverage": .9, "samples_by_lead": counts, "normalized_quantiles": q.tolist(),
        "origin_policy": "Every eligible origin per lead, with all calibration targets inside the reserved period",
        "period": "March-October 2017; excluded from hyperparameter and checkpoint selection",
        "limitation": "Time/spatial dependence prevents an exchangeability-based coverage guarantee; assess empirical coverage."}
    importance = grouped_importance(ensembles["STGAT"], outer_data["STGAT"], device)
    importance.to_csv(output/"feature_importance.csv", index=False)
    metadata.to_csv(output/"well_metadata.csv", index=False)
    context = {"wells": data.wells, "coordinates": data.coordinates.tolist(),
               "dates": [str(date.date()) for date in data.dates], "periods": periods,
               "train_stop": data.train_stop, "val_stop": data.val_stop, "history_seed": args.seeds[0]}
    (output/"plot_context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    if not args.no_plots:
        from stgat_plots import generate_all_plots
        generate_all_plots(output, studies, histories, trajectory, metrics, wells, metadata, uncertainty, importance, periods, data)
    manifest["complete"] = True
    (output/"run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.loc[(metrics.period == "benchmark") & ~metrics.common_origins & (metrics.lead_months == 1)].to_string(index=False), flush=True)
    print(f"Completed optimization and full plot collection: {output}", flush=True)


def grouped_importance(members, data, device, repeats=8):
    """Validation-only grouped permutation; groundwater anchor is permuted jointly."""
    starts = data.starts["val"]
    baseline = np.mean([forecast(m, data, starts, e, a, device) for m, e, a in members], axis=0)
    base_score, _ = skill_score(data, starts, baseline)
    groups = {"Groundwater history": [i for i, c in enumerate(data.channels)
                if c.startswith("gwl_") or c.startswith("level_filled") or c == "last_measured_level_z"]}
    for feature in data.features[1:]:
        groups[feature] = [i for i, c in enumerate(data.channels) if c == feature or c.startswith(feature+"_")]
    groups["Seasonal calendar"] = [data.channels.index("sin_month"), data.channels.index("cos_month")]
    rng = np.random.default_rng(42)
    rows = []
    x = np.stack([data.inputs[s-data.lookback:s] for s in starts])
    anchors = data.anchor_z[starts-1].astype(np.float32)
    for label, channels in groups.items():
        deltas = []
        for _ in range(repeats):
            # Permute temporal windows within each well, jointly for a physical
            # feature's values/masks/coverage. This is a diagnostic intervention.
            xx, aa = x.copy(), anchors.copy()
            for node in range(len(data.wells)):
                order = rng.permutation(len(starts))
                xx[:, :, node, channels] = x[order][:, :, node, channels]
                if label == "Groundwater history":
                    aa[:, node] = anchors[order, node]
            values = []
            for model, edges, attrs in members:
                model.eval()
                with torch.no_grad():
                    pred = model(torch.as_tensor(xx, device=device), edges, attrs, torch.as_tensor(aa, device=device)).cpu().numpy()
                values.append(pred*data.target_scale+data.target_mean)
            score, _ = skill_score(data, starts, np.mean(values, axis=0))
            deltas.append(score-base_score)
        rows.append({"feature": label, "increase_validation_skill_ratio": float(np.mean(deltas)), "sd": float(np.std(deltas))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
