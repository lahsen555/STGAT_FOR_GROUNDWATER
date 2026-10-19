"""Finish diagnostics from frozen checkpoints without training or model selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import torch

from optimize_stgat import (DataFactory, OptimConfig, CompactForecast, forecast, grouped_importance,
                            training_interval_scales, calibration_targets, calibration_quantiles, rows_for_predictions)
from stgat_plots import generate_all_plots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output = args.run.resolve()
    manifest = json.loads((output/"run.json").read_text())
    if not manifest.get("selection_frozen") or set(manifest["models"]) != {"STGAT", "GRU", "LSTM"}:
        raise ValueError("All models must already be selected and frozen.")
    def input_path(name):
        path = Path(name)
        return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path
    for name, sha in manifest["input_sha256"].items():
        if hashlib.sha256(input_path(name).read_bytes()).hexdigest() != sha:
            raise ValueError("Training inputs changed; preserve the original run.")
    torch.set_num_threads(2)
    device = torch.device("cpu")
    frame = pd.read_csv(input_path(manifest["input_path"]))
    metadata = pd.read_csv(input_path(manifest["metadata_path"]))
    horizon = manifest["forecast_horizon"]
    factory = DataFactory(frame, metadata, horizon=horizon)
    cfg = OptimConfig(**manifest["models"]["STGAT"]["config"])
    data = factory.get(cfg, manifest["training_end"], manifest["selection_end"])
    seeds = manifest["ensemble_seeds"]
    members = []
    for seed in seeds:
        saved = torch.load(output/f"STGAT_seed{seed}.pt", weights_only=True, map_location=device)
        if saved["preprocessing"] != data.preprocessing():
            raise ValueError("Reconstructed preprocessing differs from the saved checkpoint.")
        model = CompactForecast("STGAT", data, cfg).to(device)
        model.load_state_dict(saved["state_dict"])
        members.append((model, saved["edge_index"].to(device), saved["edge_attr"].to(device)))
    starts = np.arange(24, len(data.dates))
    ensemble = np.stack([forecast(model, data, starts, edges, attrs, device) for model, edges, attrs in members])
    restored = rows_for_predictions(data, starts, {"prediction_m": ensemble.mean(axis=0), "ensemble_std_m": ensemble.std(axis=0)})
    trajectory = pd.read_csv(output/"all_forecasts.csv")
    key = ["origin_month", "target_month", "lead_months", "well_id", "is_observed", "observed_m"]
    pd.testing.assert_frame_equal(restored[key], trajectory[key])
    np.testing.assert_allclose(restored.prediction_m, trajectory.STGAT, atol=1e-8, rtol=0)
    periods = {"train": (24, data.train_stop),
               "validation": (data.train_stop, int(data.dates.get_loc("2017-03-01"))),
               "calibration": (int(data.dates.get_loc("2017-03-01")), int(data.dates.get_loc("2017-11-01"))),
               "benchmark": (int(data.dates.get_loc("2017-11-01")), len(data.dates))}
    cal_starts, truth, mask = calibration_targets(data, *periods["calibration"])
    scales = training_interval_scales(data)
    quantiles, counts = calibration_quantiles(truth, ensemble.mean(axis=0)[np.isin(starts, cal_starts)], mask, scales)
    pd.DataFrame(scales, index=np.arange(1, horizon+1), columns=data.wells).rename_axis("lead_months").to_csv(output/"interval_training_scales.csv")
    nodes = {well: i for i, well in enumerate(data.wells)}
    lead_indices = restored.lead_months.to_numpy()-1
    node_indices = restored.well_id.map(nodes).to_numpy()
    restored["half_width_m"] = quantiles[lead_indices]*scales[lead_indices, node_indices]
    restored["lower_m"] = restored.prediction_m-restored.half_width_m
    restored["upper_m"] = restored.prediction_m+restored.half_width_m
    uncertainty = restored[["origin_month", "target_month", "lead_months", "well_id", "prediction_m",
                            "lower_m", "upper_m", "half_width_m", "ensemble_std_m", "observed_m", "is_observed"]]
    uncertainty.to_csv(output/"prediction_intervals.csv", index=False)
    manifest["interval_calibration"] = {"nominal_coverage": .9, "samples_by_lead": counts,
        "normalized_quantiles": quantiles.tolist(),
        "period": "March-October 2017; excluded from hyperparameter and checkpoint selection",
        "origin_policy": "Every eligible origin per lead, with all calibration targets inside the reserved period",
        "limitation": "Time/spatial dependence prevents an exchangeability-based coverage guarantee; assess empirical coverage."}
    if (output/"feature_importance.csv").exists():
        importance = pd.read_csv(output/"feature_importance.csv")
    else:
        importance = grouped_importance(members, data, device)
        importance.to_csv(output/"feature_importance.csv", index=False)
    metadata.to_csv(output/"well_metadata.csv", index=False)
    context = {"wells": data.wells, "coordinates": data.coordinates.tolist(),
               "dates": [str(date.date()) for date in data.dates], "periods": periods,
               "train_stop": data.train_stop, "val_stop": data.val_stop, "history_seed": seeds[0]}
    (output/"plot_context.json").write_text(json.dumps(context, indent=2), encoding="utf-8")
    rendering_sources = {}
    for source, dest in [(Path(__file__), "postprocessing_source_snapshot.py"),
                         (Path(__file__).with_name("optimize_stgat.py"), "postprocessing_optimizer_snapshot.py")]:
        contents = source.read_bytes()
        (output/dest).write_bytes(contents)
        rendering_sources[dest] = hashlib.sha256(contents).hexdigest()
    manifest["postprocessing"] = {"recovered_from_frozen_checkpoints": True,
        "training_and_point_predictions_unchanged": True, "source_sha256": rendering_sources,
        "reason": "Calibration originally required shared origins, including months with no measured labels. Each lead now uses all eligible held-out calibration targets."}
    (output/"run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    database = f"sqlite:///{(output/'studies.sqlite3').as_posix()}"
    studies = {kind: optuna.load_study(study_name=kind, storage=database) for kind in ["STGAT", "GRU", "LSTM"]}
    histories = {kind: pd.read_csv(output/f"{kind}_seed{seeds[0]}_history.csv") for kind in studies}
    generate_all_plots(output, studies, histories, trajectory, pd.read_csv(output/"metrics.csv"),
                       pd.read_csv(output/"metrics_per_well.csv"), metadata, uncertainty, importance, periods, data)
    manifest["complete"] = True
    (output/"run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Completed diagnostics from saved checkpoints: {output}")
    print(f"Calibration labels by lead: {counts}")


if __name__ == "__main__":
    main()
