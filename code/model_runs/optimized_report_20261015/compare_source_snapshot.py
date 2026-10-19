"""Select a one-month forecasting variant on validation, then report its benchmark.

The horizon-selection rule is defined in OPTIMIZATION_PROTOCOL.txt.
Run after both optimization experiments finish; no training or parameter tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def point_metrics(rows, model):
    rows = rows.loc[rows.is_observed]
    error = rows[model].to_numpy()-rows.observed_m.to_numpy()
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Measured labels require finite predictions.")
    local_r2 = []
    for _, well in rows.groupby("well_id"):
        y = well.observed_m.to_numpy()
        denominator = ((y-y.mean())**2).sum()
        if len(y) >= 2 and denominator > 1e-12:
            local_r2.append(1-((well[model].to_numpy()-y)**2).sum()/denominator)
    return {"n": len(rows), "scored_wells": rows.well_id.nunique(),
            "mae_m": float(np.abs(error).mean()), "rmse_m": float(np.sqrt((error**2).mean())),
            "bias_m": float(error.mean()), "macro_r2": float(np.mean(local_r2))}


def one_month(path):
    frame = pd.read_csv(path/"predictions.csv")
    frame = frame.loc[~frame.common_origins & (frame.lead_months == 1)]
    return {period: rows.sort_values(["well_id", "target_month"]).reset_index(drop=True)
            for period, rows in frame.groupby("period")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon1", type=Path, required=True)
    parser.add_argument("--horizon6", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true", help="Refresh an existing report while verifying its validation selections stay unchanged.")
    args = parser.parse_args()
    paths = {1: args.horizon1.resolve(), 6: args.horizon6.resolve()}
    manifests = {h: json.loads((p/"run.json").read_text()) for h, p in paths.items()}
    if not all(m["complete"] and m["selection_frozen"] for m in manifests.values()):
        raise ValueError("Both optimization runs must be complete and frozen.")
    if manifests[1]["input_sha256"] != manifests[6]["input_sha256"]:
        raise ValueError("Experiments used different input data.")
    output = args.output.resolve()
    if output.exists() and not args.refresh:
        raise FileExistsError("Preserve old reports; choose a new output directory.")
    prior_decision = json.loads((output/"selection.json").read_text()) if output.exists() else None
    output.mkdir(parents=True, exist_ok=args.refresh)
    frames = {h: one_month(p) for h, p in paths.items()}
    models = ["Persistence", "Seasonal", "DampedTrend", "STGAT", "GRU", "LSTM"]
    keys = ["well_id", "origin_month", "target_month", "is_observed", "observed_m"]
    # Select and serialize every variant from validation before scoring benchmark.
    pd.testing.assert_frame_equal(frames[1]["validation"][keys], frames[6]["validation"][keys])
    selection = {}
    for model in models:
        values = {h: point_metrics(frames[h]["validation"], model)["mae_m"] for h in paths}
        horizon = min(values, key=lambda h: (values[h], h))
        selection[model] = {"training_horizon": horizon, "validation_mae_m": values[horizon],
                            "candidate_validation_mae_m": values, "run": str(paths[horizon])}
    decision = {"rule": "Lowest pooled one-month measured-label validation MAE; exact ties prefer H=1.",
                "selection_period": "September 2015-February 2017", "models": selection,
                "benchmark_previously_examined": True, "selection_uses_benchmark": False,
                "protocol_sha256": hashlib.sha256((Path(__file__).parent/"OPTIMIZATION_PROTOCOL.txt").read_bytes()).hexdigest()}
    if prior_decision:
        assert prior_decision["protocol_sha256"] == decision["protocol_sha256"]
        for model, prior in prior_decision["models"].items():
            assert prior["training_horizon"] == selection[model]["training_horizon"]
            assert prior["run"] == selection[model]["run"]
            np.testing.assert_allclose(prior["validation_mae_m"], selection[model]["validation_mae_m"], atol=1e-12)
    (output/"selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    pd.testing.assert_frame_equal(frames[1]["benchmark"][keys], frames[6]["benchmark"][keys])
    combined = frames[1]["benchmark"][keys].copy()
    summary, variants = [], []
    for model in models:
        h = selection[model]["training_horizon"]
        combined[model] = frames[h]["benchmark"][model]
        summary.append({"model": model, "training_horizon": h,
                        "validation_mae_m": selection[model]["validation_mae_m"],
                        **point_metrics(frames[h]["benchmark"], model)})
        for horizon in paths:
            variants.append({"model": model, "training_horizon": horizon,
                             "selected": h == horizon,
                             "validation_mae_m": selection[model]["candidate_validation_mae_m"][horizon],
                             **point_metrics(frames[horizon]["benchmark"], model)})
    metrics = pd.DataFrame(summary)
    baseline = float(metrics.set_index("model").loc["Persistence", "mae_m"])
    metrics["mae_skill_vs_persistence"] = 1-metrics.mae_m/baseline
    metrics.to_csv(output/"recommended_metrics.csv", index=False)
    variant_table = pd.DataFrame(variants)
    variant_table.to_csv(output/"all_variants.csv", index=False)
    neural_variants = variant_table.loc[variant_table.model.isin(["STGAT", "GRU", "LSTM"])].copy()
    lowest_mae = neural_variants.loc[neural_variants.mae_m.idxmin()]
    combined.to_csv(output/"recommended_predictions.csv", index=False)
    previous = None
    if args.previous:
        old = pd.read_csv(args.previous/"metrics.csv")
        previous = old.loc[(old.split == "test") & (old.model == "STGAT")].iloc[0].to_dict()
        if int(previous["n"]) != int(metrics.iloc[0].n):
            raise ValueError("Previous benchmark sample count differs.")
        # Dates and original labels, rather than sample count alone, must agree.
        old_predictions = pd.read_csv(args.previous/"predictions.csv")
        old_predictions = old_predictions.loc[(old_predictions.split == "test") &
            (old_predictions.lead_months == 1)].rename(columns={
                "is_observed_target": "is_observed", "observed_level_m": "observed_m"})
        old_predictions = old_predictions.sort_values(["well_id", "target_month"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(old_predictions[keys], frames[1]["benchmark"][keys])
        np.testing.assert_allclose(old_predictions.Persistence_prediction_m, frames[1]["benchmark"].Persistence, atol=1e-9, rtol=0)
        (output/"previous_reference.json").write_text(json.dumps(previous, indent=2), encoding="utf-8")
    colors = {"Persistence": "#64717d", "Seasonal": "#9a65a7", "DampedTrend": "#37a1a2",
              "STGAT": "#176ca5", "GRU": "#218866", "LSTM": "#d98924"}
    displayed = metrics.loc[metrics.model != "Seasonal"].copy()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, column, label in zip(axes, ["mae_m", "rmse_m", "macro_r2"], ["MAE (m)", "RMSE (m)", "Mean per-well R²"]):
        ax.bar(displayed.model, displayed[column], color=[colors[m] for m in displayed.model])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
        ax.axhline(0, color="#888888", linewidth=.5)
    fig.suptitle("One-month existing benchmark · model variants selected on validation")
    fig.tight_layout()
    for suffix in ["png", "svg"]:
        fig.savefig(output/f"recommended_comparison.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    longer = pd.read_csv(paths[6]/"metrics.csv")
    longer = longer.loc[(longer.period == "benchmark") & longer.common_origins & longer.lead_months.isin([1, 3, 6])]
    longer.to_csv(output/"six_month_horizon_metrics.csv", index=False)
    stgat = metrics.set_index("model").loc["STGAT"]
    tradeoff = (f"Across all tested neural variants, {lowest_mae.model} trained for H={int(lowest_mae.training_horizon)} "
                f"has the lowest benchmark MAE ({lowest_mae.mae_m:.3f} m), with RMSE {lowest_mae.rmse_m:.3f} m. "
                f"The validation-selected STGAT has MAE {stgat.mae_m:.3f} m and RMSE {stgat.rmse_m:.3f} m. "
                "This observed ranking does not change the predefined validation selections.")
    text = ["STGAT OPTIMIZATION RESULTS", "", "One-month recommended variants (chosen on validation only):",
            metrics.round(5).to_string(index=False), "", tradeoff, "", "All tested neural variants:",
            neural_variants.round(5).to_string(index=False), "", "Separate six-month model; common forecast origins:",
            longer[["model", "lead_months", "n", "mae_m", "rmse_m", "macro_r2"]].round(5).to_string(index=False), "",
            "All 18 original plot categories are available in each experiment, plus a horizon comparison.",
            "Missing/imputed groundwater labels never enter training losses, selection scores or benchmark metrics.",
            "1580/52 retains its graph node and forecasts; it has no measured benchmark targets and is unscored.",
            "The compact STGAT uses a temporal GRU, gated GATv2 spatial correction, persistence/trend skip, and three fixed seeds.",
            "Environmental values have training-only scaling, availability/coverage channels, and no future feature inputs.",
            "Interpolation files were preserved. This experiment uses measured history with causal summaries to keep early CV folds honest.",
            "", "Limits:",
            "The existing benchmark had already been examined in the earlier review. These are exploratory research results, not a new untouched test.",
            "A small gain does not establish a stable advantage across aquifers or new periods. Compare both MAE and RMSE, per-well scores, and longer leads.",
            "Nominal 90% interval bands use separate March-October 2017 calibration; assess observed coverage rather than assuming 90%.",
            "Prediction band width is fixed per well/lead from training error scale and held-out residual quantiles; it is not a learned dynamic variance.",
            "Grouped permutation is a validation-only diagnostic; parameter importance is conditional on the searched space.",
            "All 45 regression tests passed before the optimization runs. Checkpoint reload predictions were verified for all ensemble members."]
    text += ["Training sources are preserved in each run's source_snapshot; plot_source_snapshot.py records the exact renderer, including formatting updates.",
             "The lookback is the recurrent input window; causal slope/change summaries can retain information from up to 24 previous months.",
             "Horizon accuracy uses common forecast origins; interval coverage uses every eligible benchmark origin separately per lead.",
             "Calibration uses all eligible origins per lead and excludes every target after October 2017. The reserved period has measurements only in July-October 2017.",
             "The additional calibration boundary regression passed with the full focused 10-test optimization suite after this correction."]
    for horizon, run in paths.items():
        coverage = pd.read_csv(run/"interval_coverage.csv")
        text += ["", f"H={horizon} interval coverage on measured benchmark labels:", coverage.round(5).to_string(index=False),
                 f"Calibration sample counts by lead: {manifests[horizon]['interval_calibration']['samples_by_lead']}"]
    if previous:
        text += ["", f"Previous reviewed STGAT one-month MAE: {previous['mae_m']:.6f} m; RMSE: {previous['rmse_m']:.6f} m.",
                 f"Recommended STGAT one-month MAE: {stgat.mae_m:.6f} m; RMSE: {stgat.rmse_m:.6f} m."]
    (output/"RESULTS.txt").write_text("\n".join(text), encoding="utf-8")
    (output/"compare_source_snapshot.py").write_bytes(Path(__file__).read_bytes())

    def page(one, six):
        return "\n".join(["<!doctype html><meta charset='utf-8'><title>Groundwater STGAT results</title>",
            "<style>body{font:16px system-ui;max-width:1300px;margin:40px auto;padding:0 24px;color:#182735}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd}img{max-width:100%}a{color:#176ca5}</style>",
            "<h1>Groundwater forecasting: optimized STGAT</h1>",
            "<p>One-month variants are selected separately for each model using September 2015-February 2017 validation MAE. The table scores identical measured labels from the previously examined November 2017-January 2020 benchmark.</p>",
            f"<p><strong>{html.escape(tradeoff)}</strong></p>",
            "<h2>Validation-selected variants</h2>",
            metrics.round(4).to_html(index=False), "<img src='recommended_comparison.png' alt='Recommended models benchmark comparison'>",
            "<h2>All tested neural variants</h2>",
            neural_variants[["model", "training_horizon", "selected", "validation_mae_m", "mae_m", "rmse_m", "macro_r2"]].round(4).to_html(index=False),
            f"<h2>Complete notebook plots</h2><p><a href='{html.escape(one)}'>Dedicated one-month experiment: all 18 original plot types plus horizon comparison</a></p>",
            f"<p><a href='{html.escape(six)}'>Six-month experiment: all 18 original plot types plus 1/3/6-month comparison</a></p>",
            "<p><a href='RESULTS.txt'>Detailed results and limitations</a> · <a href='all_variants.csv'>Both variants, without hiding unfavorable results</a> · <a href='selection.json'>Validation selection record</a></p>",
            "<p>Nominal prediction bands are calibrated on a separate period; spatial and temporal dependence limit coverage guarantees. Missing target labels are never scored.</p>"])
    local = page("../"+paths[1].name+"/plots/index.html", "../"+paths[6].name+"/plots/index.html")
    (output/"index.html").write_text(local, encoding="utf-8")
    with zipfile.ZipFile(output/"all_optimization_plots.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("index.html", page("one_month/plots/index.html", "six_month/plots/index.html"))
        for file in output.iterdir():
            if file.suffix != ".zip" and file.name not in {"index.html", "verification.json"}:
                archive.write(file, arcname=file.name)
        for horizon, prefix in [(1, "one_month"), (6, "six_month")]:
            for file in (paths[horizon]/"plots").iterdir():
                archive.write(file, arcname=f"{prefix}/plots/{file.name}")
            for name in ["plot_inventory.csv", "plot_render.json", "interval_coverage.csv"]:
                archive.write(paths[horizon]/name, arcname=f"{prefix}/{name}")
    print(metrics.to_string(index=False))
    print(f"Complete report and all plots: {output}")


if __name__ == "__main__":
    main()
