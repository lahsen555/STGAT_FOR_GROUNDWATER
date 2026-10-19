"""Reproduce every original STGNN notebook plot with measured-label masks.

Rerender without retraining: python code/stgat_plots.py --run RUN_DIRECTORY
Produces PNG/SVG files, a plot inventory, and a local HTML gallery.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from scipy.stats import gaussian_kde, pearsonr, wilcoxon

COLORS = {"STGAT": "#176ca5", "GRU": "#218866", "LSTM": "#d98924",
          "Persistence": "#64717d", "Seasonal": "#9a65a7", "DampedTrend": "#37a1a2"}
LABELS = {"DampedTrend": "Damped trend", "Seasonal": "Seasonal fallback", "Persistence": "Persistence"}


def generate_all_plots(output, studies, histories, trajectory, metrics, wells, metadata,
                       uncertainty, importance, periods, data):
    output = Path(output)
    plots = output/"plots"
    plots.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    inventory = []

    def save(fig, name, description, cell, original=True):
        fig.savefig(plots/(name+".png"), dpi=170, bbox_inches="tight")
        fig.savefig(plots/(name+".svg"), bbox_inches="tight")
        plt.close(fig)
        inventory.append({"plot": name, "original_cell": cell, "original_plot": original,
                          "description": description, "png": name+".png", "svg": name+".svg"})

    def boundaries(ax):
        ax.axvline(data.dates[periods["validation"][0]], color="#555555", linewidth=.8, linestyle="--")
        ax.axvspan(data.dates[periods["calibration"][0]], data.dates[periods["benchmark"][0]], color="#dce4eb", alpha=.5)
        ax.axvline(data.dates[periods["benchmark"][0]], color="#b44462", linewidth=.9, linestyle="--")
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.grid(alpha=.16)

    selected = metrics.loc[(metrics.period == "benchmark") & ~metrics.common_origins & (metrics.lead_months == 1)]
    model_order = [m for m in ["Persistence", "Seasonal", "DampedTrend", "STGAT", "GRU", "LSTM"] if m in set(selected.model)]
    selected = selected.set_index("model").loc[model_order]
    per = wells.loc[(wells.period == "benchmark") & ~wells.common_origins & (wells.lead_months == 1)]
    measured = trajectory.loc[(trajectory.lead_months == 1) & trajectory.is_observed &
                                (pd.to_datetime(trajectory.target_month) >= data.dates[periods["benchmark"][0]])].copy()
    errors = (measured.STGAT-measured.observed_m).to_numpy()
    title = "One-month existing benchmark · measured labels only"

    evaluator = optuna.importance.PedAnovaImportanceEvaluator(baseline_quantile=.3, evaluate_on_local=False)
    importances = optuna.importance.get_param_importances(studies["STGAT"], evaluator=evaluator)
    pd.DataFrame(list(importances.items()), columns=["parameter", "importance"]).to_csv(output/"parameter_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    names, values = list(importances), list(importances.values())
    ax.barh(names[::-1], values[::-1], color=COLORS["STGAT"])
    ax.set(xlabel="PED-ANOVA relative importance", title="STGAT hyperparameter importance · chronological CV")
    fig.tight_layout()
    save(fig, "hyperparameter_importance", "CV parameter importance; association within the searched space.", 15)

    fig, ax = plt.subplots(figsize=(10, 5))
    for kind, study in studies.items():
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        numbers = [t.number+1 for t in complete]
        values = np.asarray([t.value for t in complete])
        ax.scatter(numbers, values, color=COLORS[kind], alpha=.3, s=15)
        ax.plot(numbers, np.minimum.accumulate(values), color=COLORS[kind], label=kind)
    ax.axhline(1, color=COLORS["Persistence"], linestyle="--", label="Persistence")
    ax.set(xlabel="Completed trial", ylabel="Mean CV MAE / persistence MAE", title="Optimization history · leads 1, 3, 6")
    ax.legend()
    fig.tight_layout()
    save(fig, "optimization_history", "Three chronological folds; lower ratios are better.", 15)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for kind, history in histories.items():
        axes[0].plot(history.epoch, history.training_loss, color=COLORS[kind], label=kind)
        axes[1].plot(history.epoch, history.validation_skill_ratio, color=COLORS[kind], label=kind)
    axes[0].set(xlabel="Epoch", ylabel="Training objective", title="Configuration-specific training losses")
    axes[1].set(xlabel="Epoch", ylabel="Validation MAE / persistence MAE", title="Checkpoint selection · no calibration/benchmark labels")
    axes[1].axhline(1, linestyle="--", color="#666666")
    axes[1].legend()
    fig.tight_layout()
    save(fig, "training_curve", "First ensemble seed; losses may differ, validation skill is comparable.", 19)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    row = selected.loc["STGAT"]
    for ax, column, label in zip(axes, ["mae_m", "rmse_m", "macro_r2"], ["MAE (m)", "RMSE (m)", "Mean per-well R²"]):
        ax.bar(["STGAT"], [row[column]], color=COLORS["STGAT"])
        ax.set_ylabel(label)
        ax.set_title(f"{row[column]:.3f}")
        ax.axhline(0, color="#999999", linewidth=.6)
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, "model_metrics", "Different metrics use separate axes; R² is averaged within wells.", 20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column, label in zip(axes, ["mae_m", "rmse_m"], ["MAE (m)", "RMSE (m)"]):
        ax.bar([LABELS.get(m, m) for m in model_order], selected[column], color=[COLORS[m] for m in model_order])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle(title)
    fig.tight_layout()
    save(fig, "metrics_comparison", "All models score the same measured one-month targets.", 26)

    def heatmap(values, name, description, cell, *, vmin, vmax, cmap="viridis_r", fmt=".2f"):
        values = values.reindex(index=data.wells, columns=[m for m in model_order if m in values.columns])
        fig, ax = plt.subplots(figsize=(10, max(7, .34*len(data.wells))))
        matrix = values.to_numpy(dtype=float)
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(values.columns)), [LABELS.get(c, c) for c in values.columns], rotation=25, ha="right")
        ax.set_yticks(range(len(data.wells)), data.wells)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, format(matrix[i, j], fmt) if np.isfinite(matrix[i, j]) else "—", ha="center", va="center", fontsize=8,
                        color="white" if np.isfinite(matrix[i, j]) and (matrix[i, j]-vmin)/(vmax-vmin) < .35 else "black")
        ax.set_title(description)
        fig.colorbar(im, ax=ax, shrink=.65)
        fig.tight_layout()
        save(fig, name, description+"; blank rows have no measured targets.", cell)

    for metric, ascending, label in [("mae_m", True, "MAE"), ("rmse_m", True, "RMSE"), ("r2", False, "R²")]:
        pivot = per.pivot(index="well_id", columns="model", values=metric)
        heatmap(pivot.rank(axis=1, ascending=ascending), f"rank_{label.replace('²','2').lower()}",
                f"{label} ranks · 1 = best · existing benchmark", 28, vmin=1, vmax=len(model_order), fmt=".1f")
    r2 = per.pivot(index="well_id", columns="model", values="r2")
    heatmap(r2, "r2heatmap", "Per-well R² · colour clipped to [−1, 1], labels exact", 31,
            vmin=-1, vmax=1, cmap="RdYlGn")

    one = trajectory.loc[trajectory.lead_months == 1].copy()
    one["target_month"] = pd.to_datetime(one.target_month)
    fig, axes = plt.subplots(int(np.ceil(len(data.wells)/4)), 4, figsize=(18, 3*np.ceil(len(data.wells)/4)), squeeze=False)
    for ax, well in zip(axes.flat, data.wells):
        rows = one.loc[one.well_id == well].sort_values("target_month")
        for kind in ["STGAT", "GRU", "LSTM"]:
            ax.plot(rows.target_month, rows[kind], color=COLORS[kind], linewidth=.8, label=kind)
        ax.plot(rows.target_month, rows.observed_m, ".", color="black", markersize=2.5, label="Measured")
        ax.set_title(well, fontsize=10)
        ax.tick_params(labelsize=7)
        boundaries(ax)
    for ax in list(axes.flat)[len(data.wells):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.text(.5, .005, "Dashed grey: training ends · shaded: interval calibration · dashed red: existing benchmark starts", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .02, 1, .97))
    save(fig, "predictions_comparison", "All wells; measured values only; monthly dates retained.", 34)

    chosen = [well for well in ["3976/53", "4405/44", "4442/44"] if well in data.wells]
    fig, axes = plt.subplots(len(chosen), 1, figsize=(13, 3*len(chosen)), squeeze=False)
    for ax, well in zip(axes.flat, chosen):
        rows = one.loc[one.well_id == well].sort_values("target_month")
        ax.plot(rows.target_month, rows.observed_m, ".", color="black", label="Measured", markersize=4)
        for kind in ["STGAT", "GRU", "LSTM"]:
            ax.plot(rows.target_month, rows[kind], color=COLORS[kind], linewidth=1, label=kind)
        ax.set(title=well, ylabel="Level (m)")
        boundaries(ax)
    axes[0, 0].legend(ncol=4)
    fig.tight_layout()
    save(fig, "selected_wells_comparison", "Original selected well IDs; actual dated measurements.", 35)

    spatial = per.loc[per.model == "STGAT"].set_index("well_id").reindex(data.wells)
    coords = np.asarray(data.coordinates)
    fig, ax = plt.subplots(figsize=(10, 7))
    valid = spatial.n.to_numpy() > 0
    point = ax.scatter(coords[valid, 1], coords[valid, 0], c=spatial.rmse_m.to_numpy()[valid], s=70,
                       cmap="YlOrRd", edgecolor="#333333", linewidth=.4)
    ax.scatter(coords[~valid, 1], coords[~valid, 0], marker="x", color="#555555", s=70, label="No measured benchmark labels")
    for i, well in enumerate(data.wells):
        ax.annotate(well, (coords[i, 1], coords[i, 0]), xytext=(5 if i%2 else -5, 6 if i%3 else -11),
                    textcoords="offset points", ha="left" if i%2 else "right", fontsize=8)
    ax.set(xlabel="Longitude (°)", ylabel="Latitude (°)", title="STGAT measured-label RMSE (m) · existing benchmark")
    ax.set_aspect(1/np.cos(np.radians(coords[:, 0].mean())))
    ax.margins(.15)
    ax.grid(alpha=.15)
    ax.legend(loc="lower left", fontsize=8)
    fig.colorbar(point, ax=ax, label="RMSE (m)", shrink=.7)
    fig.tight_layout()
    save(fig, "rmse_spatial_distribution", "Geographic error map; no network tiles required; unscored wells marked.", 36)

    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(measured.observed_m, measured.STGAT, gridsize=30, mincnt=1, cmap="Blues")
    lo = min(measured.observed_m.min(), measured.STGAT.min())
    hi = max(measured.observed_m.max(), measured.STGAT.max())
    ax.plot([lo, hi], [lo, hi], "--", color="#b84444")
    ax.set(xlabel="Measured level (m)", ylabel="Forecast level (m)", title="Absolute levels · existing benchmark")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(hb, ax=ax, label="Measured forecast pairs")
    fig.text(.5, .01, "Across-well elevation differences dominate this view; assess temporal skill with per-well R².", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, 1))
    save(fig, "density_obs_vs_pred", "Measured targets only; no pooled R² claim.", 37)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors, bins=25, density=True, color=COLORS["STGAT"], alpha=.5, edgecolor="white")
    if np.std(errors) > 1e-8:
        xx = np.linspace(errors.min(), errors.max(), 200)
        ax.plot(xx, gaussian_kde(errors)(xx), color=COLORS["STGAT"])
    ax.axvline(0, color="black", linestyle="--", label="Zero error")
    ax.axvline(errors.mean(), color="#b44462", label=f"Bias {errors.mean():.3f} m")
    ax.set(xlabel="Forecast − measurement (m)", ylabel="Density", title=f"STGAT residuals · n={len(errors)} measured targets")
    ax.legend()
    fig.tight_layout()
    save(fig, "residuals_distribution", "Residuals exclude missing and imputed labels.", 38)

    imp = importance.sort_values("increase_validation_skill_ratio")
    fig, ax = plt.subplots(figsize=(9, max(4, .45*len(imp))))
    ax.barh(imp.feature, imp.increase_validation_skill_ratio, xerr=imp.sd, color=COLORS["STGAT"], capsize=3)
    ax.axvline(0, color="#777777", linewidth=.8)
    ax.set(xlabel="Increase in validation MAE/persistence ratio", title="Grouped permutation importance · selection validation only")
    fig.tight_layout()
    save(fig, "feature_importance", "Values/masks/coverage permuted jointly; groundwater anchor included in its group; diagnostic, not causal.", 40)

    unc = uncertainty.loc[(uncertainty.lead_months == 1) &
                          (pd.to_datetime(uncertainty.target_month) >= data.dates[periods["benchmark"][0]])].copy()
    unc["target_month"] = pd.to_datetime(unc.target_month)
    well = "2698/53" if "2698/53" in data.wells else data.wells[0]
    rows = unc.loc[unc.well_id == well].sort_values("target_month")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(rows.target_month, rows.lower_m, rows.upper_m, color=COLORS["STGAT"], alpha=.2, label="Validation-calibrated nominal 90% band")
    ax.plot(rows.target_month, rows.prediction_m, color=COLORS["STGAT"], label="STGAT ensemble")
    ax.plot(rows.target_month, rows.observed_m, ".", color="black", label="Measured")
    ax.set(title=f"Well {well} · empirical prediction band, no coverage guarantee", ylabel="Level (m)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "uncertainty_well", "Held-out March-October 2017 interval calibration; temporal/spatial dependence limits guarantees.", 44)

    valid = unc.loc[unc.is_observed].copy()
    valid["absolute_error_m"] = (valid.prediction_m-valid.observed_m).abs()
    covered = (valid.observed_m >= valid.lower_m) & (valid.observed_m <= valid.upper_m)
    empirical = float(covered.mean())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(valid.half_width_m, valid.absolute_error_m, color=COLORS["STGAT"], s=15, alpha=.45)
    if valid.half_width_m.std() > 1e-8:
        coefficient = np.polyfit(valid.half_width_m, valid.absolute_error_m, 1)
        xx = np.linspace(valid.half_width_m.min(), valid.half_width_m.max(), 100)
        ax.plot(xx, np.polyval(coefficient, xx), color="#555555", linestyle="--")
        corr = pearsonr(valid.half_width_m, valid.absolute_error_m).statistic
    else:
        corr = np.nan
    ax.set(xlabel="Calibrated band half-width (m)", ylabel="Absolute forecast error (m)",
           title=f"Interval diagnostic · observed coverage {empirical:.1%} · descriptive r={corr:.2f}")
    fig.tight_layout()
    save(fig, "uncertainty_vs_error", "Empirical coverage on measured benchmark targets; correlation is descriptive, not a calibration proof.", 45)

    average = unc.groupby("well_id").half_width_m.mean().reindex(data.wells)
    counts = valid.groupby("well_id").size().reindex(data.wells, fill_value=0)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.bar(data.wells, average, color=COLORS["STGAT"], alpha=.8)
    for i, count in enumerate(counts):
        if not count:
            ax.text(i, average.iloc[i], "×", ha="center", va="bottom", fontsize=15)
    ax.set(ylabel="Mean prediction-band half-width (m)", title="Uncertainty by well · × has no measured benchmark labels")
    ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    save(fig, "avg_uncertainty_per_well", "Calibrated interval widths; unscored wells remain visible.", 46)

    comparable = metrics.loc[(metrics.period == "benchmark") & metrics.common_origins & metrics.lead_months.isin([1, 3, 6])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    reference = comparable.loc[comparable.model == "Persistence"].set_index("lead_months").mae_m
    for kind in model_order:
        rows = comparable.loc[comparable.model == kind].sort_values("lead_months")
        axes[0].plot(rows.lead_months, rows.mae_m, "o-", color=COLORS[kind], label=LABELS.get(kind, kind))
        if kind in ["STGAT", "GRU", "LSTM"]:
            skill = 1-rows.set_index("lead_months").mae_m/reference
            axes[1].plot(skill.index, skill, "o-", color=COLORS[kind], label=kind)
    axes[0].set(xlabel="Lead (months)", ylabel="MAE (m)", title="Same forecast origins · measured targets at each lead", xticks=[1, 3, 6])
    axes[1].set(xlabel="Lead (months)", ylabel="MAE skill versus persistence", xticks=[1, 3, 6])
    axes[1].axhline(0, color="#777777", linestyle="--")
    axes[0].legend(fontsize=8)
    axes[1].legend()
    fig.tight_layout()
    save(fig, "horizon_comparison", "Like-for-like baseline comparison at 1, 3 and 6 months; horizon length is not used to disguise errors.", "new", False)

    calibration = []
    for lead, group in uncertainty.groupby("lead_months"):
        group = group.loc[group.is_observed & (pd.to_datetime(group.origin_month) >= data.dates[periods["benchmark"][0]-1])]
        calibration.append({"lead_months": int(lead), "n": len(group),
            "empirical_coverage": float(((group.observed_m >= group.lower_m) & (group.observed_m <= group.upper_m)).mean()),
            "mean_half_width_m": float(group.half_width_m.mean())})
    pd.DataFrame(calibration).to_csv(output/"interval_coverage.csv", index=False)
    # Reproduce the original paired test as an explicitly exploratory diagnostic.
    paired = per.pivot(index="well_id", columns="model", values="rmse_m")[["STGAT", "LSTM"]].dropna()
    difference = paired.STGAT-paired.LSTM
    if len(paired) and (difference != 0).any():
        statistic, p = wilcoxon(paired.STGAT, paired.LSTM)
        record = {"paired_wells": len(paired), "statistic": float(statistic), "uncorrected_p": float(p),
                  "caveat": "Exploratory; wells are spatially dependent and models/metrics were compared repeatedly. No superiority claim."}
    else:
        record = {"paired_wells": len(paired), "caveat": "No nonzero paired differences."}
    (output/"exploratory_wilcoxon.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    inventory = pd.DataFrame(inventory)
    inventory.to_csv(output/"plot_inventory.csv", index=False)
    if int(inventory.original_plot.sum()) != 18:
        raise AssertionError("Original plot inventory is incomplete.")
    body = ["<!doctype html><meta charset='utf-8'><title>STGAT optimization plots</title>",
            "<style>body{font:16px system-ui;margin:32px;background:#f5f7fa;color:#182735}main{max-width:1400px;margin:auto}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:24px}figure{margin:0;background:white;padding:18px;border-radius:10px}img{width:100%;height:auto}figcaption{line-height:1.5}a{color:#176ca5}</style><main>",
            "<h1>STGAT optimization and original notebook plots</h1>",
            "<p>Measured labels only. The existing benchmark was examined previously; selection uses chronological validation folds. Bands are empirically calibrated and have no exchangeability-based guarantee.</p>",
            selected[["mae_m", "rmse_m", "macro_r2"]].round(4).to_html(), "<div class='grid'>"]
    for row in inventory.itertuples():
        body.append(f"<figure><a href='{html.escape(row.png)}'><img src='{html.escape(row.png)}' loading='lazy'></a><figcaption><b>{html.escape(row.plot.replace('_',' ').title())}</b><br>{html.escape(row.description)}<br><a href='{html.escape(row.svg)}'>Vector SVG</a></figcaption></figure>")
    body.append("</div></main>")
    (plots/"index.html").write_text("\n".join(body), encoding="utf-8")
    with zipfile.ZipFile(output/"all_notebook_plots.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for file in plots.iterdir():
            archive.write(file, arcname="plots/"+file.name)
        archive.write(output/"plot_inventory.csv", arcname="plot_inventory.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output = args.run
    context = json.loads((output/"plot_context.json").read_text())
    data = SimpleNamespace(wells=context["wells"], coordinates=np.asarray(context["coordinates"]),
                           dates=pd.DatetimeIndex(context["dates"]))
    storage = f"sqlite:///{(output/'studies.sqlite3').resolve().as_posix()}"
    studies = {kind: optuna.load_study(study_name=kind, storage=storage) for kind in ["STGAT", "GRU", "LSTM"]}
    histories = {kind: pd.read_csv(output/f"{kind}_seed{context['history_seed']}_history.csv") for kind in studies}
    generate_all_plots(output, studies, histories, pd.read_csv(output/"all_forecasts.csv"),
                       pd.read_csv(output/"metrics.csv"), pd.read_csv(output/"metrics_per_well.csv"),
                       pd.read_csv(output/"well_metadata.csv"), pd.read_csv(output/"prediction_intervals.csv"),
                       pd.read_csv(output/"feature_importance.csv"), context["periods"], data)
    print(f"Regenerated the complete plot collection in {output/'plots'}")


if __name__ == "__main__":
    main()
