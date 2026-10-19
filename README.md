# Groundwater forecasting with STGAT: Haouz aquifer

Code, data, trained checkpoints and saved predictions for the groundwater forecasting study. The main model is a compact temporal GRU followed by one gated GATv2 layer and a persistence-residual forecast head. Temporal GRU/LSTM, persistence, seasonal and damped-trend baselines are included.

## Repository contents

```text
code/                 Interpolation, merging, model, optimization and plotting scripts
  tests/              Synthetic-data regression tests
  model_runs/         Recorded experiments: configurations, checkpoints, predictions,
                      source snapshots, search histories and plot galleries
data/                 Curated input data, corrected GEE export and processed model panel
gee/                  Earth Engine JavaScript, uploadable well coordinates and instructions
article/              Current manuscript source/PDF and figure scripts/notebook
  analysis/no_gat/    Saved fixed-configuration graph ablation
docs/                 Data inventory, provenance, verification logs and release notes
scripts/              Verification of saved results and optional checkpoint replay
outputs/              Your new experiments; created when running commands, ignored by Git
```

The original spreadsheet workbooks, superseded `.bak` implementations and the one-time legacy 16-well comparison audit are not required for reproduction and are excluded. The extracted monthly observations and all current regression tests are included. See [docs/script_reference.md](docs/script_reference.md) for the entry points.

## 1. Create a Python environment

The recorded experiments used Python 3.12. Run these commands from the repository root.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-tested.txt
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tested.txt
```

`requirements-tested.txt` records the versions of the original CPU experiments. `requirements.txt` provides broader dependency ranges for a fresh compatible environment; other versions/hardware can produce small numerical differences. For optional Jupyter editing, also install `requirements-notebooks.txt`. No Earth Engine Python package is needed: the exporter runs separately in its JavaScript Code Editor.

## 2. Verify the existing results first

These commands do not train models or require an Earth Engine account:

```bash
python scripts/verify_results.py
python -m unittest discover -s code/tests -p "test_*.py" -v
```

The first command verifies file/data/source hashes, raw observation labels and the aggregate saved metrics. To reconstruct the forecasts from all 18 optimized neural checkpoints plus the three no-GAT checkpoints:

```bash
python scripts/verify_results.py --checkpoints
```

Verification logs from release preparation are in `docs/`. The manifest intentionally checks the packaged files byte-for-byte; if you edit files or regenerate outputs, preserve a pristine copy for this verification. `.gitattributes` disables automatic newline conversion so Git checkout preserves the recorded hashes.

## 3. Data and missing values

The supplied CSVs allow the Python pipeline to run offline. See [data/README.md](data/README.md), [docs/data_inventory.json](docs/data_inventory.json), and [docs/reproducibility.md](docs/reproducibility.md).

- `gwl.csv`: the 20-well curated groundwater input, not the older 16-well `gwl_data.csv`.
- `Haouz_features_corrected_v1.csv`: the archived corrected environmental export.
- `wells_meta.csv`: coordinates; contains 22 candidate wells, of which the model selects the 20 in the groundwater panel.
- `final_filled.csv` and its summary: bounded forward Kalman reconstruction.
- `gwl_and_features_advanced_sumed.csv`: merged 20-well, 133-month panel, January 2009--January 2020.

The merged panel has 1,789 measured groundwater values, 762 reconstructed values and 109 unresolved entries. `level` and `is_observed` define supervision/scoring; `level_filled` does not turn reconstructed entries into observations. The optimized model ignores precomputed Kalman values in memory and uses measured history, masks and causal summaries. Well 1580/52 remains a graph node but has no measured benchmark target and is unscored.

## 4. Rebuild interpolation and merging

Preserve the archived files when checking reproducibility; write rebuilt outputs separately:

```bash
python code/Interpolation.py --input data/gwl.csv --output outputs/preprocessing/final_filled.csv --train-end 2015-08-01 --max-gap 12
python code/merging_gwl_with_features.py --groundwater outputs/preprocessing/final_filled.csv --features data/Haouz_features_corrected_v1.csv --output outputs/preprocessing/gwl_and_features_advanced_sumed.csv
```

The interpolation command also creates `final_filled_summary.csv`. The merging command writes missingness and input-hash audits alongside the merged CSV. The audit paths are relative to the merged file, so the folder can be moved. Never use global backward filling or score interpolated labels. Rebuilding a CSV with another package version may change its byte hash even when values agree; new experiments should use the corresponding regenerated audit and summary.

## 5. Run training and validation

**Use `--optimize` for the reported compact STGAT.** Running `STGNN_model.py` without this flag invokes the earlier reviewed architecture, retained for transparency.

The full protocol uses 24 Optuna trials per neural family and horizon, three chronological folds, up to 70 epochs per inner fit, up to 140 epochs per outer fit, and ensemble seeds 17, 42 and 123. It can take substantially longer than verification. Each output directory must be new.

```bash
python code/STGNN_model.py --optimize --horizon 1 --trials 24 --cv-epochs 70 --epochs 140 --seeds 17 42 123 --threads 2 --output outputs/optimized_h1
python code/STGNN_model.py --optimize --horizon 6 --trials 24 --cv-epochs 70 --epochs 140 --seeds 17 42 123 --threads 2 --output outputs/optimized_h6
python code/compare_stgat_runs.py --horizon1 outputs/optimized_h1 --horizon6 outputs/optimized_h6 --output outputs/comparison
```

By default these use the archived `data/` inputs. To train on the rebuilt panel from step 4, add these options to **both** optimization commands:

```text
--input outputs/preprocessing/gwl_and_features_advanced_sumed.csv --imputation-summary outputs/preprocessing/final_filled_summary.csv --metadata data/wells_meta.csv
```

For a brief execution check, reduce `--trials`, `--cv-epochs` and `--epochs`; such a run does **not** reproduce the reported tuning or scores. The optimized entry point chooses CUDA if available, otherwise CPU; use the recorded CPU environment to approximate the original execution conditions. Further details and selection rules are in [code/OPTIMIZATION_PROTOCOL.txt](code/OPTIMIZATION_PROTOCOL.txt).

## 6. Reproduce plots and manuscript assets

Existing experiment galleries are available locally at:

- `code/model_runs/optimized_h1_20261015/plots/index.html`
- `code/model_runs/optimized_20261015/plots/index.html`
- `code/model_runs/optimized_report_20261015/index.html`

Rerender a run without fitting:

```bash
python code/stgat_plots.py --run code/model_runs/optimized_h1_20261015
```

Plot regeneration updates files and will invalidate the pristine release manifest. For the manuscript-specific tables, figures and 90%/95% calibration summaries:

```bash
python article/build_article_assets.py
```

Alternatively, install the notebook dependencies, start Jupyter from the repository root and open `article/build_article_assets.ipynb`. The notebook's outputs were cleared for publication. The script/notebook use archived predictions, not new training runs. Generated table fragments are retained under `article/tables/`; the current `main.tex` embeds table rows for compatibility with the journal class, so numerical manuscript edits should be reviewed when changing experiments.

The no-GAT ablation is already saved under `article/analysis/no_gat/`. `python article/manuscript_ablation.py` retrains that fixed configuration and overwrites its regenerated ablation outputs; make a copy first if retaining the reference files. The ablation is post-hoc, not an untouched independent test.

Optional LaTeX compilation, using an installed TeX Live/MiKTeX distribution:

```bash
cd article
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The current manuscript and its supporting style files were copied as supplied; this packaging step does not revise the article.

## 7. Export environmental features with GEE

Use [gee/README.md](gee/README.md) to upload `gee/wells.csv`, set the asset ID in `gee/extract_features.js`, and run the export in an authenticated Earth Engine Code Editor. The historical CSV is included so editors do not need access to the author's asset.

The author supplied the JavaScript during release preparation. Its collection IDs, quality masks and overlap-weighted aggregation are documented there. A fresh export is a new data artifact: the supplied script includes WaPOR fields and December 2022, whereas the archived CSV has neither. It has been syntax-checked locally, not executed against the authenticated GEE service during packaging. Do not silently replace the archived export when reproducing the paper.

## Reported one-month comparison

Validation chooses STGAT H=1, GRU H=6 and LSTM H=1. Every score below uses the same 389 measured targets at 19 wells, November 2017--January 2020:

| Predictor | MAE (m) | RMSE (m) |
|---|---:|---:|
| Persistence | 0.619152 | 1.083073 |
| STGAT, selected | 0.604649 | 1.064433 |
| GRU, selected | 0.610412 | 1.075375 |
| LSTM, selected | 0.627236 | 1.078604 |

All six neural variants are visible in `all_variants.csv`: the unselected H=1 GRU has lower benchmark MAE (0.600149 m), and STGAT does not lead every metric or longer forecast lead. The fixed-configuration no-GAT MAE is about 0.606001 m. These are modest differences on a previously examined benchmark; see [docs/reproducibility.md](docs/reproducibility.md) for their interpretation.

