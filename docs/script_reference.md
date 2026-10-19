# Script reference

Commands below are issued from the repository root unless stated otherwise. Use a new output directory for each new experiment. The main README gives complete commands and the scientific protocol.

| Entry point | Purpose and usage |
|---|---|
| `gee/extract_features.js` | Run in authenticated GEE Code Editor after configuring an uploaded well asset; see `gee/README.md`. Not a local Node/Python program. |
| `code/Interpolation.py` | Bounded forward Kalman reconstruction: `python code/Interpolation.py --train-end 2015-08-01 --max-gap 12 --output outputs/preprocessing/final_filled.csv` |
| `code/merging_gwl_with_features.py` | One-to-one well/month merge, trailing environmental summaries, missingness and relative input-hash audit. Use the CLI in the README. |
| `code/STGNN_model.py --optimize` | Entry point for the reported compact residual STGAT, tuned GRU/LSTM and baselines. Dispatches to `optimize_stgat.py`. |
| `code/optimize_stgat.py` | The same optimized CLI directly; use `--horizon 1` or `--horizon 6`, explicit trial/epoch/seed settings and a new `--output`. |
| `code/STGNN_model.py` without `--optimize` | Earlier reviewed architecture, retained for comparison; e.g. `python code/STGNN_model.py --prepare-only --output outputs/earlier_prepare`. It has a different older validation protocol and is not the final reported optimized model. |
| `code/stgnn_data.py` | Imported data/scaling/graph utilities. No standalone CLI. |
| `code/compare_stgat_runs.py` | Validation-only choice between H=1/H=6 followed by benchmark reporting; `--horizon1 RUN1 --horizon6 RUN6 --output NEW_REPORT`. `--previous code/model_runs/review_20261015` adds the historical comparison. |
| `code/stgat_plots.py` | Rerender all experimental plot categories from a saved run: `--run RUN`. No fitting. Updates plot snapshots and invalidates the pristine file manifest. |
| `code/finish_optimized_report.py` | Recovery/postprocessing from already frozen checkpoints, not ordinary training. It rewrites diagnostics in `--run RUN`; work on a full copy of a run under `outputs/` when using it. |
| `scripts/verify_results.py` | Check the pristine release and saved scores. Add `--checkpoints` to replay all 21 checkpoints without fitting. |
| `code/tests/test_*.py` | Synthetic-data tests via `python -m unittest discover -s code/tests -p "test_*.py" -v`. |
| `article/build_article_assets.py` | Rebuild manuscript figures/table fragments and calibration analyses from the archived runs. This updates article artifacts; preserve the reference copy first. |
| `article/build_article_assets.ipynb` | Editable Jupyter version; launch from the repository root or `article/`. Outputs are cleared. Install `requirements-notebooks.txt`. |
| `article/convert_manuscript_to_notebook.py` | Optional conversion utility: `python article/convert_manuscript_to_notebook.py article/build_article_assets.py outputs/manuscript_plots.ipynb`. Create `outputs/` first. |
| `article/manuscript_ablation.py` | Retrain the fixed selected H=1 model with its graph layer/gate removed. Writes to `article/analysis/no_gat/`; preserve the archived copy before rerunning. |

The `source_snapshot/`, `plot_source_snapshot.py` and `postprocessing_*_snapshot.py` files inside recorded runs are execution provenance, not additional entry points. Their original hashes are retained. Do not execute those copies as independent projects: they intentionally refer to the original module layout.

The pipeline starts from the included curated monthly observations and archived GEE export. It does not parse the original ABHT spreadsheet workbooks. A one-time comparison audit of the superseded 16-well interpolation implementation was excluded along with the obsolete backup implementation on which that audit depends.
