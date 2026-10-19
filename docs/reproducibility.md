# Experimental protocol and interpretation

The frozen reference experiments are `code/model_runs/optimized_h1_20261015` and `optimized_20261015`. Both contain input hashes, exact training-source snapshots, selected configurations, three checkpoints/model, histories, Optuna studies/trial tables, dated predictions, missing-label masks and plots. `optimized_report_20261015` combines their validation-selected lead-1 variants. `review_20261015` is an earlier corrected implementation, included for historical comparison rather than the final claim.

The optimized model ignores the precomputed Kalman values in memory. It uses raw observed history with masks, last-measurement age/reference, calendar terms and causal slope/change summaries. Feature coverage screening and scaling use only each fold's training data. Missing/reconstructed targets are never supervised or scored.

The common initial history is January 2009--December 2010. Outer training targets end August 2015; September 2015--February 2017 selects checkpoints and training horizons; March--October 2017 calibrates intervals; November 2017--January 2020 is the benchmark. Inner train/validation ends are August 2012/2013, August 2013/2014 and August 2014/2015. Only July--October 2017 has measured calibration labels. The benchmark had been examined in earlier development and is not an untouched prospective test.

Each STGAT/GRU/LSTM receives 24 trials per training horizon, with the same folds and seeds 17/42/123 for outer ensembles. H=6 emits leads 1--6 directly and is evaluated at leads 1/3/6. One-month variant selection uses pooled observed-label validation MAE and prefers H=1 on exact ties. The selected variants are STGAT H=1, GRU H=6 and LSTM H=1. All variants remain reported, including the unselected H=1 GRU with lower benchmark MAE. At longer leads, GRU/LSTM can outperform STGAT.

MAE/RMSE pool the available measured labels in metres. Macro R2 averages per-well R2 values, avoiding a misleading near-perfect pooled R2 driven by different absolute well levels. The primary comparison has 540 forecasts but 389 observed targets at 19 wells. Well 1580/52 has no observed benchmark targets; well 2122/52 contributes eight. Common-origin multi-lead scores use a different cohort and must not be mixed with the full lead-1 table.

The fixed no-GAT control uses the selected STGAT H=1 configuration and stopping rule with spatial attention/gate removed. Its three checkpoints and forecasts are under `article/analysis/no_gat/`. It was added post-hoc. Its approximately 0.606001 m MAE is close to STGAT's 0.604649 m; this does not establish a stable spatial advantage or show that limited spatial data caused the small difference.

Run-level bands are nominal 90% empirical bands, based on separately held-out, training-scale-normalized absolute residual quantiles. Manuscript assets additionally report 95% bands from the same calibration pool (70 one-month observations). Their width is fixed per well/lead, not a time-varying learned variance or MC input-noise envelope. Spatial and temporal dependence prevent an exchangeability-based coverage guarantee. Report measured coverage and full width together.

The new GEE source supplied by the author records exact collection IDs and sampling settings, but differs from the archived CSV in end month and WaPOR columns. It has not been run on GEE during packaging. MOD16A2GF and composite release timing require care when interpreting retrospective evaluation as operational forecasting. See `gee/README.md`.

## Packaging changes

The release rebases serialized path strings to repository-relative locations and adds relative merge-audit path support. The merged input audits resolve relative filenames against the audit directory, not the shell's current directory. Postprocessing understands repository-relative run input paths. The report refresh accepts relocated run directories. Manuscript scripts use their containing repository, and notebook outputs are removed.

Frozen input CSVs, checkpoints, prediction CSVs and exact training-source snapshots are preserved. Active code changes affect filesystem resolution only; the scientific forward/training equations are unchanged. `docs/source_provenance.json` records original and packaged hashes, while `docs/file_manifest.json` records the pristine release. Original source manifests continue to refer to their original snapshots rather than the portability-adjusted active files. Legacy serialized script hashes in the no-GAT protocol document its execution history; the portable current entry point is `article/manuscript_ablation.py`.

Run the verifier before modifying/rebuilding anything. Regenerating figures or data changes the release manifest by design. Save new experiments under `outputs/` and choose a new output directory for each run.
