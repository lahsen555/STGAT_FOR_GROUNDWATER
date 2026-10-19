# Local release verification - 19 October 2026

- A fresh Python 3.12 environment was installed from `requirements-tested.txt`. The complete installed package list is recorded in `validation_environment.txt`; no virtual environment is distributed.
- All **47 regression tests passed**, including the added relocation/integrity regression. Existing tests cover interpolation, merging, observation masks, chronological preprocessing, graph isolation, attention edge attributes, masked losses, multi-horizon calibration and checkpoint restoration. See `test_results.txt`.
- The saved report was independently recomputed from its aligned prediction CSV: **389 measured targets at 19 wells**, with the unobserved targets excluded. Recorded metrics agreed.
- All **18 optimized neural checkpoints** and **three no-GAT checkpoints** were loaded in the fresh environment and their ensemble forecasts compared with archived dated forecasts. The optimized-model maximum discrepancy was approximately 5.7e-14 m; the allowed cross-platform replay tolerance is 1e-5 m. No training or horizon selection was performed during this replay. See `checkpoint_verification.txt`.
- The merging CLI was run on the supplied reconstruction and environmental export into a separate output directory. Its numerical panel was checked against the archived panel, and the regenerated relative-path input audit was verified. The archived files were retained.
- The earlier reviewed CLI's `--prepare-only` path was exercised as a compatibility check. Its older validation dates in `preparation_smoke_test.txt` are not the final optimized selection/calibration calendar; use the protocol in `reproducibility.md` for the reported model.
- `node --check gee/extract_features.js` passed. This validates JavaScript syntax only. Authenticated Earth Engine execution, asset permissions and a new feature export were not performed.
- The manuscript source and PDF were copied from the current user-edited article. This release preparation does not recompile or revise it.

The full 24-trial-per-family optimization was not rerun for packaging. The supplied forecasts, checkpoints, input CSVs and training-source snapshots were preserved. New trial runs should use a new directory under `outputs/`.

Git's untracked-file inventory was checked in an empty local staging repository: all 306 distributed files are included, while generated outputs/caches are excluded. This local `.git` directory is not distributed.

GitHub Actions is configured but has not been executed on GitHub. No GitHub repository, commit, release or public upload was created. The authors still need to choose a code license and confirm redistribution terms for the agency observations before a public push.
