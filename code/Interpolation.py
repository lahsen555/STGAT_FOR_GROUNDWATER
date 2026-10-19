"""Monthly groundwater gap filling for rolling, one-month-ahead forecasting.

Fit each well's state-space parameters on the training block only, then run a
forward Kalman filter through the complete series with those parameters fixed.
Observed measurements are never replaced. Validation/test observations update
the filter only when their month is reached; this assumes measurements become
available at month end, before forecasting the next month.

Training-period estimates reuse parameters learned from the entire training
block. They are training preparation, not a walk-forward evaluation of this
imputer. For each new backtest fold, run this script with that fold's train end.
Missing target labels must be masked in STGAT training/evaluation using
``is_observed``; ``level_filled`` is suitable for historical model inputs.

Example, from any directory:
    python path/to/code/Interpolation.py --train-end 2015-07-01

Without --train-end, the first 60% of calendar months are used for fitting,
matching the current STGNN_model.py. The default 12-month prediction limit is
an explicit guard, not a validated hydrological threshold. Validate it and the
seasonal model using withheld contiguous gaps within the training data.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents


def _prepare_monthly_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Validate monthly keys and retain the original observed level column."""
    data = df.copy()
    if "level" not in data and "NP (m)" in data:
        data = data.rename(columns={"NP (m)": "level"})
    required = {"well_id", "date", "level"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(data.columns))}")
    data = data[["well_id", "date", "level"]].copy()
    if data.empty:
        raise ValueError("No groundwater rows were supplied.")
    if data["well_id"].isna().any():
        raise ValueError("well_id cannot be missing.")
    data["well_id"] = data["well_id"].astype(str).str.strip()
    if data["well_id"].eq("").any():
        raise ValueError("well_id cannot be empty.")
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data["date"].isna().any():
        raise ValueError("date cannot be missing.")
    month_starts = data["date"].dt.to_period("M").dt.to_timestamp()
    if not data["date"].eq(month_starts).all():
        raise ValueError("Dates must be month starts; aggregate daily records explicitly first.")
    if data.duplicated(["well_id", "date"]).any():
        raise ValueError("Duplicate well/month records; resolve them before gap filling.")
    data["level"] = pd.to_numeric(data["level"], errors="raise").astype(float)
    if np.isinf(data["level"].to_numpy()).any():
        raise ValueError("Levels must be finite numbers or missing.")
    return data.sort_values(["well_id", "date"]).reset_index(drop=True)


def _make_model(y: pd.Series, seasonal_period: int) -> UnobservedComponents:
    # Retain the original model specification; its superiority is not assumed.
    return UnobservedComponents(
        y,
        level="local linear trend",
        seasonal=seasonal_period,
        stochastic_seasonal=True,
    )


def seasonal_kalman_gapfill(
    df: pd.DataFrame,
    seasonal_period: int = 12,
    *,
    train_end: str | pd.Timestamp,
    min_observations: int = 24,
    max_gap: int = 12,
) -> pd.DataFrame:
    """Preserve observations and fill eligible gaps with forward predictions.

    All wells receive the same monthly grid, including absent input rows.
    ``max_gap`` is months elapsed since the last measurement: the first
    ``max_gap`` months of a longer gap may be filled, later months remain NaN.
    Leading gaps remain NaN. No backward fill or cross-well fill is used.
    Insufficient-data and failed-fit wells retain all of their observations.
    """
    for name, value, minimum in (
        ("seasonal_period", seasonal_period, 2),
        ("min_observations", min_observations, 2),
        ("max_gap", max_gap, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    data = _prepare_monthly_levels(df)
    full_index = pd.date_range(data["date"].min(), data["date"].max(), freq="MS", name="date")
    cutoff = pd.Timestamp(train_end)
    if pd.isna(cutoff) or cutoff != cutoff.to_period("M").to_timestamp():
        raise ValueError("train_end must be a month-start date.")
    if not full_index.min() <= cutoff <= full_index.max():
        raise ValueError("train_end must lie within the input date range.")

    outputs = []
    for well, group in data.groupby("well_id", sort=True):
        well_df = group.set_index("date").reindex(full_index)
        well_df["well_id"] = well
        y = well_df["level"]
        observed = y.notna()
        well_df["is_observed"] = observed
        well_df["level_filled"] = y.copy()
        well_df["imputation_method"] = np.where(observed, "observed", "unfilled")

        positions = np.arange(len(y), dtype=float)
        last_observed = pd.Series(np.where(observed, positions, np.nan), index=full_index).ffill()
        age = pd.Series(positions, index=full_index) - last_observed
        well_df["months_since_observation"] = age.astype("Int64")
        leading = ~observed & age.isna()
        too_long = ~observed & age.gt(max_gap)
        eligible = ~observed & age.between(1, max_gap)
        well_df.loc[leading, "imputation_method"] = "unfilled_leading"
        well_df.loc[too_long, "imputation_method"] = "unfilled_gap_limit"

        training = y.loc[:cutoff]
        if eligible.any():
            if training.notna().sum() < min_observations or len(training) < 2 * seasonal_period:
                well_df.loc[eligible, "imputation_method"] = "unfilled_insufficient_training"
                warnings.warn(f"Well {well}: insufficient training observations for seasonal fitting.", RuntimeWarning)
            else:
                try:
                    # Center using training data only to improve numerical conditioning.
                    center = float(training.mean())
                    fitted = _make_model(training - center, seasonal_period).fit(
                        method="powell", maxiter=1000, disp=False, cov_type="none"
                    )
                    if not fitted.mle_retvals.get("converged", False):
                        raise RuntimeError("state-space optimizer did not converge")
                    if not np.isfinite(np.asarray(fitted.params)).all() or not np.isfinite(fitted.llf):
                        raise RuntimeError("state-space fit returned non-finite parameters/likelihood")
                    filtered = _make_model(y - center, seasonal_period).filter(fitted.params)
                    # fittedvalues are one-step predictions, NOT smoothed estimates.
                    predicted = pd.Series(np.asarray(filtered.fittedvalues) + center, index=full_index)
                    fillable = eligible & np.isfinite(predicted)
                    well_df.loc[fillable, "level_filled"] = predicted.loc[fillable]
                    well_df.loc[fillable, "imputation_method"] = "kalman_prediction"
                    well_df.loc[eligible & ~fillable, "imputation_method"] = "unfilled_nonfinite_prediction"
                    if (eligible & ~fillable).any():
                        warnings.warn(f"Well {well}: non-finite predictions left unfilled.", RuntimeWarning)
                except (ValueError, RuntimeError, np.linalg.LinAlgError, FloatingPointError) as error:
                    well_df.loc[eligible, "imputation_method"] = "unfilled_model_failure"
                    warnings.warn(f"Well {well}: gap filling failed: {error}", RuntimeWarning)

        well_df["is_imputed"] = ~observed & well_df["level_filled"].notna()
        outputs.append(well_df.reset_index())

    return pd.concat(outputs, ignore_index=True)[
        ["well_id", "date", "level", "level_filled", "is_observed", "is_imputed",
         "imputation_method", "months_since_observation"]
    ]


def gap_summary(filled: pd.DataFrame, train_end: str | pd.Timestamp) -> pd.DataFrame:
    """Summarize coverage without silently dropping difficult wells."""
    rows = []
    for well, group in filled.groupby("well_id", sort=True):
        missing = group["level"].isna().to_numpy()
        changes = np.diff(np.r_[False, missing, False].astype(int))
        lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
        rows.append({
            "well_id": well, "months": len(group),
            "observed": int(group["is_observed"].sum()),
            "missing_original": int(missing.sum()),
            "missing_fraction": float(missing.mean()),
            "longest_gap_months": int(lengths.max()) if lengths.size else 0,
            "imputed": int(group["is_imputed"].sum()),
            "unfilled": int(group["level_filled"].isna().sum()),
            "training_observed": int(group.loc[group["date"] <= pd.Timestamp(train_end), "is_observed"].sum()),
            "train_end": pd.Timestamp(train_end).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=data_dir / "gwl.csv")
    parser.add_argument("--output", type=Path, default=data_dir / "final_filled.csv")
    parser.add_argument("--train-end", help="Last training month (inclusive), YYYY-MM-01.")
    parser.add_argument("--train-fraction", type=float, default=0.6, help="Used only when --train-end is omitted.")
    parser.add_argument("--max-gap", type=int, default=12, help="Maximum months since a measurement to predict.")
    parser.add_argument("--seasonal-period", type=int, default=12)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output paths must differ; preserve the observed source data.")
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be between 0 and 1.")
    levels = _prepare_monthly_levels(pd.read_csv(args.input))
    dates = pd.date_range(levels["date"].min(), levels["date"].max(), freq="MS")
    train_months = int(len(dates) * args.train_fraction)
    if args.train_end is None and train_months < 1:
        parser.error("Not enough months for the selected training fraction.")
    cutoff = pd.Timestamp(args.train_end) if args.train_end else dates[train_months - 1]
    filled = seasonal_kalman_gapfill(levels, seasonal_period=args.seasonal_period,
                                    train_end=cutoff, max_gap=args.max_gap)
    summary = gap_summary(filled, cutoff)
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    if summary_path.resolve() == args.input.resolve():
        parser.error("Summary output would overwrite the observed source data.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    filled.to_csv(args.output, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Training cutoff: {cutoff:%Y-%m-%d}; max gap: {args.max_gap} months")
    print(summary.to_string(index=False))
    print(f"Saved {args.output} and {summary_path}")
    if filled["level_filled"].isna().any():
        warnings.warn("Unfilled levels remain. Use masks or an explicit well/window selection; "
                      "global ffill/bfill in STGNN_model.py would undo these safeguards.", RuntimeWarning)


if __name__ == "__main__":
    main()
