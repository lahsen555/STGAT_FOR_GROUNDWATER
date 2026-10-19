"""Merge monthly groundwater observations with corrected GEE features.

The groundwater panel defines the wells and forecast period. Feature columns
retain their corrected unit-labelled names. Trailing windows are calculated
on a complete monthly feature calendar BEFORE selecting the groundwater dates,
so historical feature data can support the first groundwater input windows.
Windows include the labelled month: use them after month end to forecast the
following month, not to forecast the same month before its data are available.

No target-based feature selection, scaling, temporal imputation, PET estimate,
or groundwater-ET attribution is performed here. Keep source missing values
and coverage flags for the model preprocessing review. Use is_observed and
level for supervised target masks; level_filled contains historical inputs.

Run from any directory:
    python path/to/code/merging_gwl_with_features.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


KEYS = ["well_id", "date"]
EXPORT_METADATA = ["system:index", ".geo"]
ROLLING_FEATURES = {
    "precipitation_sum_2m_mm": ("precipitation_mm", 2, "sum"),
    "Snowfall_sum_12m_mm": ("Snowfall_mm", 12, "sum"),
    "ET_MODIS_sum_12m_mm": ("ET_MODIS_mm", 12, "sum"),
    "NDVI_MODIS_mean_12m": ("NDVI_MODIS", 12, "mean"),
    "LST_mean_12m_C": ("LST_C", 12, "mean"),
    "Runoff_SubSurface_sum_12m_mm": ("Runoff_SubSurface_mm", 12, "sum"),
}
NUMERIC_FEATURES = {
    "DEF_mm", "ET_MODIS_mm", "ET_MODIS_mm_coverage", "Elevation_m", "LST_C",
    "LST_C_coverage", "Landsat_scene_count", "NDVI_Landsat", "NDVI_MODIS",
    "NDVI_MODIS_coverage", "PDSI", "Runoff_SubSurface_mm", "Runoff_Total_mm",
    "Slope_deg", "Snowfall_mm", "Snowmelt_mm", "SoilMoisture_kgm2",
    "SoilTemp_K", "TotalEvaporation_mm", "precipitation_mm",
}


def _monthly_keys(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    data = frame.copy()
    if data.empty:
        raise ValueError(f"{label}: no rows supplied.")
    if not set(KEYS).issubset(data):
        raise ValueError(f"{label}: required key columns are {KEYS}.")
    if data[KEYS].isna().any().any():
        raise ValueError(f"{label}: well_id and date cannot be missing.")
    data["well_id"] = data["well_id"].astype(str).str.strip()
    if data["well_id"].eq("").any():
        raise ValueError(f"{label}: well_id cannot be empty.")
    data["date"] = pd.to_datetime(data["date"], format="mixed", errors="raise")
    if data["date"].dt.tz is not None:
        raise ValueError(f"{label}: use timezone-naive monthly dates.")
    month_starts = data["date"].dt.to_period("M").dt.to_timestamp()
    if not data["date"].eq(month_starts).all():
        raise ValueError(f"{label}: dates must be month labels or month-start dates.")
    if data.duplicated(KEYS).any():
        raise ValueError(f"{label}: duplicate well/month keys; resolve before merging.")
    return data.sort_values(KEYS).reset_index(drop=True)


def _finite_numeric(data: pd.DataFrame, columns: set[str], label: str) -> None:
    for column in sorted(columns & set(data.columns)):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if np.isinf(data[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{label}: {column} contains infinite values.")


def _groundwater(frame: pd.DataFrame) -> pd.DataFrame:
    data = _monthly_keys(frame, "Groundwater")
    required = {"level", "level_filled", "is_observed", "is_imputed"}
    if not required.issubset(data):
        raise ValueError(f"Groundwater: missing columns {sorted(required - set(data.columns))}.")
    _finite_numeric(data, {"level", "level_filled"}, "Groundwater")
    for column in ["is_observed", "is_imputed"]:
        parsed = data[column].astype(str).str.lower().map({"true": True, "false": False})
        if parsed.isna().any():
            raise ValueError(f"Groundwater: {column} must contain True/False with no missing flags.")
        data[column] = parsed.astype(bool)
    observed = data["level"].notna()
    imputed = ~observed & data["level_filled"].notna()
    if not data["is_observed"].eq(observed).all() or not data["is_imputed"].eq(imputed).all():
        raise ValueError("Groundwater observation/imputation flags do not match the level columns.")
    if not data.loc[observed, "level"].eq(data.loc[observed, "level_filled"]).all():
        raise ValueError("Groundwater level_filled has changed measured values.")
    return data


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "Well_ID" in data:
        if "well_id" in data:
            raise ValueError("Features contain both Well_ID and well_id; use one identifier column.")
        data = data.rename(columns={"Well_ID": "well_id"})
    data = _monthly_keys(data, "Features")
    required = {spec[0] for spec in ROLLING_FEATURES.values()} | {"NDVI_Landsat", "Runoff_Total_mm"}
    if not required.issubset(data):
        raise ValueError(f"Features: missing corrected-schema columns {sorted(required - set(data.columns))}.")
    _finite_numeric(data, NUMERIC_FEATURES, "Features")
    for column in [c for c in data if c.endswith("_coverage")]:
        _finite_numeric(data, {column}, "Features")
        invalid = data[column].notna() & ~data[column].between(0, 1)
        if invalid.any():
            raise ValueError(f"Features: {column} must be a fraction between 0 and 1.")
    return data.drop(columns=EXPORT_METADATA, errors="ignore")


def merge_groundwater_features(groundwater: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the groundwater keys, with a validated one-to-one join.

    Missing feature *rows* raise an error. Missing values within existing rows
    remain missing and are reported separately. Original Landsat NDVI remains
    unchanged; an explicitly labelled NDVI_combined column provides the same-
    month MODIS fallback, with NDVI_source recording which sensor supplied it.
    The two products are not assumed to be calibrated or interchangeable.
    """
    levels = _groundwater(groundwater)
    source = _features(features)
    reserved = set(ROLLING_FEATURES) | {"NDVI_combined", "NDVI_source", "feature_row_available"}
    if reserved & set(source.columns):
        raise ValueError(f"Features already contain reserved derived columns: {sorted(reserved & set(source.columns))}.")
    overlap = (set(levels.columns) & (set(source.columns) | reserved)) - set(KEYS)
    if overlap:
        raise ValueError(f"Groundwater/feature columns overlap: {sorted(overlap)}.")

    wells = levels["well_id"].unique()
    source = source.loc[source["well_id"].isin(wells)].copy()
    if source.empty:
        raise ValueError("No feature rows match the groundwater wells.")
    first = min(source["date"].min(), levels["date"].min())
    calendar = pd.date_range(first, levels["date"].max(), freq="MS", name="date")
    index = pd.MultiIndex.from_product([wells, calendar], names=KEYS)
    source["feature_row_available"] = True
    # Reindex missing months explicitly: rolling over available rows alone
    # would treat a two-month gap as one step and produce incorrect windows.
    monthly = source.set_index(KEYS).reindex(index).reset_index()
    monthly["feature_row_available"] = monthly["feature_row_available"].eq(True)

    monthly["NDVI_combined"] = monthly["NDVI_Landsat"].combine_first(monthly["NDVI_MODIS"])
    monthly["NDVI_source"] = "missing"
    monthly.loc[monthly["NDVI_MODIS"].notna(), "NDVI_source"] = "MODIS"
    monthly.loc[monthly["NDVI_Landsat"].notna(), "NDVI_source"] = "Landsat"
    for output, (column, window, aggregation) in ROLLING_FEATURES.items():
        monthly[output] = monthly.groupby("well_id", sort=False)[column].transform(
            lambda series: series.rolling(window=window, min_periods=window).agg(aggregation)
        )

    merged = levels.merge(monthly, on=KEYS, how="left", validate="one_to_one", sort=False)
    unmatched = ~merged["feature_row_available"].fillna(False)
    if unmatched.any():
        sample = merged.loc[unmatched, KEYS].head(3).to_dict("records")
        raise ValueError(f"Missing feature rows for {int(unmatched.sum())} groundwater keys; examples: {sample}.")
    return merged.sort_values(KEYS).reset_index(drop=True)


def missingness_report(merged: pd.DataFrame) -> pd.DataFrame:
    """Return overall and per-well missingness for numeric measurements/features."""
    numeric = merged.select_dtypes(include="number").columns
    rows = []
    for well, group in [("ALL", merged), *list(merged.groupby("well_id", sort=True))]:
        for column in numeric:
            count = int(group[column].isna().sum())
            rows.append({"well_id": well, "column": column, "missing": count,
                         "rows": len(group), "missing_fraction": count / len(group)})
    return pd.DataFrame(rows)


def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--groundwater", type=Path, default=data_dir / "final_filled.csv")
    parser.add_argument("--features", type=Path, default=data_dir / "Haouz_features_corrected_v1.csv")
    parser.add_argument("--output", type=Path, default=data_dir / "gwl_and_features_advanced_sumed.csv")
    args = parser.parse_args()
    report_path = args.output.with_name(args.output.stem + "_missingness.csv")
    audit_path = args.output.with_name(args.output.stem + "_audit.json")
    inputs = {args.groundwater.resolve(), args.features.resolve()}
    if inputs & {args.output.resolve(), report_path.resolve(), audit_path.resolve()}:
        parser.error("Output paths must not overwrite either input file.")
    groundwater = pd.read_csv(args.groundwater)
    features = pd.read_csv(args.features)
    merged = merge_groundwater_features(groundwater, features)
    missingness = missingness_report(merged)
    per_well = missingness.loc[missingness["well_id"] != "ALL"]
    all_missing = per_well.loc[per_well["missing_fraction"] == 1, ["well_id", "column"]]
    audit = {
        "rows": len(merged), "wells": int(merged["well_id"].nunique()),
        "first_month": merged["date"].min().strftime("%Y-%m"),
        "last_month": merged["date"].max().strftime("%Y-%m"),
        "observed_groundwater": int(merged["is_observed"].sum()),
        "imputed_groundwater": int(merged["is_imputed"].sum()),
        "unresolved_groundwater": int(merged["level_filled"].isna().sum()),
        "unmatched_feature_rows": int((~merged["feature_row_available"]).sum()),
        "modis_ndvi_fallback_rows": int(merged["NDVI_source"].eq("MODIS").sum()),
        "entirely_missing_well_columns": all_missing.to_dict("records"),
        "partial_coverage_nonmissing_values": {
            column: int((merged[column].lt(1) & merged[column.removesuffix("_coverage")].notna()).sum())
            for column in merged if column.endswith("_coverage") and column.removesuffix("_coverage") in merged
        },
        "rolling_windows": {name: {"source": col, "months": months, "aggregation": agg}
                            for name, (col, months, agg) in ROLLING_FEATURES.items()},
        "input_sha256": {Path(os.path.relpath(path.resolve(), args.output.resolve().parent)).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in [args.groundwater, args.features]},
        "runtime_versions": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    missingness.to_csv(report_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Merged {audit['rows']} rows for {audit['wells']} wells, {audit['first_month']} to {audit['last_month']}.")
    print(f"Groundwater: {audit['observed_groundwater']} observed, {audit['imputed_groundwater']} imputed, "
          f"{audit['unresolved_groundwater']} unresolved. Unmatched feature rows: {audit['unmatched_feature_rows']}.")
    print(missingness.loc[(missingness["well_id"] == "ALL") & (missingness["missing"] > 0)].to_string(index=False))
    print(f"Saved {args.output}\nMissingness: {report_path}\nAudit: {audit_path}")


if __name__ == "__main__":
    main()
