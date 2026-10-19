"""Training-only preprocessing and calendar-aware groundwater forecast splits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_FEATURES = ["ET_MODIS_mm", "LST_C", "NDVI_MODIS", "Runoff_Total_mm", "precipitation_mm"]


@dataclass
class ForecastData:
    dates: pd.DatetimeIndex
    wells: list[str]
    inputs: np.ndarray             # (calendar time, nodes, channels), finite
    targets: np.ndarray            # standardized observed levels; zero elsewhere
    measured: np.ndarray           # original measured heads (NaN allowed)
    target_mask: np.ndarray
    anchor_m: np.ndarray            # causal last measured head, for persistence
    anchor_z: np.ndarray
    age: np.ndarray                 # months since last measurement
    target_mean: np.ndarray
    target_scale: np.ndarray
    value_mean: np.ndarray
    value_scale: np.ndarray
    coordinates: np.ndarray         # latitude, longitude, ordered like nodes
    features: list[str]
    channels: list[str]
    train_stop: int
    val_stop: int
    lookback: int
    horizon: int
    starts: dict[str, np.ndarray]
    feature_report: list[dict]

    def preprocessing(self) -> dict:
        return {
            "wells": self.wells, "physical_features": self.features, "input_channels": self.channels,
            "target_mean": self.target_mean.tolist(), "target_scale": self.target_scale.tolist(),
            "value_mean": self.value_mean.tolist(), "value_scale": self.value_scale.tolist(),
            "train_end": str(self.dates[self.train_stop - 1].date()),
            "validation_end": str(self.dates[self.val_stop - 1].date()),
            "lookback": self.lookback, "horizon": self.horizon,
            "feature_coverage_training_only": self.feature_report,
        }

    def target_indices(self, split: str) -> np.ndarray:
        return self.starts[split][:, None] + np.arange(self.horizon)[None, :]

    def coverage(self) -> pd.DataFrame:
        rows = []
        for split in self.starts:
            indices = self.target_indices(split)
            for node, well in enumerate(self.wells):
                count = int(self.target_mask[indices, node].sum())
                rows.append({"split": split, "well_id": well, "forecast_windows": len(indices),
                             "measured_targets": count,
                             "unmeasured_targets": int(indices.size - count),
                             "score_available": count > 0})
        return pd.DataFrame(rows)


def _boolean(series: pd.Series, name: str) -> np.ndarray:
    parsed = series.astype(str).str.lower().map({"true": True, "false": False})
    if parsed.isna().any():
        raise ValueError(f"{name} must contain True/False with no missing flags.")
    return parsed.to_numpy(dtype=bool)


def _training_stats(values: np.ndarray, stop: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-node/channel statistics; unseen node/channels use training-wide stats."""
    training = values[:stop]
    valid = np.isfinite(training)
    count = valid.sum(axis=0)
    sums = np.where(valid, training, 0).sum(axis=0)
    mean = sums / np.maximum(count, 1)
    global_count = valid.sum(axis=(0, 1))
    global_mean = sums.sum(axis=0) / np.maximum(global_count, 1)
    mean = np.where(count > 0, mean, global_mean[None, :])
    variance = np.where(valid, (training - mean) ** 2, 0).sum(axis=0) / np.maximum(count, 1)
    global_variance = np.where(valid, (training - global_mean) ** 2, 0).sum(axis=(0, 1)) / np.maximum(global_count, 1)
    variance = np.where(count > 0, variance, global_variance[None, :])
    scale = np.sqrt(np.maximum(variance, 0))
    return mean, np.where(scale > 1e-6, scale, 1.0)


def prepare_data(
    frame: pd.DataFrame, metadata: pd.DataFrame, *, lookback: int = 12, horizon: int = 1,
    train_fraction: float = .6, val_fraction: float = .2, train_end: str | None = None,
    features: list[str] | None = None, min_feature_coverage: float = .8,
) -> ForecastData:
    """Mask labels and inputs separately; never fill across wells or from the future.

    A missing input is represented by standardized zero plus availability=0.
    This is a numerical placeholder, not an estimated physical measurement.
    Existing finite Kalman history values are allowed as inputs. Only original
    measured levels supervise training and contribute to validation/test scores.
    """
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
           for value in [lookback, horizon]):
        raise ValueError("lookback and horizon must be positive integers.")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1 or train_fraction + val_fraction >= 1:
        raise ValueError("Training/validation fractions must be positive and sum to less than 1.")
    if not 0 <= min_feature_coverage <= 1:
        raise ValueError("min_feature_coverage must be between 0 and 1.")
    data = frame.copy()
    required = {"well_id", "date", "level", "level_filled", "is_observed", "is_imputed"}
    if not required.issubset(data):
        raise ValueError(f"Missing groundwater columns: {sorted(required - set(data.columns))}")
    if data.empty or data[["well_id", "date"]].isna().any().any():
        raise ValueError("Groundwater keys cannot be empty or missing.")
    data["well_id"] = data["well_id"].astype(str).str.strip()
    data["date"] = pd.to_datetime(data["date"], format="mixed", errors="raise")
    if data["well_id"].eq("").any() or not data.date.eq(data.date.dt.to_period("M").dt.to_timestamp()).all():
        raise ValueError("Use nonempty well IDs and month-start dates.")
    if data.duplicated(["well_id", "date"]).any():
        raise ValueError("Duplicate well/month rows must be resolved before training.")
    for col in ["level", "level_filled"]:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if np.isinf(data[col].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values in {col}.")
    observed = _boolean(data["is_observed"], "is_observed")
    imputed = _boolean(data["is_imputed"], "is_imputed")
    if not np.array_equal(observed, data["level"].notna()) or not np.array_equal(
        imputed, ~observed & data["level_filled"].notna()
    ):
        raise ValueError("Observation/imputation masks do not match the level columns.")
    if not data.loc[observed, "level"].eq(data.loc[observed, "level_filled"]).all():
        raise ValueError("Measured groundwater values have been replaced in level_filled.")
    data["is_observed"] = observed
    dates = pd.date_range(data.date.min(), data.date.max(), freq="MS", name="date")
    train_stop = int(len(dates) * train_fraction)
    if train_end is not None:
        cutoff = pd.Timestamp(train_end)
        if cutoff not in dates:
            raise ValueError("train_end must be a month-start date within the data range.")
        train_stop = int(dates.get_loc(cutoff)) + 1
    val_stop = int(len(dates) * (train_fraction + val_fraction))
    if not lookback + horizon <= train_stop < val_stop < len(dates):
        raise ValueError("Not enough months for the lookback/horizon and three date splits.")
    wells = sorted(data.well_id.unique().tolist())
    grid = pd.MultiIndex.from_product([dates, wells], names=["date", "well_id"])
    panel = data.set_index(["date", "well_id"]).reindex(grid)
    shape = (len(dates), len(wells))
    measured = panel["level"].to_numpy(dtype=float).reshape(shape)
    mask = panel["is_observed"].eq(True).to_numpy(copy=True).reshape(shape)
    filled = panel["level_filled"].to_numpy(dtype=float).reshape(shape)
    counts = mask[:train_stop].sum(axis=0)
    if (counts < 2).any():
        raise ValueError(f"Need >=2 measured training levels per node: {np.array(wells)[counts < 2].tolist()}")
    target_mean, target_scale = _training_stats(measured[:, :, None], train_stop)
    target_mean, target_scale = target_mean[:, 0], target_scale[:, 0]
    targets = np.where(mask, (measured - target_mean) / target_scale, 0).astype(np.float32)

    candidates = DEFAULT_FEATURES if features is None else features
    if len(set(candidates)) != len(candidates) or {"level", "level_filled", "is_observed", "is_imputed"} & set(candidates):
        raise ValueError("Specify unique exogenous numeric feature names; groundwater history is included automatically.")
    kept, report = [], []
    for col in candidates:
        if col not in panel:
            raise ValueError(f"Requested feature {col!r} is absent from the merged CSV.")
        panel[col] = pd.to_numeric(panel[col], errors="raise")
        array = panel[col].to_numpy(dtype=float).reshape(shape)
        if np.isinf(array).any():
            raise ValueError(f"Infinite feature values in {col}.")
        fraction = float(np.isfinite(array[:train_stop]).mean())
        use = fraction > 0 and fraction >= min_feature_coverage
        report.append({"feature": col, "training_coverage": fraction, "retained": use})
        if use:
            kept.append(col)
    physical = ["level_filled", *kept]
    values = np.stack([filled, *[panel[c].to_numpy(dtype=float).reshape(shape) for c in kept]], axis=2)
    mean, scale = _training_stats(values, train_stop)
    mean[:, 0], scale[:, 0] = target_mean, target_scale
    available = np.isfinite(values)
    standardized = np.where(available, (values - mean) / scale, 0)

    # Causal last-measurement history is an explicitly marked persistence
    # reference. It does not fill missing labels or replace the level_filled input.
    positions = np.arange(len(dates), dtype=float)
    last_positions = pd.DataFrame(np.where(mask, positions[:, None], np.nan)).ffill().to_numpy()
    age = positions[:, None] - last_positions
    anchor_m = pd.DataFrame(measured).ffill().to_numpy()
    anchor_z = np.where(np.isfinite(anchor_m), (anchor_m - target_mean) / target_scale, 0)
    month = dates.month.to_numpy()
    sin_month = np.broadcast_to(np.sin(2*np.pi*(month-1)/12)[:, None], shape)
    cos_month = np.broadcast_to(np.cos(2*np.pi*(month-1)/12)[:, None], shape)
    channels = [*physical, *[c + "_available" for c in physical], "gwl_is_observed",
                "gwl_measurement_age_capped_48m", "last_measured_level_z", "sin_month", "cos_month"]
    blocks = [standardized, available.astype(float), mask[:, :, None].astype(float),
              (np.minimum(np.nan_to_num(age, nan=48), 48)/48)[:, :, None],
              anchor_z[:, :, None], sin_month[:, :, None], cos_month[:, :, None]]
    for feature in kept:
        coverage_col = feature + "_coverage"
        if coverage_col in panel:
            coverage = pd.to_numeric(panel[coverage_col], errors="raise").to_numpy(dtype=float).reshape(shape)
            if ((np.isfinite(coverage)) & ((coverage < 0) | (coverage > 1))).any() or np.isinf(coverage).any():
                raise ValueError(f"Invalid coverage fractions in {coverage_col}.")
            blocks.extend([np.nan_to_num(coverage, nan=0)[:, :, None], np.isfinite(coverage)[:, :, None]])
            channels.extend([coverage_col, coverage_col + "_available"])
    inputs = np.concatenate(blocks, axis=2).astype(np.float32)
    if not np.isfinite(inputs).all():
        raise ValueError("Preprocessing produced non-finite model inputs.")

    if {"Well_ID", "well_id"}.issubset(metadata):
        raise ValueError("Metadata has ambiguous Well_ID and well_id columns.")
    meta = metadata.rename(columns={"Well_ID": "well_id"}).copy()
    if not {"well_id", "latitude", "longitude"}.issubset(meta):
        raise ValueError("Well metadata requires Well_ID/well_id, latitude, longitude.")
    meta["well_id"] = meta.well_id.astype(str).str.strip()
    meta = meta.loc[meta.well_id.isin(wells)]
    if meta.well_id.duplicated().any() or set(meta.well_id) != set(wells):
        raise ValueError("Missing or duplicate coordinates for selected groundwater wells.")
    coordinates = meta.set_index("well_id").loc[wells, ["latitude", "longitude"]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or (np.abs(coordinates) > [90, 180]).any():
        raise ValueError("Invalid latitude/longitude coordinates.")
    starts = {}
    for name, lo, hi in [("train", 0, train_stop), ("val", train_stop, val_stop), ("test", val_stop, len(dates))]:
        # Every target in a multi-month horizon must remain in its own split.
        starts[name] = np.arange(max(lookback, lo), hi - horizon + 1, dtype=int)
        if not len(starts[name]) or not mask[starts[name][:, None]+np.arange(horizon)].any():
            raise ValueError(f"No measured target windows in {name}.")
    return ForecastData(dates, wells, inputs, targets, measured, mask, anchor_m, anchor_z, age,
                        target_mean, target_scale, mean, scale, coordinates, physical, channels,
                        train_stop, val_stop, lookback, horizon, starts, report)


def spatial_graph(coordinates: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric geographic k-nearest-neighbour graph with proximity edge features."""
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be positive.")
    radians = np.radians(coordinates)
    dlat = radians[:, 0, None] - radians[None, :, 0]
    dlon = radians[:, 1, None] - radians[None, :, 1]
    a = np.sin(dlat/2)**2 + np.cos(radians[:, 0, None])*np.cos(radians[None, :, 0])*np.sin(dlon/2)**2
    distances = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    pairs = set()
    for target in range(len(coordinates)):
        neighbors = [j for j in np.argsort(distances[target], kind="stable") if j != target][:k]
        for neighbor in neighbors:
            pairs.update([(int(neighbor), target), (target, int(neighbor))])
    edges = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2).T
    edge_distances = distances[edges[0], edges[1]]
    nonzero = edge_distances[edge_distances > 0]
    width = float(np.median(nonzero)) if len(nonzero) else 1.0
    attributes = np.exp(-(edge_distances / width)**2).astype(np.float32)[:, None]
    return edges, attributes, edge_distances


def resolve_imputation_cutoff(input_path: Path, summary_path: Path | None,
                              requested_train_end: str | None) -> tuple[str | None, dict]:
    """Respect the actual imputer fit period and verify merge-source hashes."""
    details = {"imputation_summary_verified": False, "merge_sources_verified": False}
    audit_path = input_path.with_name(input_path.stem + "_audit.json")
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        for source, expected in audit.get("input_sha256", {}).items():
            path = Path(source)
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("Merge sources changed; regenerate the merged CSV before training.")
        details["merge_sources_verified"] = bool(audit.get("input_sha256"))
    if summary_path is not None and summary_path.exists():
        summary = pd.read_csv(summary_path)
        if not {"well_id", "train_end"}.issubset(summary) or summary.train_end.isna().any():
            raise ValueError("Invalid interpolation summary.")
        fit_end = pd.to_datetime(summary.train_end, errors="raise").max()
        if requested_train_end is not None and pd.Timestamp(requested_train_end) < fit_end:
            raise ValueError(f"Imputation used data through {fit_end:%Y-%m-%d}; rerun interpolation for an earlier split.")
        details.update({"imputation_summary_verified": True, "imputer_fit_end": str(fit_end.date())})
        return requested_train_end or str(fit_end.date()), details
    return requested_train_end, details
