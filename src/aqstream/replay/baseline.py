"""Load pinned, trusted, locally-produced B01 artifacts; never retrain for replay."""

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from aqstream.artifacts import file_sha256
from aqstream.data import DataContractError, HDF5Reader
from aqstream.features import COMMON_NAMES, QUALITY_NAMES, FallbackState, YearGrid
from aqstream.features.causal import readonly
from aqstream.models import Baselines
from aqstream.prediction_plan import PredictionPlan


@dataclass(frozen=True)
class FrozenBaseline:
    directory: Path
    manifest: dict
    grid: YearGrid
    fitted: FallbackState
    models: Baselines

    @property
    def validation_start(self) -> int:
        return sum(self.grid.lengths[:-1])

    @property
    def station_ids(self) -> tuple[str, ...]:
        return self.fitted.station_ids

    def evaluation_plan(self) -> PredictionPlan:
        return PredictionPlan(
            self.station_ids,
            self.validation_start,
            sum(self.grid.lengths),
            0,
            self.manifest["run_id"],
            "natural_missingness",
            0,
        )


def load_frozen_baseline(directory: Path, manifest_sha256: str) -> FrozenBaseline:
    if file_sha256(directory / "manifest.json") != manifest_sha256:
        raise DataContractError("Baseline manifest differs from pinned SHA256")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or manifest.get("task_id") != "B01":
        raise DataContractError("Replay requires a successful trusted B01 run")
    for name in (
        "models.joblib",
        "fallback.json",
        "cohort.json",
        "plan.json",
        "predictions_validation.csv",
        "metrics.json",
    ):
        record = manifest["artifacts"][name]
        if (directory / name).stat().st_size != record["bytes"] or file_sha256(
            directory / name
        ) != record["sha256"]:
            raise DataContractError(f"Frozen baseline artifact changed: {name}")
    if manifest["dependencies"]["scikit-learn"] != importlib.metadata.version("scikit-learn"):
        raise DataContractError(
            "Frozen model requires the recorded scikit-learn version; no automatic refit"
        )
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    grid = YearGrid(tuple(plan["grid_years"]), tuple(plan["grid_lengths"]))
    if grid.years[-1] != 2024:
        raise DataContractError("Baseline validation year must be 2024; test gate 2025 is CLOSED")
    state = json.loads((directory / "fallback.json").read_text(encoding="utf-8"))
    for name in ("station_ids", "medians", "train_counts", "train_years"):
        state[name] = tuple(state[name])
    fitted = FallbackState(**state)
    cohort = json.loads((directory / "cohort.json").read_text(encoding="utf-8"))
    if tuple(cohort["station_ids"]) != fitted.station_ids:
        raise DataContractError("Frozen fallback and cohort IDs differ")
    if manifest["feature_spec"]["common"] != list(COMMON_NAMES) or manifest["feature_spec"][
        "quality"
    ] != list(QUALITY_NAMES):
        raise DataContractError("Frozen feature specification differs from current common kernel")
    # Pickle is loaded only after its manifest and payload hashes match the pinned local run.
    models = joblib.load(directory / "models.joblib")
    if not isinstance(models, Baselines) or not models.is_fitted:
        raise DataContractError("Frozen artifact is not a fitted Baselines instance")
    if (
        models.standard.get_params() != models.quality_aware.get_params()
        or models.standard.get_params() != manifest["model_params"]
    ):
        raise DataContractError("Frozen model parameters differ from baseline manifest")
    return FrozenBaseline(directory, manifest, grid, fitted, models)


def cache_source(
    reader: HDF5Reader,
    baseline: FrozenBaseline,
    *,
    validation_steps: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if not 1 <= validation_steps <= baseline.grid.lengths[-1]:
        raise DataContractError("Requested validation steps exceed the frozen grid")
    stop = baseline.validation_start + validation_steps
    values = np.full((stop, len(baseline.station_ids)), np.nan)
    valid = np.zeros_like(values, dtype=bool)
    offset = 0
    references = []
    for year, length in zip(baseline.grid.years, baseline.grid.lengths, strict=True):
        info = reader.inspect(year)
        if info.shape[0] != length:
            raise DataContractError("Source year length differs from frozen baseline")
        end = min(length, stop - offset)
        for block in reader.iter_blocks(
            year=year, station_ids=baseline.station_ids, start=0, stop=end, chunk_size=chunk_size
        ):
            values[offset + block.start : offset + block.stop] = block.values
            valid[offset + block.start : offset + block.stop] = block.source_valid
        references.append(
            {
                "file": info.path.name,
                "rows_read": end,
                "size_bytes_observed": info.path.stat().st_size,
            }
        )
        offset += length
    return readonly(values), readonly(valid), references
