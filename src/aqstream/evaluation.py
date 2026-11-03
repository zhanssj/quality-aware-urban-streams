"""Streaming diagnostic metrics on one explicitly shared target intersection."""

import csv
import hashlib
import math
from pathlib import Path

from aqstream.data import DataContractError
from aqstream.prediction_plan import PredictionPlan

MODEL_NAMES = ("persistence", "standard", "quality_aware")


def evaluate_predictions(path: Path, *, plan: PredictionPlan | None = None) -> dict:
    if plan is None:
        plan = PredictionPlan.load(path.parent / "evaluation_plan.json")
    scheduled = eligible = common_scored = 0
    predicted = dict.fromkeys(MODEL_NAMES, 0)
    absolute = dict.fromkeys(MODEL_NAMES, 0.0)
    squared = dict.fromkeys(MODEL_NAMES, 0.0)
    per_station: dict[str, dict] = {}
    digest = hashlib.sha256()
    seen = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_id",
            "station_id",
            "origin_index",
            "target_index",
            "target_year",
            "target_local_index",
            "run_id",
            "scenario_id",
            "wait_minutes",
            "split",
            "source_valid",
            "target",
            *MODEL_NAMES,
        }
        if not required.issubset(reader.fieldnames or []):
            raise DataContractError(f"Missing prediction columns in {path}")
        for row in reader:
            plan.validate_row(row)
            key = row["sample_id"]
            if key in seen:
                raise DataContractError(f"Duplicate prediction sample ID: {key}")
            seen.add(key)
            if int(row["target_year"]) != 2024 or row["split"] != "validation":
                raise DataContractError("Evaluation is validation-only; test gate 2025 is CLOSED")
            origin, target = int(row["origin_index"]), int(row["target_index"])
            if target != origin + 1 or key != f"{row['station_id']}:{origin}:{target}":
                raise DataContractError(f"Inconsistent sample identity: {key}")
            if row["source_valid"] not in ("0", "1"):
                raise DataContractError("source_valid must be 0 or 1")
            good = row["source_valid"] == "1"
            truth = float(row["target"]) if row["target"] else None
            if good != (truth is not None and math.isfinite(truth)) or (
                not good and truth is not None
            ):
                raise DataContractError("Truth must be finite iff source_valid, otherwise empty")
            values = {name: float(row[name]) if row[name] else math.nan for name in MODEL_NAMES}
            scheduled += 1
            for name, value in values.items():
                predicted[name] += int(math.isfinite(value))
            if not good:
                continue
            eligible += 1
            if not all(math.isfinite(value) for value in values.values()):
                continue
            common_scored += 1
            digest.update((key + "\n").encode())
            station = per_station.setdefault(
                row["station_id"],
                {
                    "n": 0,
                    "absolute": dict.fromkeys(MODEL_NAMES, 0.0),
                },
            )
            station["n"] += 1
            for name, prediction in values.items():
                error = prediction - truth
                absolute[name] += abs(error)
                squared[name] += error * error
                station["absolute"][name] += abs(error)
    if scheduled != plan.expected_count:
        raise DataContractError(
            f"Incomplete expected schedule: {scheduled} present, {plan.expected_count} planned"
        )
    metrics = {}
    for name in MODEL_NAMES:
        mae = absolute[name] / common_scored if common_scored else None
        rmse = math.sqrt(squared[name] / common_scored) if common_scored else None
        if mae is not None and not (math.isfinite(mae) and math.isfinite(rmse)):
            raise DataContractError("Nonfinite metric arithmetic")
        metrics[name] = {
            "mae": mae,
            "rmse": rmse,
            "scheduled": scheduled,
            "predicted": predicted[name],
            "prediction_coverage": predicted[name] / scheduled if scheduled else None,
            "missing_predictions": scheduled - predicted[name],
            "eligible_truth": eligible,
            "scored": common_scored,
            "skipped_missing_truth": scheduled - eligible,
            "skipped_missing_any_prediction": eligible - common_scored,
            "scored_target_ids_sha256": digest.hexdigest(),
            "macro_station_mae": sum(
                station["absolute"][name] / station["n"] for station in per_station.values()
            )
            / len(per_station)
            if per_station
            else None,
        }
    return {
        "split": "validation",
        "units_verified": False,
        "units": "file_units",
        "interpretation": "diagnostic_only",
        "target_policy": "source-valid shared intersection",
        "schedule_validated": True,
        "planned": plan.expected_count,
        "present_rows": scheduled,
        "expected_plan_sha256": plan.fingerprint,
        "context": {
            "run_id": plan.run_id,
            "scenario_id": plan.scenario_id,
            "wait_minutes": plan.wait_minutes,
        },
        "models": metrics,
    }
