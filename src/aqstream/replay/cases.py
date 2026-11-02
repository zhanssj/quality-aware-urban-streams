"""Per-policy immutable prediction records, shared-context completeness and physical timers."""

import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np

from aqstream.artifacts import write_json
from aqstream.data import DataContractError
from aqstream.evaluation import MODEL_NAMES, evaluate_predictions
from aqstream.prediction_plan import PredictionPlan
from aqstream.replay.baseline import FrozenBaseline
from aqstream.replay.calendars import Calendar
from aqstream.replay.stream import ReplaySnapshot, replay_snapshots


class CaseWriter:
    def __init__(
        self,
        directory: Path,
        *,
        baseline: FrozenBaseline,
        calendar: Calendar,
        plan: PredictionPlan,
        values: np.ndarray,
        source_valid: np.ndarray,
        batch_steps: int,
    ):
        directory.mkdir(parents=True, exist_ok=False)
        self.directory, self.baseline, self.calendar, self.plan = (
            directory,
            baseline,
            calendar,
            plan,
        )
        self.values, self.source_valid, self.batch_steps = values, source_valid, batch_steps
        self.pending: list[ReplaySnapshot] = []
        self.batch_id = 0
        self.available_lag1 = 0
        self.forecast_seconds = self.receive_seconds = 0.0
        self.inference_seconds = dict.fromkeys(MODEL_NAMES, 0.0)
        plan.save(directory / "evaluation_plan.json")
        self._prediction_file = (directory / "predictions.csv").open(
            "x", encoding="utf-8", newline=""
        )
        self._timing_file = (directory / "processing_batches.csv").open(
            "x", encoding="utf-8", newline=""
        )
        self._writer = csv.DictWriter(
            self._prediction_file,
            fieldnames=[
                "run_id",
                "scenario_id",
                "wait_minutes",
                "sample_id",
                "station_id",
                "origin_index",
                "target_index",
                "target_year",
                "target_local_index",
                "split",
                "origin_minutes",
                "cutoff_minutes",
                "target_end_minutes",
                "remaining_lead_minutes",
                "source_valid",
                "target",
                *MODEL_NAMES,
                "has_history",
                "persistence_fallback",
                "processing_batch_id",
            ],
        )
        self._writer.writeheader()
        self._timing_writer = csv.DictWriter(
            self._timing_file,
            fieldnames=[
                "processing_batch_id",
                "target_start",
                "target_stop",
                "samples",
                "wait_minutes",
                "forecast_construction_seconds",
                "delivery_processing_seconds",
                *[f"{name}_inference_seconds" for name in MODEL_NAMES],
            ],
        )
        self._timing_writer.writeheader()

    def add(self, snapshot: ReplaySnapshot) -> None:
        if snapshot.wait_minutes != self.plan.wait_minutes:
            raise DataContractError("Snapshot wait differs from planned policy")
        self.pending.append(snapshot)
        self.available_lag1 += int(snapshot.features.quality[:, 0].sum())
        self.forecast_seconds += snapshot.features.compute_seconds
        self.receive_seconds += snapshot.receive_seconds
        if len(self.pending) >= self.batch_steps:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        common = np.concatenate([snapshot.features.common for snapshot in self.pending])
        quality = np.concatenate([snapshot.features.quality for snapshot in self.pending])
        persistence = np.concatenate([snapshot.features.persistence for snapshot in self.pending])
        predictions, elapsed = self.baseline.models.predict(common, quality, persistence)
        for name in MODEL_NAMES:
            self.inference_seconds[name] += elapsed[name]
        index = 0
        for snapshot in self.pending:
            frame = snapshot.features
            for column, identity in enumerate(frame.sample_ids):
                # Source truth is joined only after all models have returned predictions.
                good = bool(self.source_valid[identity.target_index, column])
                has_history = int(frame.quality[column, -1])
                row = {
                    "run_id": self.plan.run_id,
                    "scenario_id": self.plan.scenario_id,
                    "wait_minutes": self.plan.wait_minutes,
                    "sample_id": identity.key,
                    **asdict(identity),
                    "split": identity.split,
                    "origin_minutes": snapshot.origin_minutes,
                    "cutoff_minutes": snapshot.cutoff_minutes,
                    "target_end_minutes": snapshot.target_end_minutes,
                    "remaining_lead_minutes": snapshot.remaining_lead_minutes,
                    "source_valid": int(good),
                    "target": float(self.values[identity.target_index, column]) if good else "",
                    "has_history": has_history,
                    "persistence_fallback": 1 - has_history,
                    "processing_batch_id": self.batch_id,
                }
                row.update({name: float(predictions[name][index]) for name in MODEL_NAMES})
                self._writer.writerow(row)
                index += 1
        self._timing_writer.writerow(
            {
                "processing_batch_id": self.batch_id,
                "target_start": self.pending[0].features.sample_ids[0].target_index,
                "target_stop": self.pending[-1].features.sample_ids[0].target_index + 1,
                "samples": len(common),
                "wait_minutes": self.plan.wait_minutes,
                "forecast_construction_seconds": sum(
                    item.features.compute_seconds for item in self.pending
                ),
                "delivery_processing_seconds": sum(item.receive_seconds for item in self.pending),
                **{f"{name}_inference_seconds": elapsed[name] for name in MODEL_NAMES},
            }
        )
        self.batch_id += 1
        self.pending.clear()

    def close(self):
        self._prediction_file.close()
        self._timing_file.close()

    def finish(self) -> dict:
        self.flush()
        self.close()
        metrics = evaluate_predictions(self.directory / "predictions.csv", plan=self.plan)
        write_json(self.directory / "metrics.json", metrics)
        result = {
            "case_id": self.directory.name,
            "scenario_id": self.calendar.scenario_id,
            "scenario_name": self.calendar.spec.name,
            "seed": self.calendar.seed,
            "wait_minutes": self.plan.wait_minutes,
            "remaining_lead_minutes": 60 - self.plan.wait_minutes,
            "target_start": self.plan.target_start,
            "target_stop": self.plan.target_stop,
            "expected_plan_sha256": self.plan.fingerprint,
            "calendar_sha256": self.calendar.fingerprint,
            "statistics": self.calendar.statistics,
            "lag1_available_at_cutoff": self.available_lag1,
            "lag1_available_fraction_all_planned_inputs": self.available_lag1
            / self.plan.expected_count,
            "lag1_not_delivered_after_drop": self.calendar.statistics["source_valid_input_cells"]
            - self.calendar.statistics["artificially_lost_source_valid"]
            - self.available_lag1,
            "timings": {
                "forecast_construction_seconds": self.forecast_seconds,
                "delivery_processing_seconds": self.receive_seconds,
                "inference_seconds": self.inference_seconds,
            },
            "metrics": metrics,
        }
        if result["lag1_not_delivered_after_drop"] < 0:
            raise DataContractError("Availability accounting exceeded eligible input messages")
        write_json(self.directory / "case.json", result)
        return result


def replay_calendar(
    directory: Path,
    *,
    baseline: FrozenBaseline,
    values: np.ndarray,
    source_valid: np.ndarray,
    calendar: Calendar,
    run_id: str,
    target_start: int,
    target_stop: int,
    waits: tuple[int, ...],
    batch_steps: int,
    memory,
) -> list[dict]:
    writers = {}
    try:
        for wait in waits:
            plan = PredictionPlan(
                baseline.station_ids,
                target_start,
                target_stop,
                baseline.grid.point(target_start)[1],
                run_id,
                calendar.scenario_id,
                wait,
            )
            key = f"{calendar.scenario_id}_wait{wait}"
            writers[wait] = CaseWriter(
                directory / key,
                baseline=baseline,
                calendar=calendar,
                plan=plan,
                values=values,
                source_valid=source_valid,
                batch_steps=batch_steps,
            )
        for snapshot in replay_snapshots(
            values,
            source_valid,
            baseline.fitted,
            baseline.grid,
            calendar,
            target_start=target_start,
            target_stop=target_stop,
            waits=waits,
        ):
            writers[snapshot.wait_minutes].add(snapshot)
            if len(writers[snapshot.wait_minutes].pending) == 0:
                memory.check()
        return [writer.finish() for writer in writers.values()]
    finally:
        for writer in writers.values():
            writer.close()


def compare_with_baseline(
    case_directory: Path, baseline_csv: Path, *, tolerance: float = 1e-10
) -> dict:
    """Compare every scheduled prediction/truth, including source-invalid targets."""
    plan = PredictionPlan.load(case_directory / "evaluation_plan.json")
    maxima = dict.fromkeys(MODEL_NAMES, 0.0)
    compared = 0
    with (
        baseline_csv.open(encoding="utf-8", newline="") as reference_handle,
        (case_directory / "predictions.csv").open(encoding="utf-8", newline="") as actual_handle,
    ):
        reference = csv.DictReader(reference_handle)
        for row in csv.DictReader(actual_handle):
            expected = next(reference, None)
            while expected is not None and int(expected["target_index"]) < plan.target_start:
                expected = next(reference, None)
            if expected is None or expected["sample_id"] != row["sample_id"]:
                raise DataContractError("Zero control identities differ from B01")
            if (
                expected["source_valid"] != row["source_valid"]
                or expected["target"] != row["target"]
            ):
                raise DataContractError("Zero control source truth differs from B01")
            for name in MODEL_NAMES:
                difference = abs(float(row[name]) - float(expected[name]))
                maxima[name] = max(maxima[name], difference)
                if difference > tolerance:
                    raise DataContractError(
                        f"Zero control {name} differs from frozen B01: {difference}"
                    )
            compared += 1
    if compared != plan.expected_count:
        raise DataContractError("Zero control did not compare every planned sample")
    result = {
        "status": "PASS",
        "compared": compared,
        "tolerance": tolerance,
        "max_absolute_prediction_difference": maxima,
    }
    write_json(case_directory / "baseline_parity.json", result)
    return result
