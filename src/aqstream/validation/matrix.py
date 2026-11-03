from pathlib import Path

import numpy as np

from aqstream.artifacts import write_json
from aqstream.data import DataContractError
from aqstream.prediction_plan import PredictionPlan
from aqstream.replay.stream import replay_snapshots
from aqstream.validation.metrics import evaluate_npz, save_predictions, slice_labels
from aqstream.validation.models import MODEL_NAMES, predict


def bootstrap_options(config):
    return {
        "block_steps": config["bootstrap_block_steps"],
        "replicates": config["bootstrap_replicates"],
        "seed": config["bootstrap_seed"],
    }


def case_predictions(
    baseline, suites, values, valid, calendar, *, start, stop, waits, batch_steps, memory
):
    arrays = {
        wait: np.full((stop - start, len(baseline.station_ids), len(MODEL_NAMES)), np.nan)
        for wait in waits
    }
    pending = {wait: [] for wait in waits}
    counts = dict.fromkeys(waits, 0)
    timing = {
        wait: {
            "feature_seconds": 0.0,
            "receive_seconds": 0.0,
            "models": {
                name: {"transform_seconds": 0.0, "inference_seconds": 0.0} for name in MODEL_NAMES
            },
        }
        for wait in waits
    }

    def flush(wait):
        batch = pending[wait]
        if not batch:
            return
        predictions, timers = predict(
            suites,
            np.concatenate([s.features.common for s in batch]),
            np.concatenate([s.features.quality for s in batch]),
            np.concatenate([s.features.persistence for s in batch]),
        )
        left = batch[0].features.sample_ids[0].target_index - start
        arrays[wait][left : left + len(batch)] = predictions.reshape(
            len(batch), len(baseline.station_ids), len(MODEL_NAMES)
        )
        for name, parts in timers.items():
            for part, seconds in parts.items():
                timing[wait]["models"][name][part] += seconds
        batch.clear()
        memory.check()

    for snapshot in replay_snapshots(
        values,
        valid,
        baseline.fitted,
        baseline.grid,
        calendar,
        target_start=start,
        target_stop=stop,
        waits=tuple(waits),
    ):
        wait = snapshot.wait_minutes
        expected_target = start + counts[wait]
        year, local = baseline.grid.point(expected_target)
        actual = tuple(
            (s.station_id, s.target_index, s.origin_index, s.target_year, s.target_local_index)
            for s in snapshot.features.sample_ids
        )
        expected = tuple(
            (station, expected_target, expected_target - 1, year, local)
            for station in baseline.station_ids
        )
        if actual != expected:
            raise DataContractError("Forecast identity differs from predeclared tensor position")
        counts[wait] += 1
        timing[wait]["feature_seconds"] += snapshot.features.compute_seconds
        timing[wait]["receive_seconds"] += snapshot.receive_seconds
        pending[wait].append(snapshot)
        if len(pending[wait]) >= batch_steps:
            flush(wait)
    for wait in waits:
        flush(wait)
        if counts[wait] != stop - start or not np.isfinite(arrays[wait]).all():
            raise DataContractError("Missing scheduled tensor forecasts")
    return arrays, timing


def execute_calendar(
    output: Path,
    baseline,
    suites,
    values,
    valid,
    calendar,
    config,
    *,
    run_id,
    start,
    stop,
    waits,
    memory,
):
    # Every policy gets an independent expected plan before any predictions are generated.
    plans, directories = {}, {}
    for wait in waits:
        directory = output / f"{calendar.scenario_id}_wait{wait}"
        directory.mkdir(parents=True, exist_ok=False)
        plan = PredictionPlan(
            baseline.station_ids,
            start,
            stop,
            baseline.grid.point(start)[1],
            run_id,
            calendar.scenario_id,
            wait,
        )
        plan.save(directory / "evaluation_plan.json")
        plans[wait], directories[wait] = plan, directory
    predictions, timings = case_predictions(
        baseline,
        suites,
        values,
        valid,
        calendar,
        start=start,
        stop=stop,
        waits=waits,
        batch_steps=config["batch_steps"],
        memory=memory,
    )
    cases = []
    for wait in waits:
        directory, plan = directories[wait], plans[wait]
        labels = slice_labels(calendar, wait, config["recovery_steps"])
        save_predictions(directory / "predictions.npz", predictions[wait], plan, labels)
        metrics = evaluate_npz(
            directory / "predictions.npz",
            output.parent / "truth.npz",
            bootstrap=bootstrap_options(config),
        )
        write_json(directory / "metrics.json", metrics)
        case = {
            "case_id": directory.name,
            "scenario_name": calendar.spec.name,
            "seed": calendar.seed,
            "wait_minutes": wait,
            "remaining_lead_minutes": 60 - wait,
            "calendar_sha256": calendar.fingerprint,
            "statistics": calendar.statistics,
            "timings": timings[wait],
            "metrics": metrics,
        }
        write_json(directory / "case.json", case)
        cases.append(case)
    return cases
