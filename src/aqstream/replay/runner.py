"""Frozen B01 controlled validation replay, no fitting and no wall-clock waiting."""

import importlib.metadata
import json
import os
import platform
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from aqstream.artifacts import file_sha256, write_json
from aqstream.cli import REPOSITORY, _git
from aqstream.data import DataContractError, HDF5Reader
from aqstream.evaluation import evaluate_predictions
from aqstream.experiment import MemoryMonitor
from aqstream.replay.baseline import cache_source, load_frozen_baseline
from aqstream.replay.calendars import ScenarioSpec, make_calendar
from aqstream.replay.cases import compare_with_baseline, replay_calendar
from aqstream.replay.config import SCENARIOS, load_replay_config


def run_replay(config_path: Path) -> Path:
    wall_started = perf_counter()
    started = datetime.now(UTC)
    config = load_replay_config(config_path)
    baseline = load_frozen_baseline(config.baseline_run, config.baseline_manifest_sha256)
    if config.validation_steps > baseline.grid.lengths[-1]:
        raise DataContractError("Pilot validation_steps exceeds baseline grid")
    run_id = f"B02-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    output = Path(os.environ.get("AQ_ARTIFACT_ROOT", "artifacts")) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_bytes(config_path.read_bytes())
    (output / "cases").mkdir()
    (output / "calendars").mkdir()
    specs = [
        (SCENARIOS[name], seed)
        for name in config.scenarios
        for seed in ((0,) if name == "clean_source" else config.validation_seeds)
    ]
    expanded = [
        {
            "spec": asdict(spec),
            "seed": seed,
            "waits": config.waits if spec.delay_fraction or spec.name == "clean_source" else (0,),
        }
        for spec, seed in specs
    ]
    write_json(
        output / "plan.json",
        {
            "config": {**asdict(config), "baseline_run": str(config.baseline_run)},
            "station_ids": baseline.station_ids,
            "target_start": baseline.validation_start,
            "target_stop": baseline.validation_start + config.validation_steps,
            "case_schedules": expanded,
            "pilot_policy_cases": sum(len(item["waits"]) for item in expanded),
            "training_mode": "frozen_clean_B01; augmented_training_NOT_RUN",
            "training_mixture_api": "weighted views implemented and tested; not fitted in B02",
            "time_semantics": "provisional interval_end, step=60 simulated minutes",
        },
    )
    dirty = _git("status", "--porcelain")
    manifest = {
        "task_id": "B02",
        "run_id": run_id,
        "status": "RUNNING",
        "code_commit": _git("rev-parse", "HEAD"),
        "dirty_tree": None if dirty is None else bool(dirty),
        "started_at": started.isoformat(),
        "baseline_manifest_sha256": config.baseline_manifest_sha256,
        "baseline_run_id": baseline.manifest["run_id"],
        "baseline_code_commit": baseline.manifest["code_commit"],
        "frozen_models_sha256": baseline.manifest["artifacts"]["models.joblib"]["sha256"],
        "frozen_cohort_sha256": baseline.manifest["cohort_sha256"],
        "config_sha256": file_sha256(config_path),
        "plan_sha256": file_sha256(output / "plan.json"),
        "dependency_lock_sha256": file_sha256(REPOSITORY / "requirements.lock.txt"),
        "command_argv": sys.argv,
        "test_gate": "CLOSED",
        "units_verified": False,
        "training_mode": "frozen_clean_B01",
        "augmented_model_fit": "NOT_RUN",
        "python_version": sys.version,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "h5py",
                "scikit-learn",
                "scipy",
                "joblib",
                "threadpoolctl",
                "psutil",
            )
        },
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
        },
        "time_semantics": "provisional interval_end; 60 simulated minutes per index step",
    }
    write_json(output / "manifest.json", manifest)
    cases, controls = [], []
    try:
        with MemoryMonitor(config.max_rss_mb) as memory, threadpool_limits(limits=config.threads):
            baseline_plan = baseline.evaluation_plan()
            baseline_plan.save(output / "baseline_evaluation_plan.json")
            rechecked = evaluate_predictions(
                baseline.directory / "predictions_validation.csv", plan=baseline_plan
            )
            original_metrics = json.loads(
                (baseline.directory / "metrics.json").read_text(encoding="utf-8")
            )
            if rechecked["models"] != original_metrics["models"]:
                raise DataContractError("N1 re-evaluation changed original B01 numerical metrics")
            write_json(output / "baseline_verified_metrics.json", rechecked)
            reader = HDF5Reader()
            read_steps = (
                baseline.grid.lengths[-1] if config.full_zero_control else config.validation_steps
            )
            values, source_valid, references = cache_source(
                reader, baseline, validation_steps=read_steps, chunk_size=config.chunk_size
            )
            memory.check()
            start = baseline.validation_start

            def execute(spec, seed, stop, waits):
                calendar = make_calendar(
                    source_valid[start - 1 : stop - 1],
                    baseline.station_ids,
                    input_start=start - 1,
                    spec=spec,
                    seed=seed,
                )
                calendar.save(output / "calendars" / calendar.scenario_id)
                results = replay_calendar(
                    output / "cases",
                    baseline=baseline,
                    values=values,
                    source_valid=source_valid,
                    calendar=calendar,
                    run_id=run_id,
                    target_start=start,
                    target_stop=stop,
                    waits=waits,
                    batch_steps=config.batch_steps,
                    memory=memory,
                )
                if spec.random_drop_rate == spec.block_duration == spec.delay_fraction == 0:
                    for case in results:
                        case["baseline_parity"] = compare_with_baseline(
                            output / "cases" / case["case_id"],
                            baseline.directory / "predictions_validation.csv",
                        )
                print(
                    f"B02 completed {calendar.scenario_id}: waits={list(waits)}, "
                    f"planned={results[0]['metrics']['planned']}",
                    flush=True,
                )
                return results

            if config.full_zero_control:
                controls = execute(
                    ScenarioSpec("zero_control_full"), 0, sum(baseline.grid.lengths), (0,)
                )
            for spec, seed in specs:
                waits = config.waits if spec.delay_fraction or spec.name == "clean_source" else (0,)
                cases.extend(execute(spec, seed, start + config.validation_steps, waits))
            shared_ids = {
                metric["scored_target_ids_sha256"]
                for case in cases
                for metric in case["metrics"]["models"].values()
            }
            if len(shared_ids) != 1:
                raise DataContractError("Scenario methods/policies do not share scored targets")
            write_json(
                output / "summary_metrics.json",
                {"pilot_cases": cases, "full_zero_controls": controls},
            )
            memory.check()
            manifest.update(
                status="PASS",
                exit_code=0,
                finished_at=datetime.now(UTC).isoformat(),
                pilot_cases=len(cases),
                full_zero_control_cases=len(controls),
                shared_scored_target_ids_sha256=next(iter(shared_ids)),
                cached_source_shape=values.shape,
                source_reads=references,
                historical_dataset_inputs=baseline.manifest["inputs"],
                peak_rss_sampled_bytes=memory.peak,
                threadpools=threadpool_info(),
                timings={
                    "controlled_workload_wall_seconds": perf_counter() - wall_started,
                    "wall_scope": (
                        "includes baseline integrity/evaluation, source read and replay; "
                        "excludes final artifact hashing/manifest write"
                    ),
                },
                artifacts={
                    path.relative_to(output).as_posix(): {
                        "sha256": file_sha256(path),
                        "bytes": path.stat().st_size,
                    }
                    for path in output.rglob("*")
                    if path.is_file() and path != output / "manifest.json"
                },
            )
    except Exception as exc:
        manifest.update(
            status="FAIL",
            exit_code=2,
            error=f"{type(exc).__name__}: {exc}",
            finished_at=datetime.now(UTC).isoformat(),
        )
        write_json(output / "manifest.json", manifest)
        raise
    write_json(output / "manifest.json", manifest)
    return output
