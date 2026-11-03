import csv
import importlib.metadata
import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from aqstream.artifacts import file_sha256, write_json
from aqstream.cli import REPOSITORY, _git
from aqstream.data import DataContractError, HDF5Reader
from aqstream.experiment import MemoryMonitor
from aqstream.features import COMMON_NAMES, QUALITY_NAMES
from aqstream.replay.baseline import load_frozen_baseline
from aqstream.replay.calendars import make_calendar
from aqstream.replay.config import SCENARIOS
from aqstream.validation.benchmark import benchmark_workload
from aqstream.validation.config import load_config
from aqstream.validation.matrix import execute_calendar
from aqstream.validation.metrics import select_regime
from aqstream.validation.models import MODEL_NAMES, fit_suites


def read_source(baseline, validation_steps):
    if type(validation_steps) is not int or not 0 <= validation_steps <= baseline.grid.lengths[-1]:
        raise DataContractError("Read only bounded train/validation, never test")
    reader = HDF5Reader()
    stop = baseline.validation_start + validation_steps
    values = np.full((stop, len(baseline.station_ids)), np.nan)
    valid = np.zeros_like(values, dtype=bool)
    offset, references = 0, []
    for year, length in zip(baseline.grid.years, baseline.grid.lengths, strict=True):
        if offset >= stop:
            break
        info = reader.inspect(year)
        if info.shape[0] != length:
            raise DataContractError("Source year shape differs from frozen grid")
        rows = min(length, stop - offset)
        for block in reader.iter_blocks(
            year=year, station_ids=baseline.station_ids, start=0, stop=rows, chunk_size=168
        ):
            values[offset + block.start : offset + block.stop] = block.values
            valid[offset + block.start : offset + block.stop] = block.source_valid
        references.append(
            {
                "year": year,
                "file": info.path.name,
                "rows_read": rows,
                "size_bytes_observed": info.path.stat().st_size,
            }
        )
        offset += length
    values.setflags(write=False)
    valid.setflags(write=False)
    return values, valid, references


def zero_parity(output, baseline, truth_values, truth_valid):
    with np.load(output / "predictions.npz", allow_pickle=False) as data:
        predictions = data["predictions"]
    reference = np.empty((baseline.grid.lengths[-1], len(baseline.station_ids), 3))
    compared = 0
    names = ("persistence", "standard", "quality_aware")
    with (baseline.directory / "predictions_validation.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            target, column = divmod(index, len(baseline.station_ids))
            if (
                row["station_id"] != baseline.station_ids[column]
                or int(row["target_index"]) != target + baseline.validation_start
            ):
                raise DataContractError("Unexpected B01 identity in zero-control reference")
            reference[target, column] = [float(row[name]) for name in names]
            if target < len(predictions):
                if bool(int(row["source_valid"])) != bool(truth_valid[target, column]) or (
                    truth_valid[target, column]
                    and float(row["target"]) != truth_values[target, column]
                ):
                    raise DataContractError("Zero-control truth differs from B01")
                compared += 1
    if index + 1 != baseline.grid.lengths[-1] * len(baseline.station_ids):
        raise DataContractError("Incomplete original B01 CSV")
    difference = np.abs(predictions[:, :, [0, 1, 4]] - reference[: len(predictions)])
    maxima = difference.max(axis=(0, 1))
    if np.any(maxima > 1e-10):
        raise DataContractError(f"Clean zero-control changed B01 predictions: {maxima}")
    return {
        "status": "PASS",
        "compared_per_method": compared,
        "tolerance": 1e-10,
        "max_absolute_difference": dict(zip(names, maxima.tolist(), strict=True)),
    }


def run_validation(config_path: Path):
    started, wall = datetime.now(UTC), perf_counter()
    config = load_config(config_path)
    baseline = load_frozen_baseline(
        Path(config["baseline_run"]), config["baseline_manifest_sha256"]
    )
    original = json.loads((baseline.directory / "plan.json").read_text(encoding="utf-8"))["config"]
    if (
        config["train_stride"] != original["train_stride"]
        or config["model_seed"] != original["seed"]
    ):
        raise DataContractError("Retain original B01 stride and model seed")
    if config["validation_steps"] > baseline.grid.lengths[-1]:
        raise DataContractError("Validation exceeds frozen grid")
    run_id = f"T06-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    output = Path(os.environ.get("AQ_ARTIFACT_ROOT", "artifacts")) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_bytes(config_path.read_bytes())
    write_json(
        output / "plan.json",
        {
            "config": config,
            "station_ids": baseline.station_ids,
            "target_start": baseline.validation_start,
            "target_stop": baseline.validation_start + config["validation_steps"],
            "models": MODEL_NAMES,
            "training_modes": [
                "clean_B01_frozen_plus_same_data_ablations",
                "augmented_weighted_common_mixture",
            ],
            "test_gate": "CLOSED",
        },
    )
    manifest = {
        "task_id": "T06",
        "run_id": run_id,
        "status": "RUNNING",
        "started_at": started.isoformat(),
        "code_commit": _git("rev-parse", "HEAD"),
        "dirty_tree": bool(_git("status", "--porcelain")),
        "config_sha256": file_sha256(config_path),
        "plan_sha256": file_sha256(output / "plan.json"),
        "baseline_manifest_sha256": config["baseline_manifest_sha256"],
        "baseline_run_id": baseline.manifest["run_id"],
        "frozen_cohort_sha256": baseline.manifest["cohort_sha256"],
        "test_gate": "CLOSED",
        "units_verified": False,
        "command_argv": sys.argv,
        "dependency_lock_sha256": file_sha256(REPOSITORY / "requirements.lock.txt"),
        "python": sys.version,
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
        "historical_dataset_inputs": baseline.manifest["inputs"],
    }
    write_json(output / "manifest.json", manifest)
    try:
        with (
            MemoryMonitor(config["max_rss_mb"]) as memory,
            threadpool_limits(limits=config["threads"]),
        ):
            train_values, train_valid, train_reads = read_source(baseline, 0)
            suites, fit_plan = fit_suites(
                baseline, train_values, train_valid, config, output, memory
            )
            joblib.dump(suites, output / "models.joblib")
            (output / "models").mkdir()
            for regime, models in suites.items():
                for name, model in models.items():
                    joblib.dump(model, output / "models" / f"{regime}_{name}.joblib")
            del train_values, train_valid
            # All fits completed before this first numeric validation read.
            values, valid, source_reads = read_source(baseline, config["validation_steps"])
            first = baseline.validation_start
            np.savez_compressed(
                output / "truth.npz",
                values=values[first:],
                valid=valid[first:],
                station_ids=np.array(baseline.station_ids),
                target_indices=np.arange(first, len(values)),
            )
            phases, parity = {}, {}
            for phase, steps, seeds in (
                (
                    "pilot",
                    config["pilot_steps"],
                    config["validation_seeds"][: config["pilot_seed_count"]],
                ),
                ("final", config["validation_steps"], config["validation_seeds"]),
            ):
                folder = output / f"{phase}_cases"
                folder.mkdir()
                cases = []
                for name in config["scenarios"]:
                    for seed in [0] if name == "clean_source" else seeds:
                        stop = first + steps
                        spec = SCENARIOS[name]
                        calendar = make_calendar(
                            valid[first - 1 : stop - 1],
                            baseline.station_ids,
                            input_start=first - 1,
                            spec=spec,
                            seed=seed,
                        )
                        calendar.save(output / "calendars" / phase / calendar.scenario_id)
                        waits = (
                            config["waits"]
                            if name == "clean_source" or spec.delay_fraction
                            else (0,)
                        )
                        results = execute_calendar(
                            folder,
                            baseline,
                            suites,
                            values,
                            valid,
                            calendar,
                            config,
                            run_id=run_id,
                            start=first,
                            stop=stop,
                            waits=waits,
                            memory=memory,
                        )
                        cases.extend(results)
                        print(
                            f"T06 {phase} {calendar.scenario_id}: {len(results)} policies, "
                            f"{results[0]['metrics']['planned']} samples each",
                            flush=True,
                        )
                hashes = {case["metrics"]["scored_target_ids_sha256"] for case in cases}
                if len(hashes) != 1:
                    raise DataContractError(
                        "Methods/regimes/scenarios have different scored targets"
                    )
                parity[phase] = zero_parity(
                    folder / "clean_source_seed0_wait0",
                    baseline,
                    values[first : first + steps],
                    valid[first : first + steps],
                )
                phases[phase] = cases
                write_json(output / f"{phase}_metrics.json", cases)
                memory.check()
            selection = select_regime(phases["final"])
            write_json(output / "selection.json", selection)
            benchmark = benchmark_workload(
                output / "benchmark", baseline, suites, values, valid, config
            )
            freeze = {
                "schema_version": 1,
                "task_id": "T06",
                "scope": "diagnostic file_units protocol; physical claims unresolved",
                "code_commit": manifest["code_commit"],
                "run_id": run_id,
                "test_gate": "CLOSED",
                "owner_test_authorization": None,
                "selection": selection,
                "validation_year": 2024,
                "validation_target_range": [first, len(values)],
                "train_years": baseline.grid.years[:-1],
                "train_stride": config["train_stride"],
                "fit_plan_sha256": file_sha256(output / "fit_plan.json"),
                "test_target_year": 2025,
                "test_interval": (
                    "all source positions; validate published shape in separately authorized T07"
                ),
                "validation_seeds": config["validation_seeds"],
                "test_seeds": [6101, 6102, 6103, 6104, 6105],
                "train_mixture_seeds": config["train_mixture_seeds"],
                "training_mixture_weights": config["training_mixture_weights"],
                "scenarios": config["scenarios"],
                "scenario_specs": {name: asdict(SCENARIOS[name]) for name in config["scenarios"]},
                "waits": config["waits"],
                "remaining_lead_minutes": [60, 55, 45],
                "time_semantics": (
                    "provisional interval_end; 60 simulated minutes per index; "
                    "not verified physical timestamps"
                ),
                "units_verified": False,
                "units": "file_units",
                "common_features": COMMON_NAMES,
                "full_quality_features": QUALITY_NAMES,
                "primary_methods": ["persistence", "standard", "quality_aware"],
                "ablations": ["masks", "age", "native_missing"],
                "model_params": baseline.models.standard.get_params(),
                "model_bank_sha256": file_sha256(output / "models.joblib"),
                "individual_model_sha256": {
                    p.stem: file_sha256(p) for p in sorted((output / "models").glob("*.joblib"))
                },
                "cohort_station_ids": baseline.station_ids,
                "cohort_sha256": baseline.manifest["cohort_sha256"],
                "fallback_sha256": baseline.manifest["artifacts"]["fallback.json"]["sha256"],
                "baseline_manifest_sha256": config["baseline_manifest_sha256"],
                "dependency_lock_sha256": manifest["dependency_lock_sha256"],
                "config_sha256": manifest["config_sha256"],
                "source_sha256": {
                    p.relative_to(REPOSITORY).as_posix(): file_sha256(p)
                    for p in sorted((REPOSITORY / "src/aqstream").rglob("*.py"))
                },
                "bootstrap": {
                    "block_steps": config["bootstrap_block_steps"],
                    "replicates": config["bootstrap_replicates"],
                    "seed": config["bootstrap_seed"],
                    "method": "paired non-overlapping time blocks; stations together; percentile95",
                },
                "recovery_steps": config["recovery_steps"],
                "table_templates": {
                    "accuracy": [
                        "phase",
                        "scenario",
                        "seed",
                        "wait",
                        "regime",
                        "method",
                        "MAE",
                        "RMSE",
                        "macro_MAE",
                        "scheduled",
                        "predicted",
                        "scored",
                    ],
                    "paired": [
                        "scenario",
                        "seed",
                        "wait",
                        "comparison",
                        "delta_MAE",
                        "CI_low",
                        "CI_high",
                    ],
                    "performance": ["timer", "p50", "p95", "throughput", "RSS", "threads", "batch"],
                },
            }
            write_json(output / "protocol_freeze.json", freeze)
            manifest.update(
                status="PASS",
                exit_code=0,
                finished_at=datetime.now(UTC).isoformat(),
                source_reads=source_reads,
                training_source_reads=train_reads,
                fit_summary={
                    "original_fit_labels": fit_plan["original_fit_labels"],
                    "augmented_rows": fit_plan["augmented_rows"],
                    "total_weight": fit_plan["total_weight"],
                },
                pilot_cases=len(phases["pilot"]),
                final_cases=len(phases["final"]),
                zero_control=parity,
                selection=selection,
                final_scored_target_ids_sha256=phases["final"][0]["metrics"][
                    "scored_target_ids_sha256"
                ],
                peak_rss_sampled_bytes=memory.peak,
                benchmark_summary=benchmark["summary"],
                threadpools=threadpool_info(),
                hardware=benchmark["hardware"],
                timings={
                    "workload_wall_seconds": perf_counter() - wall,
                    "scope": (
                        "config/load/train/source/pilot/final/bootstrap/benchmark/freeze; "
                        "excludes final output hashing and manifest write"
                    ),
                },
            )
        manifest["artifacts"] = {
            p.relative_to(output).as_posix(): {"sha256": file_sha256(p), "bytes": p.stat().st_size}
            for p in sorted(output.rglob("*"))
            if p.is_file() and p != output / "manifest.json"
        }
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
