import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from aqstream.artifacts import file_sha256, write_json
from aqstream.data import DataContractError
from aqstream.experiment import MemoryMonitor
from aqstream.replay.baseline import load_frozen_baseline
from aqstream.replay.calendars import make_calendar
from aqstream.replay.config import SCENARIOS
from aqstream.validation.config import load_config
from aqstream.validation.matrix import case_predictions


def benchmark_workload(output, baseline, suites, values, valid, config):
    output.mkdir(parents=True, exist_ok=False)
    start = baseline.validation_start
    stop = start + config["benchmark_steps"]
    measured, warmups = [], []
    with MemoryMonitor(config["max_rss_mb"]) as memory, threadpool_limits(limits=config["threads"]):
        for repeat in range(config["benchmark_warmup"] + config["benchmark_repeats"]):
            began = perf_counter()
            calendar = make_calendar(
                valid[start - 1 : stop - 1],
                baseline.station_ids,
                input_start=start - 1,
                spec=SCENARIOS["delay30"],
                seed=config["validation_seeds"][0],
            )
            predictions, timing = case_predictions(
                baseline,
                suites,
                values,
                valid,
                calendar,
                start=start,
                stop=stop,
                waits=(0,),
                batch_steps=config["batch_steps"],
                memory=memory,
            )
            serial_start = perf_counter()
            np.savez_compressed(output / f"repeat_{repeat}.npz", predictions=predictions[0])
            serialization = perf_counter() - serial_start
            elapsed = perf_counter() - began
            record = {
                "repeat": repeat,
                "job_seconds": elapsed,
                "serialization_seconds": serialization,
                "samples": (stop - start) * len(baseline.station_ids),
                "samples_per_second": (stop - start) * len(baseline.station_ids) / elapsed,
                **timing[0],
            }
            (warmups if repeat < config["benchmark_warmup"] else measured).append(record)
    summaries = {}
    paths = {
        "job_seconds": [r["job_seconds"] for r in measured],
        "feature_seconds": [r["feature_seconds"] for r in measured],
        "receive_seconds": [r["receive_seconds"] for r in measured],
        "serialization_seconds": [r["serialization_seconds"] for r in measured],
        "samples_per_second": [r["samples_per_second"] for r in measured],
    }
    for name in measured[0]["models"]:
        for timer in ("transform_seconds", "inference_seconds"):
            paths[f"{name}/{timer}"] = [r["models"][name][timer] for r in measured]
    for name, samples in paths.items():
        summaries[name] = {
            "p50": float(np.quantile(samples, 0.5)),
            "p95": float(np.quantile(samples, 0.95)),
        }
    result = {
        "status": "PASS",
        "warmups": warmups,
        "measured": measured,
        "summary": summaries,
        "peak_rss_sampled_bytes": memory.peak,
        "threads": config["threads"],
        "batch_steps": config["batch_steps"],
        "scope": (
            "calendar/observer/delivery/features/all 12 regime-method outputs/local NPZ write; "
            "excludes source/model loading, fit, scoring"
        ),
        "timed_simulated_wait_seconds": 0,
        "policy_wait_minutes": 0,
        "delay": "30 percent uniform 1..30 simulated minutes",
        "observations_per_repeat": measured[0]["samples"],
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": psutil.cpu_count(),
            "ram_bytes": psutil.virtual_memory().total,
        },
        "python": sys.version,
    }
    write_json(output / "benchmark.json", result)
    return result


def run_benchmark(run: Path, expected_manifest_sha256: str, output: Path):
    started = perf_counter()
    if file_sha256(run / "manifest.json") != expected_manifest_sha256:
        raise DataContractError("Benchmark requires pinned trusted T06 manifest")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS"
        or manifest.get("task_id") != "T06"
        or manifest.get("test_gate") != "CLOSED"
    ):
        raise DataContractError("Benchmark requires successful closed-gate T06 run")
    for name in ("models.joblib", "config.yaml"):
        if file_sha256(run / name) != manifest["artifacts"][name]["sha256"]:
            raise DataContractError("Changed trusted benchmark input")
    config = load_config(run / "config.yaml")
    baseline = load_frozen_baseline(
        Path(config["baseline_run"]), config["baseline_manifest_sha256"]
    )
    suites = joblib.load(run / "models.joblib")
    from aqstream.validation.runner import read_source

    values, valid, references = read_source(baseline, config["benchmark_steps"])
    result = benchmark_workload(output, baseline, suites, values, valid, config)
    result.update(
        cli_wall_seconds=perf_counter() - started,
        source_reads=references,
        manifest_sha256=expected_manifest_sha256,
        model_sha256=file_sha256(run / "models.joblib"),
    )
    write_json(output / "benchmark.json", result)
    return output
