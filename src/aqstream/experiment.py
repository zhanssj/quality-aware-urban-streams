from __future__ import annotations

import csv
import importlib.metadata
import os
import platform
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import joblib
import numpy as np
import psutil
from threadpoolctl import threadpool_info, threadpool_limits

from aqstream.artifacts import file_sha256, write_json
from aqstream.baseline_config import load_baseline_config
from aqstream.cli import REPOSITORY, _git, _recorded_input
from aqstream.data import DataContractError, HDF5Reader
from aqstream.evaluation import MODEL_NAMES, evaluate_predictions
from aqstream.features import COMMON_NAMES, QUALITY_NAMES, YearGrid, fit_fallback, scheduled_samples
from aqstream.models import Baselines


class MemoryMonitor:
    """Sample RSS every 50 ms; enforce the guard at phase/batch checkpoints."""

    def __init__(self, max_mb: int):
        self.limit = max_mb * 1024**2
        self.peak = 0
        self._stop = Event()
        self._thread = Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self._stop.is_set():
            self.peak = max(self.peak, process.memory_info().rss)
            self._stop.wait(0.05)

    def __enter__(self):
        self.peak = psutil.Process().memory_info().rss
        self.check()
        self._thread.start()
        return self

    def check(self):
        self.peak = max(self.peak, psutil.Process().memory_info().rss)
        if self.peak > self.limit:
            raise DataContractError(f"RSS guard exceeded: {self.peak} > {self.limit} bytes")

    def __exit__(self, *args):
        self._stop.set()
        self._thread.join()


def choose_cohort(path: Path, count: int, seed: int) -> tuple[str, ...]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        ids = [row["station_id"] for row in csv.DictReader(handle)]
    if len(set(ids)) != len(ids) or any(not value.strip() for value in ids) or len(ids) < count:
        raise DataContractError("Candidate cohort requires sufficient unique nonempty station IDs")
    selected = np.random.Generator(np.random.PCG64(seed)).choice(
        sorted(ids), size=count, replace=False
    )
    return tuple(sorted(selected.tolist()))


def run_experiment(config_path: Path) -> Path:
    config = load_baseline_config(config_path)
    reader = HDF5Reader()
    ids = choose_cohort(config.cohort_candidates, config.cohort_size, config.seed)
    grid = YearGrid.from_reader(reader, (*config.train_years, config.validation_year))
    train_steps = sum(grid.lengths[:-1])
    scheduled_train = len(range(0, train_steps, config.train_stride)) * len(ids)
    if scheduled_train > config.max_train_samples:
        raise DataContractError("Scheduled training set exceeds predeclared max_train_samples")
    started = datetime.now(UTC)
    run_id = f"B01-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    output = Path(os.environ.get("AQ_ARTIFACT_ROOT", "artifacts")) / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_bytes(config_path.read_bytes())
    cohort = {
        "station_ids": ids,
        "seed": config.seed,
        "candidate_sha256": file_sha256(config.cohort_candidates),
        "algorithm": (
            "PCG64(seed).choice(sorted string IDs, without replacement); sort selected IDs"
        ),
    }
    write_json(output / "cohort.json", cohort)
    plan = {
        "config": {**asdict(config), "cohort_candidates": str(config.cohort_candidates)},
        "cohort_sha256": file_sha256(output / "cohort.json"),
        "grid_years": grid.years,
        "grid_lengths": grid.lengths,
        "scheduled_train_before_truth_filter": scheduled_train,
        "scheduled_validation": grid.lengths[-1] * len(ids),
        "training_selection": "target_index % train_stride == 0, all cohort stations",
        "rss_guard": "50ms sampled RSS, checked at phase/batch boundaries; not an OS hard limit",
    }
    write_json(output / "plan.json", plan)  # Saved before reading labels or comparing models.
    wall_started = perf_counter()
    dirty = _git("status", "--porcelain")
    manifest = {
        "run_id": run_id,
        "task_id": "B01",
        "status": "RUNNING",
        "code_commit": _git("rev-parse", "HEAD"),
        "dirty_tree": None if dirty is None else bool(dirty),
        "started_at": started.isoformat(),
        "config_sha256": file_sha256(config_path),
        "plan_sha256": file_sha256(output / "plan.json"),
        "cohort_sha256": file_sha256(output / "cohort.json"),
        "test_gate": "CLOSED",
        "units_verified": False,
        "time_semantics": "provisional_interval_end; row indices",
        "command_argv": sys.argv,
    }
    write_json(output / "manifest.json", manifest)
    try:
        with MemoryMonitor(config.max_rss_mb) as memory, threadpool_limits(limits=config.threads):
            fallback = fit_fallback(
                reader, ids, years=config.train_years, chunk_size=config.chunk_size
            )
            write_json(output / "fallback.json", asdict(fallback))
            common, quality, labels = [], [], []
            pending_features, pending_truth = [], []
            feature_seconds = 0.0
            predict_seconds = dict.fromkeys(MODEL_NAMES, 0.0)
            models = Baselines(seed=config.seed, model_params=config.model_params)
            train_scheduled_actual = train_source_valid = 0
            fields = [
                "run_id",
                "sample_id",
                "station_id",
                "origin_index",
                "target_index",
                "target_year",
                "target_local_index",
                "split",
                "cutoff_index",
                "source_valid",
                "target",
                *MODEL_NAMES,
                "has_history",
                "persistence_fallback",
                "scenario_id",
                "wait_minutes",
            ]
            with (
                (output / "training_sample_ids.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as train_file,
                (output / "predictions_validation.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as pred_file,
            ):
                train_writer = csv.writer(train_file)
                train_writer.writerow(
                    [
                        "sample_id",
                        "station_id",
                        "origin_index",
                        "target_index",
                        "label_source_valid",
                    ]
                )
                writer = csv.DictWriter(pred_file, fieldnames=fields)
                writer.writeheader()

                def predict_pending():
                    if not pending_features:
                        return
                    inputs_common = np.concatenate([frame.common for frame in pending_features])
                    inputs_quality = np.concatenate([frame.quality for frame in pending_features])
                    persistence = np.concatenate([frame.persistence for frame in pending_features])
                    predictions, elapsed = models.predict(
                        inputs_common, inputs_quality, persistence
                    )
                    for name in MODEL_NAMES:
                        predict_seconds[name] += elapsed[name]
                    index = 0
                    for frame, truth in zip(pending_features, pending_truth, strict=True):
                        for column, identity in enumerate(frame.sample_ids):
                            has_history = int(frame.quality[column, -1])
                            good = bool(truth.source_valid[column])
                            row = {
                                "run_id": run_id,
                                "sample_id": identity.key,
                                **asdict(identity),
                                "split": identity.split,
                                "cutoff_index": identity.target_index,
                                "source_valid": int(good),
                                "target": float(truth.values[column]) if good else "",
                                "has_history": has_history,
                                "persistence_fallback": 1 - has_history,
                                "scenario_id": "natural_missingness",
                                "wait_minutes": 0,
                            }
                            row.update(
                                {name: float(predictions[name][index]) for name in MODEL_NAMES}
                            )
                            writer.writerow(row)
                            index += 1
                    pending_features.clear()
                    pending_truth.clear()
                    memory.check()

                for features, truth in scheduled_samples(
                    reader, fallback, grid, chunk_size=config.chunk_size
                ):
                    feature_seconds += features.compute_seconds
                    identity = features.sample_ids[0]
                    if identity.split == "train":
                        if identity.target_index % config.train_stride == 0:
                            train_scheduled_actual += len(ids)
                            for column, sample in enumerate(features.sample_ids):
                                train_writer.writerow(
                                    [
                                        sample.key,
                                        sample.station_id,
                                        sample.origin_index,
                                        sample.target_index,
                                        int(truth.source_valid[column]),
                                    ]
                                )
                            valid = truth.source_valid
                            common.append(features.common[valid])
                            quality.append(features.quality[valid])
                            labels.append(truth.values[valid])
                            train_source_valid += int(valid.sum())
                        # Fit before the generator resumes and opens validation numeric blocks.
                        if identity.target_index == train_steps - 1:
                            train_file.flush()
                            if train_scheduled_actual != scheduled_train:
                                raise DataContractError(
                                    "Actual scheduled training count differs from plan"
                                )
                            x, q, y = (
                                np.concatenate(common),
                                np.concatenate(quality),
                                np.concatenate(labels),
                            )
                            np.savez(output / "train_inputs.npz", common=x, quality=q, labels=y)
                            write_json(
                                output / "fit_plan.json",
                                {
                                    "scheduled_train": train_scheduled_actual,
                                    "source_valid_fit_labels": len(y),
                                    "common_shape": x.shape,
                                    "quality_shape": q.shape,
                                    "training_sample_ids_sha256": file_sha256(
                                        output / "training_sample_ids.csv"
                                    ),
                                    "train_inputs_sha256": file_sha256(output / "train_inputs.npz"),
                                    "model_params": models.standard.get_params(),
                                },
                            )
                            memory.check()
                            print(
                                f"B01 fit: {len(y)} train labels; fixed cohort {len(ids)} stations",
                                flush=True,
                            )
                            models.fit(x, q, y)
                            joblib.dump(models, output / "models.joblib")
                            common.clear()
                            quality.clear()
                            labels.clear()
                            del x, q, y
                            memory.check()
                    else:
                        pending_features.append(features)
                        pending_truth.append(truth)
                        if len(pending_features) >= config.predict_batch_steps:
                            predict_pending()
                predict_pending()
            metrics = evaluate_predictions(output / "predictions_validation.csv")
            write_json(output / "metrics.json", metrics)
            manifest.update(
                status="PASS",
                exit_code=0,
                finished_at=datetime.now(UTC).isoformat(),
                split_definition={
                    "train": config.train_years,
                    "validation": [2024],
                    "test": [2025],
                },
                inputs=[_recorded_input(reader.path_for(year)) for year in grid.years],
                feature_spec={
                    "common": COMMON_NAMES,
                    "quality": QUALITY_NAMES,
                    "rolling": "causally filled, ddof=0; fallback padding; age=-1 without history",
                },
                train_scheduled=scheduled_train,
                train_scored_for_fit=train_source_valid,
                model_params=models.standard.get_params(),
                seed=config.seed,
                dependency_lock_sha256=file_sha256(REPOSITORY / "requirements.lock.txt"),
                python_version=platform.python_version(),
                dependencies={
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
                hardware={
                    "platform": platform.platform(),
                    "cpu": platform.processor(),
                    "logical_cpus": os.cpu_count(),
                    "ram_bytes": psutil.virtual_memory().total,
                },
                threadpools=threadpool_info(),
                peak_rss_sampled_bytes=memory.peak,
                timings={
                    "job_wall_seconds": perf_counter() - wall_started,
                    "feature_compute_seconds": feature_seconds,
                    "fit_seconds": models.fit_seconds,
                    "inference_seconds": predict_seconds,
                    "simulated_wait_minutes": 0,
                },
                artifacts={
                    path.name: {"sha256": file_sha256(path), "bytes": path.stat().st_size}
                    for path in output.iterdir()
                    if path.is_file() and path.name != "manifest.json"
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
