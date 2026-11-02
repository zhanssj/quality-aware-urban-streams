from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from aqstream.config import load_prepare_config
from aqstream.data import DataContractError, HDF5Reader

REPOSITORY = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPOSITORY), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _recorded_input(path: Path) -> dict:
    """Reuse previous provenance without claiming to rehash a full HDF5."""
    result = {
        "file": path.name,
        "size_bytes_observed": path.stat().st_size,
        "mtime_ns_observed": path.stat().st_mtime_ns,
        "recorded_sha256": None,
        "sha256_verified_in_this_run": False,
        "dataset_revision": None,
        "provenance": None,
    }
    manifest = path.parent / "download_manifest.json"
    if manifest.is_file():
        try:
            content = json.loads(manifest.read_text(encoding="utf-8"))
            record = next(item for item in content["files"] if item["name"] == path.name)
            result.update(
                recorded_sha256=record.get("sha256"),
                dataset_revision=content.get("revision"),
                provenance="existing download_manifest.json; not reverified",
                provenance_manifest_sha256=_sha256(manifest),
                recorded_size_matches=record.get("size_bytes") == path.stat().st_size,
            )
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            raise DataContractError(f"Invalid provenance manifest {manifest}: {exc}") from exc
    return result


def prepare(config_path: Path) -> Path:
    """Bounded loader smoke only; does not save raw inputs or prepare features."""
    started = datetime.now(UTC)
    config = load_prepare_config(config_path)
    reader = HDF5Reader()
    info = reader.inspect(config.year)
    counts = np.zeros(len(config.station_ids), dtype=np.int64)
    rows = blocks = available = 0
    selection_digest = hashlib.sha256()
    reading_started = perf_counter()
    for block in reader.iter_blocks(
        year=config.year,
        station_ids=config.station_ids,
        start=config.start,
        stop=config.stop,
        chunk_size=config.chunk_size,
        parameter=config.parameter,
    ):
        rows += block.stop - block.start
        blocks += 1
        available += int(block.source_valid.sum())
        counts += block.source_valid.sum(axis=0)
        # A content fingerprint of this small sanitized selection, not the source file.
        selection_digest.update(block.values.astype("<f8").tobytes())
        selection_digest.update(block.source_valid.tobytes())
    read_seconds = perf_counter() - reading_started
    input_record = _recorded_input(info.path)
    run_id = f"T01-{started.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    dirty = _git("status", "--porcelain")
    manifest = {
        "run_id": run_id,
        "task_id": "T01",
        "status": "PASS",
        "purpose": "bounded_loader_smoke_only",
        "code_commit": _git("rev-parse", "HEAD"),
        "dirty_tree": None if dirty is None else bool(dirty),
        "config_sha256": _sha256(config_path),
        "dependency_lock_sha256": _sha256(REPOSITORY / "requirements.lock.txt"),
        "input": input_record,
        "selection": {
            "year": config.year,
            "station_ids": config.station_ids,
            "parameter_requested": config.parameter,
            "start": config.start,
            "stop": config.stop,
            "chunk_size": config.chunk_size,
            "sanitized_blocks_sha256": selection_digest.hexdigest(),
            "hash_encoding": "per block: row-major little-endian float64 then bool bytes",
        },
        "source_tensor_shape": info.shape,
        "output_shape": [rows, len(config.station_ids)],
        "blocks_read": blocks,
        "available_elements": available,
        "unavailable_elements": rows * len(config.station_ids) - available,
        "available_by_station": dict(zip(config.station_ids, counts.tolist(), strict=True)),
        "split_definition": {"train": [2021, 2022, 2023], "validation": [2024], "test": [2025]},
        "test_gate": "CLOSED",
        "units_verified": False,
        "time_semantics": "row_index_only; provisional_interval_end for future replay",
        "model_training": "NOT_RUN",
        "predictions": "NOT_RUN",
        "timing": {
            "block_read_wall_seconds": read_seconds,
            "feature_computation_ms": None,
            "inference_ms": None,
            "simulated_wait_minutes": None,
        },
        "python_version": sys.version,
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("numpy", "h5py", "PyYAML")
        },
        "hardware": {
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
        },
        "command_argv": sys.argv,
        "exit_code": 0,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
    }
    output = config.artifact_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / "config.yaml").write_bytes(config_path.read_bytes())
    destination = output / "manifest.json"
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quality-aware sensor stream research core (B01)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("prepare", help="read at most 168 train rows and 2–4 stations")
    smoke.add_argument("--config", required=True, type=Path)
    train = subparsers.add_parser("train", help="fixed bounded B01 train/validation experiment")
    train.add_argument("--config", required=True, type=Path)
    evaluate = subparsers.add_parser("evaluate", help="recompute shared-target validation metrics")
    evaluate.add_argument("--predictions", required=True, type=Path)
    evaluate.add_argument("--out", required=True, type=Path)
    evaluate.add_argument("--plan", type=Path, help="expected plan; default alongside predictions")
    evaluate.add_argument("--split", choices=["validation"], default="validation")
    export = subparsers.add_parser(
        "export-review", help="export whitelisted B01 source/docs/evidence"
    )
    export.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare(args.config)
            print(f"PASS bounded loader smoke; manifest: {manifest}")
        elif args.command == "train":
            from aqstream.experiment import run_experiment

            output = run_experiment(args.config)
            print(f"PASS B01 train/validation; artifacts: {output}")
        elif args.command == "evaluate":
            from aqstream.artifacts import write_json
            from aqstream.evaluation import evaluate_predictions
            from aqstream.prediction_plan import PredictionPlan

            plan = PredictionPlan.load(args.plan) if args.plan else None
            metrics = evaluate_predictions(args.predictions, plan=plan)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.out, metrics)
            print(f"PASS diagnostic validation metrics: {args.out}")
        elif args.command == "export-review":
            from aqstream.artifacts import export_review

            print(f"PASS review bundle: {export_review(REPOSITORY, args.out)}")
    except (DataContractError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0
