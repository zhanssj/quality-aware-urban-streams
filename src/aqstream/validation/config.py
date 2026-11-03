import math
import re
from pathlib import Path

import yaml

from aqstream.data import DataContractError
from aqstream.replay.config import SCENARIOS

FIELDS = {
    "task_id",
    "baseline_run",
    "baseline_manifest_sha256",
    "validation_year",
    "validation_steps",
    "pilot_steps",
    "pilot_seed_count",
    "validation_seeds",
    "train_mixture_seeds",
    "scenarios",
    "waits",
    "training_mixture_weights",
    "train_stride",
    "model_seed",
    "batch_steps",
    "threads",
    "max_rss_mb",
    "bootstrap_block_steps",
    "bootstrap_replicates",
    "bootstrap_seed",
    "recovery_steps",
    "benchmark_steps",
    "benchmark_warmup",
    "benchmark_repeats",
}


def load_config(path: Path) -> dict:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DataContractError(f"Invalid T06 config: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != FIELDS:
        raise DataContractError("Expected exact T06 config fields")
    if (
        raw["task_id"] != "T06"
        or type(raw["validation_year"]) is not int
        or raw["validation_year"] != 2024
    ):
        raise DataContractError("T06 is validation 2024 only; test gate 2025 is CLOSED")
    for key, low, high in (
        ("validation_steps", 1, 8784),
        ("pilot_steps", 1, 8784),
        ("pilot_seed_count", 3, 3),
        ("train_stride", 1, 1000),
        ("model_seed", 0, 2**32 - 1),
        ("batch_steps", 1, 512),
        ("threads", 1, 8),
        ("max_rss_mb", 256, 4096),
        ("bootstrap_block_steps", 1, 8784),
        ("bootstrap_replicates", 20, 10000),
        ("bootstrap_seed", 1, 2**32 - 1),
        ("recovery_steps", 1, 168),
        ("benchmark_steps", 1, 512),
        ("benchmark_warmup", 1, 10),
        ("benchmark_repeats", 2, 30),
    ):
        if type(raw[key]) is not int or not low <= raw[key] <= high:
            raise DataContractError(f"Invalid T06 bound: {key}")
    if (
        raw["pilot_steps"] > raw["validation_steps"]
        or raw["benchmark_steps"] > raw["validation_steps"]
    ):
        raise DataContractError("Pilot/benchmark must fit inside validation")
    for name in ("validation_seeds", "train_mixture_seeds"):
        items = raw[name]
        if (
            not isinstance(items, list)
            or not items
            or len(items) > 5
            or any(type(x) is not int or not 0 < x < 2**32 for x in items)
            or len(set(items)) != len(items)
        ):
            raise DataContractError(f"Expected unique positive {name}")
    if len(raw["validation_seeds"]) != 5 or set(raw["validation_seeds"]) & set(
        raw["train_mixture_seeds"]
    ):
        raise DataContractError("Five validation seeds independent of train are required")
    scenarios = raw["scenarios"]
    if (
        not isinstance(scenarios, list)
        or not scenarios
        or len(set(scenarios)) != len(scenarios)
        or any(name not in SCENARIOS for name in scenarios)
        or "clean_source" not in scenarios
    ):
        raise DataContractError("Invalid predefined scenarios")
    if raw["waits"] != [0, 5, 15] or any(type(x) is not int for x in raw["waits"]):
        raise DataContractError("T06 waits must be [0, 5, 15]")
    weights = raw["training_mixture_weights"]
    if (
        not isinstance(weights, dict)
        or weights.get("clean_source") != 0.25
        or len(weights) < 2
        or any(name not in SCENARIOS for name in weights)
    ):
        raise DataContractError("Training mixture requires clean weight .25 and predefined views")
    if any(
        type(w) not in (int, float) or not math.isfinite(w) or w <= 0 for w in weights.values()
    ) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-12):
        raise DataContractError("Training mixture weights must be positive and sum to one")
    if (
        not isinstance(raw["baseline_run"], str)
        or not raw["baseline_run"]
        or not isinstance(raw["baseline_manifest_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", raw["baseline_manifest_sha256"])
    ):
        raise DataContractError("Trusted local baseline and pinned SHA256 required")
    return raw
