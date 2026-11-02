import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from aqstream.data import DataContractError


@dataclass(frozen=True)
class BaselineConfig:
    train_years: tuple[int, ...]
    validation_year: int
    cohort_candidates: Path
    cohort_size: int
    seed: int
    train_stride: int
    max_train_samples: int
    chunk_size: int
    predict_batch_steps: int
    threads: int
    max_rss_mb: int
    model_params: dict


def load_baseline_config(path: Path) -> BaselineConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DataContractError(f"Cannot load baseline config {path}: {exc}") from exc
    required = {"task_id", *BaselineConfig.__dataclass_fields__}
    if not isinstance(raw, dict) or set(raw) != required or raw["task_id"] != "B01":
        raise DataContractError(f"{path}: expected B01 config keys {sorted(required)}")
    years = raw["train_years"]
    if not isinstance(years, list) or not years or any(type(y) is not int for y in years):
        raise DataContractError("train_years must be a list of training years")
    if years[-1] != 2023 or years[0] < 2021 or years != list(range(years[0], 2024)):
        raise DataContractError("train_years must be contiguous training years ending in 2023")
    if type(raw["validation_year"]) is not int or raw["validation_year"] != 2024:
        raise DataContractError("Validation is 2024 only; test gate 2025 is CLOSED")
    limits = {
        "cohort_size": (1, 32),
        "seed": (0, 2**32 - 1),
        "train_stride": (1, 1000),
        "max_train_samples": (1, 200000),
        "chunk_size": (1, 512),
        "predict_batch_steps": (1, 512),
        "threads": (1, 8),
        "max_rss_mb": (256, 4096),
    }
    for name, (low, high) in limits.items():
        if type(raw[name]) is not int or not low <= raw[name] <= high:
            raise DataContractError(f"{name} must be an integer between {low} and {high}")
    params = raw["model_params"]
    if not isinstance(params, dict) or set(params) != {
        "max_iter",
        "max_leaf_nodes",
        "learning_rate",
        "l2_regularization",
        "min_samples_leaf",
    }:
        raise DataContractError("Expected fixed shared HistGradientBoosting model_params")
    for name, low, high in (
        ("max_iter", 1, 200),
        ("max_leaf_nodes", 2, 63),
        ("min_samples_leaf", 1, 1000),
    ):
        if type(params[name]) is not int or not low <= params[name] <= high:
            raise DataContractError(f"{name} must be an integer between {low} and {high}")
    for name in ("learning_rate", "l2_regularization"):
        if type(params[name]) not in (int, float) or not math.isfinite(params[name]):
            raise DataContractError(f"{name} must be a finite number")
    if not 0 < params["learning_rate"] <= 1 or params["l2_regularization"] < 0:
        raise DataContractError("Invalid learning_rate/l2_regularization")
    if not isinstance(raw["cohort_candidates"], str) or not raw["cohort_candidates"]:
        raise DataContractError("cohort_candidates must be a path")
    values = {name: raw[name] for name in BaselineConfig.__dataclass_fields__}
    values.update(train_years=tuple(years), cohort_candidates=Path(raw["cohort_candidates"]))
    return BaselineConfig(**values)
