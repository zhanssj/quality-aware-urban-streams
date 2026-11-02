import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from aqstream.data import DataContractError
from aqstream.replay.calendars import ScenarioSpec

SCENARIOS = {
    spec.name: spec
    for spec in (
        ScenarioSpec("clean_source"),
        ScenarioSpec("random_drop05", random_drop_rate=0.05),
        ScenarioSpec("random_drop10", random_drop_rate=0.10),
        ScenarioSpec("random_drop20", random_drop_rate=0.20),
        ScenarioSpec("single_block6", block_duration=6, block_station_count=1),
        ScenarioSpec("single_block24", block_duration=24, block_station_count=1),
        ScenarioSpec("group_block6", block_duration=6, block_station_fraction=0.25),
        ScenarioSpec("group_block24", block_duration=24, block_station_fraction=0.25),
        ScenarioSpec("delay10", delay_fraction=0.10),
        ScenarioSpec("delay30", delay_fraction=0.30),
        ScenarioSpec("random10_delay30", random_drop_rate=0.10, delay_fraction=0.30),
    )
}


@dataclass(frozen=True)
class ReplayConfig:
    baseline_run: Path
    baseline_manifest_sha256: str
    validation_year: int
    validation_steps: int
    validation_seeds: tuple[int, ...]
    train_mixture_seeds: tuple[int, ...]
    scenarios: tuple[str, ...]
    waits: tuple[int, ...]
    training_mixture_weights: dict[str, float]
    full_zero_control: bool
    chunk_size: int
    batch_steps: int
    threads: int
    max_rss_mb: int


def load_replay_config(path: Path) -> ReplayConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DataContractError(f"Cannot read replay config {path}: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != {"task_id", *ReplayConfig.__dataclass_fields__}:
        raise DataContractError("Invalid B02 replay config fields")
    if (
        raw["task_id"] != "B02"
        or type(raw["validation_year"]) is not int
        or raw["validation_year"] != 2024
    ):
        raise DataContractError("B02 is validation 2024 only; test gate 2025 is CLOSED")
    for name, low, high in (
        ("validation_steps", 1, 8784),
        ("chunk_size", 1, 512),
        ("batch_steps", 1, 512),
        ("threads", 1, 8),
        ("max_rss_mb", 256, 4096),
    ):
        if type(raw[name]) is not int or not low <= raw[name] <= high:
            raise DataContractError(f"Invalid B02 bound: {name}")
    for name in ("validation_seeds", "train_mixture_seeds"):
        values = raw[name]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise DataContractError(f"Expected unique nonempty {name}")
        if any(type(seed) is not int or not 0 < seed < 2**32 for seed in values):
            raise DataContractError(f"Invalid seeds: {name}")
    if set(raw["validation_seeds"]) & set(raw["train_mixture_seeds"]):
        raise DataContractError("Train mixture and validation seeds must be independent")
    if (
        not isinstance(raw["scenarios"], list)
        or not raw["scenarios"]
        or len(set(raw["scenarios"])) != len(raw["scenarios"])
    ):
        raise DataContractError("Expected unique scenario names")
    if (
        any(name not in SCENARIOS for name in raw["scenarios"])
        or "clean_source" not in raw["scenarios"]
    ):
        raise DataContractError("Use predefined B02 scenarios including clean_source control")
    if raw["waits"] != [0, 5, 15] or any(type(wait) is not int for wait in raw["waits"]):
        raise DataContractError("B02 waits must be [0, 5, 15]")
    weights = raw["training_mixture_weights"]
    if (
        not isinstance(weights, dict)
        or not weights
        or any(name not in SCENARIOS for name in weights)
    ):
        raise DataContractError("Invalid training mixture scenarios")
    if any(
        type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0
        for weight in weights.values()
    ):
        raise DataContractError("Invalid training mixture weight")
    if not math.isclose(sum(weights.values()), 1, rel_tol=0, abs_tol=1e-12):
        raise DataContractError("Training mixture weights must sum to one")
    if type(raw["full_zero_control"]) is not bool:
        raise DataContractError("full_zero_control must be boolean")
    if not isinstance(raw["baseline_run"], str) or not raw["baseline_run"]:
        raise DataContractError("baseline_run must identify trusted local B01 artifacts")
    if not isinstance(raw["baseline_manifest_sha256"], str) or not re.fullmatch(
        r"[a-f0-9]{64}", raw["baseline_manifest_sha256"]
    ):
        raise DataContractError("A pinned baseline manifest SHA256 is required")
    values = {name: raw[name] for name in ReplayConfig.__dataclass_fields__}
    values["baseline_run"] = Path(raw["baseline_run"])
    for name in ("validation_seeds", "train_mixture_seeds", "scenarios", "waits"):
        values[name] = tuple(values[name])
    return ReplayConfig(**values)
