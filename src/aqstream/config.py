from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from aqstream.data import DataContractError


@dataclass(frozen=True)
class PrepareConfig:
    year: int
    station_ids: tuple[str, ...]
    start: int
    stop: int
    chunk_size: int
    parameter: str
    artifact_root: Path


def load_prepare_config(path: Path) -> PrepareConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DataContractError(f"Cannot load config {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DataContractError(f"{path}: expected a configuration mapping")
    if raw.get("task_id") != "T01":
        raise DataContractError(f"{path}: only T01 prepare is implemented; baseline is a T03 plan")
    required = {"task_id", "year", "station_ids", "start", "stop", "chunk_size", "parameter"}
    if set(raw) != required:
        raise DataContractError(f"{path}: expected exactly these keys: {sorted(required)}")
    for key in ("year", "start", "stop", "chunk_size"):
        if type(raw[key]) is not int:
            raise DataContractError(f"{path}: {key} must be an integer")
    if raw["year"] not in (2021, 2022, 2023):
        raise DataContractError(f"{path}: T01 smoke is restricted to training years 2021–2023")
    if not 0 <= raw["start"] < raw["stop"] or raw["stop"] - raw["start"] > 168:
        raise DataContractError(f"{path}: smoke requires 1–168 rows, with 0 <= start < stop")
    if not 1 <= raw["chunk_size"] <= 168:
        raise DataContractError(f"{path}: chunk_size must be between 1 and 168")
    ids = raw["station_ids"]
    if not isinstance(ids, list) or not 2 <= len(ids) <= 4:
        raise DataContractError(f"{path}: smoke requires 2–4 station IDs")
    if any(type(item) not in (int, str) or not str(item).strip() for item in ids):
        raise DataContractError(f"{path}: station IDs must be nonempty strings or integers")
    selected = tuple(str(item) for item in ids)
    if len(set(selected)) != len(selected):
        raise DataContractError(f"{path}: duplicate station IDs")
    if not isinstance(raw["parameter"], str) or not raw["parameter"].strip():
        raise DataContractError(f"{path}: parameter must be a pollutant name")
    artifact_root = os.environ.get("AQ_ARTIFACT_ROOT", "artifacts")
    if not artifact_root.strip():
        raise DataContractError("AQ_ARTIFACT_ROOT must be a nonempty directory path")
    return PrepareConfig(
        raw["year"],
        selected,
        raw["start"],
        raw["stop"],
        raw["chunk_size"],
        raw["parameter"],
        Path(artifact_root).expanduser(),
    )
