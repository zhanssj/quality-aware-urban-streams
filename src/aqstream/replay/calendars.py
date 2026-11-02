"""One immutable, seeded calendar on the event grid, shared by all methods/waits."""

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aqstream.artifacts import write_json
from aqstream.data import DataContractError
from aqstream.features.causal import readonly


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    random_drop_rate: float = 0.0
    block_duration: int = 0
    block_period: int = 168
    block_station_count: int = 0
    block_station_fraction: float = 0.0
    delay_fraction: float = 0.0
    min_delay_minutes: int = 1
    max_delay_minutes: int = 30

    def __post_init__(self):
        if not isinstance(self.name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.name):
            raise DataContractError("Scenario name must be a safe lowercase identifier")
        for name in ("random_drop_rate", "block_station_fraction", "delay_fraction"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise DataContractError(f"Invalid scenario probability: {name}")
        for name in (
            "block_duration",
            "block_period",
            "block_station_count",
            "min_delay_minutes",
            "max_delay_minutes",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise DataContractError(f"Invalid scenario integer: {name}")
        if not 0 <= self.block_duration <= self.block_period or self.block_period == 0:
            raise DataContractError("Block duration must fit a positive period")
        if self.block_duration and (
            bool(self.block_station_count) == bool(self.block_station_fraction)
        ):
            raise DataContractError("Block requires exactly one station count or fraction")
        if not 1 <= self.min_delay_minutes <= self.max_delay_minutes:
            raise DataContractError("Delay range must be positive and ordered")


@dataclass(frozen=True)
class Calendar:
    spec: ScenarioSpec
    seed: int
    station_ids: tuple[str, ...]
    input_start: int
    dropped: np.ndarray
    delay_minutes: np.ndarray
    statistics: dict
    blocks: tuple[dict, ...]
    source_valid_sha256: str

    @property
    def scenario_id(self) -> str:
        return f"{self.spec.name}_seed{self.seed}"

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                {
                    "spec": asdict(self.spec),
                    "seed": self.seed,
                    "station_ids": self.station_ids,
                    "input_start": self.input_start,
                    "source_valid_sha256": self.source_valid_sha256,
                },
                sort_keys=True,
            ).encode()
        )
        digest.update(self.dropped.tobytes())
        digest.update(self.delay_minutes.astype("<f8").tobytes())
        return digest.hexdigest()

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(
            directory / "calendar.json",
            {
                "scenario_id": self.scenario_id,
                "spec": asdict(self.spec),
                "seed": self.seed,
                "station_ids": self.station_ids,
                "input_start": self.input_start,
                "input_stop": self.input_start + self.dropped.shape[0],
                "calendar_sha256": self.fingerprint,
                "source_valid_sha256": self.source_valid_sha256,
                "statistics": self.statistics,
                "blocks": self.blocks,
                "rng": "PCG64 SeedSequence([seed, stream]); drop=101, delay=202, block=303",
            },
        )
        np.savez_compressed(
            directory / "calendar.npz", dropped=self.dropped, delay_minutes=self.delay_minutes
        )
        with (directory / "changed_events.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["event_index", "station_id", "artificial_drop", "delay_minutes", "arrival_minutes"]
            )
            for local, column in np.argwhere(self.dropped | (self.delay_minutes > 0)):
                index = self.input_start + int(local)
                delay = float(self.delay_minutes[local, column])
                dropped = bool(self.dropped[local, column])
                writer.writerow(
                    [
                        index,
                        self.station_ids[column],
                        int(dropped),
                        delay,
                        "" if dropped else (index + 1) * 60 + delay,
                    ]
                )


def make_calendar(
    source_valid: np.ndarray,
    station_ids: tuple[str, ...],
    *,
    input_start: int,
    spec: ScenarioSpec,
    seed: int,
) -> Calendar:
    if (
        source_valid.dtype != bool
        or source_valid.ndim != 2
        or source_valid.shape[1] != len(station_ids)
    ):
        raise DataContractError(
            "Calendar requires a boolean source-valid grid matching station IDs"
        )
    if not source_valid.shape[0] or not station_ids or len(set(station_ids)) != len(station_ids):
        raise DataContractError("Calendar requires nonempty grid and unique station IDs")
    if type(seed) is not int or seed < 0 or type(input_start) is not int or input_start < 0:
        raise DataContractError("Calendar seed/input_start must be nonnegative integers")
    steps, stations = source_valid.shape
    valid_indices = np.flatnonzero(source_valid)
    requested = np.zeros_like(source_valid)
    rng_drop = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 101])))
    order = rng_drop.permutation(valid_indices)
    requested.flat[order[: round(spec.random_drop_rate * len(valid_indices))]] = True
    blocks = []
    if spec.block_duration:
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 303])))
        count = spec.block_station_count or math.ceil(spec.block_station_fraction * stations)
        if not 1 <= count <= stations:
            raise DataContractError("Block station count exceeds cohort")
        columns = np.sort(rng.choice(stations, count, replace=False))
        for left in range(0, steps, spec.block_period):
            stop = min(left + spec.block_period, steps)
            if stop - left < spec.block_duration:
                continue  # No truncation of the declared duration.
            start = int(rng.integers(left, stop - spec.block_duration + 1))
            requested[start : start + spec.block_duration, columns] = True
            blocks.append(
                {
                    "start": input_start + start,
                    "stop": input_start + start + spec.block_duration,
                    "station_ids": [station_ids[int(column)] for column in columns],
                }
            )
    dropped = requested & source_valid
    eligible_messages = source_valid & ~dropped
    rng_delay = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, 202])))
    candidates = rng_delay.permutation(np.flatnonzero(eligible_messages))
    selected = candidates[: round(spec.delay_fraction * len(candidates))]
    delay = np.zeros(source_valid.shape, dtype=np.float64)
    delay.flat[selected] = rng_delay.integers(
        spec.min_delay_minutes, spec.max_delay_minutes + 1, len(selected)
    )
    total, usable, lost = source_valid.size, int(source_valid.sum()), int(dropped.sum())
    stats = {
        "total_input_cells": total,
        "source_valid_input_cells": usable,
        "natural_unavailable_cells": total - usable,
        "artificially_lost_source_valid": lost,
        "loss_fraction_source_valid": lost / usable if usable else None,
        "loss_fraction_all_cells": lost / total,
        "total_unavailable_after_drop_fraction": (total - usable + lost) / total,
        "requested_block_or_random_cells": int(requested.sum()),
        "requested_already_source_invalid": int((requested & ~source_valid).sum()),
        "blocks_count": len(blocks),
        "delayed_messages": len(selected),
        "delay_eligible_messages": len(candidates),
        "delayed_fraction_eligible_messages": len(selected) / len(candidates)
        if len(candidates)
        else None,
        "delay_minutes_min": float(delay.flat[selected].min()) if len(selected) else None,
        "delay_minutes_max": float(delay.flat[selected].max()) if len(selected) else None,
    }
    return Calendar(
        spec,
        seed,
        tuple(station_ids),
        input_start,
        readonly(dropped),
        readonly(delay),
        stats,
        tuple(blocks),
        hashlib.sha256(source_valid.tobytes()).hexdigest(),
    )


def training_mixture(
    source_valid: np.ndarray,
    station_ids: tuple[str, ...],
    *,
    specs: tuple[ScenarioSpec, ...],
    weights: tuple[float, ...],
    train_seed: int,
) -> tuple[tuple[Calendar, float], ...]:
    """Weighted global calendar views, no fit and no implicit overweighting of clean copies."""
    if len(specs) != len(weights) or not specs or len({spec.name for spec in specs}) != len(specs):
        raise DataContractError(
            "Training mixture requires unique scenarios and one weight per view"
        )
    if any(
        type(weight) not in (int, float) or not math.isfinite(weight) or weight <= 0
        for weight in weights
    ):
        raise DataContractError("Training mixture weights must be finite and positive")
    if not math.isclose(sum(weights), 1, rel_tol=0, abs_tol=1e-12):
        raise DataContractError("Training mixture weights must sum to one")
    return tuple(
        (
            make_calendar(source_valid, station_ids, input_start=0, spec=spec, seed=train_seed),
            weight,
        )
        for spec, weight in zip(specs, weights, strict=True)
    )
