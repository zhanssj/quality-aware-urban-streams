"""Expected scheduled identities and one run/scenario/wait context (N1)."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from aqstream.data import DataContractError


@dataclass(frozen=True)
class PredictionPlan:
    station_ids: tuple[str, ...]
    target_start: int
    target_stop: int
    target_local_start: int
    run_id: str
    scenario_id: str
    wait_minutes: float
    target_year: int = 2024

    def __post_init__(self):
        object.__setattr__(self, "station_ids", tuple(self.station_ids))
        if not self.station_ids or len(set(self.station_ids)) != len(self.station_ids):
            raise DataContractError("Expected plan requires unique nonempty station IDs")
        if any(not isinstance(value, str) or not value.strip() for value in self.station_ids):
            raise DataContractError("Expected station IDs must be nonempty strings")
        for name in ("target_start", "target_stop", "target_local_start", "target_year"):
            if type(getattr(self, name)) is not int:
                raise DataContractError(f"Expected plan {name} must be an integer")
        if self.target_year != 2024:
            raise DataContractError("Evaluation is validation-only; test gate 2025 is CLOSED")
        if not 0 <= self.target_start < self.target_stop or self.target_local_start < 0:
            raise DataContractError("Invalid expected target grid")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.run_id, self.scenario_id)
        ):
            raise DataContractError("Expected plan requires run_id and scenario_id")
        if type(self.wait_minutes) not in (int, float) or not math.isfinite(self.wait_minutes):
            raise DataContractError("Invalid expected wait_minutes")
        if self.wait_minutes not in (0, 5, 15):
            raise DataContractError("Expected wait_minutes must be 0, 5 or 15")

    @property
    def expected_count(self) -> int:
        return len(self.station_ids) * (self.target_stop - self.target_start)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def validate_row(self, row: dict) -> None:
        try:
            origin, target = int(row["origin_index"]), int(row["target_index"])
            year, local = int(row["target_year"]), int(row["target_local_index"])
            wait = float(row["wait_minutes"])
            if (
                row["station_id"] not in self.station_ids
                or not self.target_start <= target < self.target_stop
            ):
                raise DataContractError(f"Unexpected sample ID: {row['sample_id']}")
            if origin != target - 1 or row["sample_id"] != f"{row['station_id']}:{origin}:{target}":
                raise DataContractError(f"Inconsistent sample identity: {row['sample_id']}")
            if (
                year != self.target_year
                or local != target - self.target_start + self.target_local_start
            ):
                raise DataContractError("Sample target year/local index differs from expected grid")
            if (row["run_id"], row["scenario_id"], wait) != (
                self.run_id,
                self.scenario_id,
                self.wait_minutes,
            ):
                raise DataContractError("Mixed or unexpected run/scenario/wait context")
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, DataContractError):
                raise
            raise DataContractError(f"Invalid planned prediction row: {exc}") from exc

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )

    @classmethod
    def load(cls, path: Path) -> PredictionPlan:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
                raise DataContractError("Expected plan fields do not match schema")
            return cls(**raw)
        except (OSError, TypeError, ValueError) as exc:
            raise DataContractError(f"Cannot load expected prediction plan {path}: {exc}") from exc
