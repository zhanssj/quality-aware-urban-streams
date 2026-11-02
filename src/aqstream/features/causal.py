"""Frozen train preprocessing and scheduled forecasts from past observations only."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from aqstream.data import DataContractError, HDF5Reader

LAGS = (1, 2, 3, 6, 12, 24)
COMMON_NAMES = tuple(f"lag_{lag}" for lag in LAGS) + ("mean_6", "std_6", "mean_24", "std_24")
QUALITY_NAMES = tuple(f"observed_{lag}" for lag in LAGS) + (
    "age",
    "missing_ratio_6",
    "missing_ratio_24",
    "current_gap",
    "has_history",
)


def readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class FallbackState:
    station_ids: tuple[str, ...]
    medians: tuple[float, ...]
    train_counts: tuple[int, ...]
    global_median: float
    train_years: tuple[int, ...]

    def __post_init__(self):
        n = len(self.station_ids)
        if not n or len(set(self.station_ids)) != n:
            raise DataContractError("Fitted fallback requires unique station IDs")
        if len(self.medians) != n or len(self.train_counts) != n:
            raise DataContractError("Fitted fallback dimensions do not match station IDs")
        if not np.isfinite([*self.medians, self.global_median]).all():
            raise DataContractError("Fitted fallback must be finite")
        if not self.train_years or any(year not in (2021, 2022, 2023) for year in self.train_years):
            raise DataContractError("Fallback fit is restricted to train years 2021–2023")


def fit_fallback(
    reader: HDF5Reader,
    station_ids: Sequence[str],
    *,
    years: Sequence[int] = (2021, 2022, 2023),
    chunk_size: int = 168,
) -> FallbackState:
    if not years or any(type(year) is not int or year not in (2021, 2022, 2023) for year in years):
        raise DataContractError("Fallback fit is restricted to train years 2021–2023")
    if len(set(years)) != len(years):
        raise DataContractError("Duplicate training years")
    ids = tuple(station_ids)
    samples: list[list[np.ndarray]] = [[] for _ in ids]
    for year in years:
        for block in reader.iter_blocks(year=year, station_ids=ids, chunk_size=chunk_size):
            if block.station_ids != ids:
                raise DataContractError("Training station order mismatch")
            for column in range(len(ids)):
                valid = block.source_valid[:, column] & np.isfinite(block.values[:, column])
                samples[column].append(block.values[valid, column])
    per_station = [np.concatenate(parts) if parts else np.empty(0) for parts in samples]
    counts = tuple(values.size for values in per_station)
    if not sum(counts):
        raise DataContractError("No source-valid training observations for fallback fit")
    global_median = float(np.median(np.concatenate(per_station)))
    medians = tuple(
        float(np.median(values)) if values.size else global_median for values in per_station
    )
    return FallbackState(ids, medians, counts, global_median, tuple(years))


@dataclass(frozen=True)
class SampleID:
    station_id: str
    origin_index: int
    target_index: int
    target_year: int
    target_local_index: int

    @property
    def key(self) -> str:
        return f"{self.station_id}:{self.origin_index}:{self.target_index}"

    @property
    def split(self) -> str:
        if self.target_year in (2021, 2022, 2023):
            return "train"
        if self.target_year == 2024:
            return "validation"
        raise DataContractError("Test gate 2025 is CLOSED")


@dataclass(frozen=True)
class YearGrid:
    years: tuple[int, ...]
    lengths: tuple[int, ...]

    def __post_init__(self):
        if not self.years or self.years != tuple(range(self.years[0], self.years[-1] + 1)):
            raise DataContractError("Years must form a chronological contiguous grid")
        if any(year not in (2021, 2022, 2023, 2024) for year in self.years):
            raise DataContractError("Test gate 2025 is CLOSED")
        if len(self.lengths) != len(self.years) or any(length <= 0 for length in self.lengths):
            raise DataContractError("Grid requires nonempty lengths for each year")

    @classmethod
    def from_reader(cls, reader: HDF5Reader, years: Sequence[int]) -> YearGrid:
        # Validate years before inspecting any file, including before an attempted 2025 open.
        cls(tuple(years), tuple(1 for _ in years))
        return cls(tuple(years), tuple(reader.inspect(year).shape[0] for year in years))

    def point(self, index: int) -> tuple[int, int]:
        if not 0 <= index < sum(self.lengths):
            raise DataContractError(f"Target index outside grid: {index}")
        offset = 0
        for year, length in zip(self.years, self.lengths, strict=True):
            if index < offset + length:
                return year, index - offset
            offset += length
        raise AssertionError("unreachable grid position")


@dataclass(frozen=True)
class FeatureFrame:
    sample_ids: tuple[SampleID, ...]
    common: np.ndarray
    quality: np.ndarray
    persistence: np.ndarray
    compute_seconds: float = 0.0

    @property
    def standard(self) -> np.ndarray:
        return self.common

    @property
    def quality_aware(self) -> np.ndarray:
        return readonly(np.column_stack((self.common, self.quality)))


@dataclass(frozen=True)
class TruthFrame:
    sample_ids: tuple[SampleID, ...]
    values: np.ndarray
    source_valid: np.ndarray


def frame_from_history(
    fitted: FallbackState,
    grid: YearGrid,
    target: int,
    filled: np.ndarray,
    available: np.ndarray,
    last_values: np.ndarray,
    last_index: np.ndarray,
    current_gap: np.ndarray,
    *,
    started: float,
) -> FeatureFrame:
    """One common feature kernel for ordinary and delivery-aware observer snapshots."""
    year, local = grid.point(target)
    common = [filled[-np.array(LAGS)].T]
    missing = []
    for window in (6, 24):
        common.extend((filled[-window:].mean(axis=0), filled[-window:].std(axis=0)))
        missing.append(1 - available[-window:].mean(axis=0))
    has_history = last_index >= 0
    age = np.where(has_history, target - 1 - last_index, -1)
    quality = np.column_stack(
        (available[-np.array(LAGS)].T, age, *missing, current_gap, has_history)
    ).astype(np.float64)
    values = np.column_stack(common)
    if not np.isfinite(values).all():
        raise DataContractError("Feature arithmetic produced nonfinite common values")
    ids = tuple(
        SampleID(station, target - 1, target, year, local) for station in fitted.station_ids
    )
    return FeatureFrame(
        ids,
        readonly(values),
        readonly(quality),
        readonly(last_values.copy()),
        perf_counter() - started,
    )


class CausalFeatures:
    """Observer state. forecast accepts identity only, never truth or future events.

    forecast(t) occurs before observe(t). The 24-position buffer includes unavailable
    positions; forward filling uses the last source-available past observation.
    """

    def __init__(self, fitted: FallbackState, grid: YearGrid):
        if not isinstance(fitted, FallbackState):
            raise DataContractError("A frozen fitted train-only fallback is required")
        self.fitted = fitted
        self.grid = grid
        self.next_index = 0
        self._last = np.array(fitted.medians, dtype=np.float64)
        self._last_index = np.full(len(fitted.station_ids), -1, dtype=np.int64)
        self._gap = np.zeros(len(fitted.station_ids), dtype=np.int64)
        self._filled = np.tile(self._last, (24, 1))
        self._available = np.zeros_like(self._filled, dtype=bool)

    def forecast(self) -> FeatureFrame:
        started = perf_counter()
        target = self.next_index
        slots = np.arange(target - 24, target) % 24
        return frame_from_history(
            self.fitted,
            self.grid,
            target,
            self._filled[slots],
            self._available[slots],
            self._last,
            self._last_index,
            self._gap,
            started=started,
        )

    def observe(self, index: int, values: np.ndarray, available: np.ndarray) -> None:
        if index != self.next_index:
            raise DataContractError("Observations must preserve every grid position in order")
        if values.shape != self._last.shape or available.shape != values.shape:
            raise DataContractError("Observation dimensions differ from fitted station order")
        valid = (available == 1) & np.isfinite(values)
        self._last = np.where(valid, values, self._last)
        self._last_index = np.where(valid, index, self._last_index)
        self._gap = np.where(valid, 0, self._gap + 1)
        self._filled[index % 24] = self._last
        self._available[index % 24] = valid
        self.next_index += 1


def scheduled_samples(
    reader: HDF5Reader,
    fitted: FallbackState,
    grid: YearGrid,
    *,
    chunk_size: int = 168,
) -> Iterator[tuple[FeatureFrame, TruthFrame]]:
    builder = CausalFeatures(fitted, grid)
    for year, length in zip(grid.years, grid.lengths, strict=True):
        rows = 0
        for block in reader.iter_blocks(
            year=year, station_ids=fitted.station_ids, chunk_size=chunk_size
        ):
            for offset in range(block.stop - block.start):
                # Forecast is constructed before the target row is passed to the observer.
                features = builder.forecast()
                raw, mask = block.values[offset], block.source_valid[offset]
                truth = TruthFrame(features.sample_ids, readonly(raw.copy()), readonly(mask.copy()))
                yield features, truth
                builder.observe(builder.next_index, raw, mask)
                rows += 1
        if rows != length:
            raise DataContractError(f"Grid/reader row count mismatch in {year}")
