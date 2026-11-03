"""Virtual delivery observer. No truth, corruption type, or future calendar in features."""

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from aqstream.data import DataContractError
from aqstream.features import FallbackState, FeatureFrame, YearGrid
from aqstream.features.causal import frame_from_history


@dataclass(frozen=True)
class SensorEvent:
    event_index: int
    station_id: str
    value: float
    arrival_minutes: float

    @property
    def interval_end_minutes(self) -> int:
        return (self.event_index + 1) * 60


class DeliveryHistory:
    """As-of snapshots with event-time ordering and idempotent received messages.

    Clock is monotone. Prefix last-known state is updated by the newest event index,
    while an old late message may still update a historical window in future forecasts.
    """

    def __init__(self, fitted: FallbackState, grid: YearGrid):
        self.fitted, self.grid = fitted, grid
        self._columns = {station: i for i, station in enumerate(fitted.station_ids)}
        self._delivered = np.full((sum(grid.lengths), len(fitted.station_ids)), np.nan)
        self._prefix_value = np.array(fitted.medians, dtype=np.float64)
        self._prefix_index = np.full(len(fitted.station_ids), -1, dtype=np.int64)
        self._prefix_cursor = 0
        self._last_target = -1
        self._minimum_target = 0
        self._clock = -math.inf
        self.delivered_count = 0

    def prime(self, values: np.ndarray, available: np.ndarray, *, first_target: int) -> None:
        """Already-delivered clean history strictly before the perturbed input scope."""
        if self.delivered_count or self._last_target >= 0:
            raise DataContractError("History can only be primed before replay")
        if (
            values.shape != available.shape
            or values.ndim != 2
            or values.shape[1] != len(self._columns)
        ):
            raise DataContractError("Invalid priming history dimensions")
        if len(values) > first_target - 1 or not 0 <= first_target < sum(self.grid.lengths):
            raise DataContractError("Priming history must precede the first perturbed input")
        valid = (available == 1) & np.isfinite(values)
        self._minimum_target = first_target
        self._delivered[: len(values)] = np.where(valid, values, np.nan)
        self.delivered_count = int(valid.sum())
        self._prefix_cursor = max(0, first_target - 24)
        for column in range(len(self._columns)):
            indices = np.flatnonzero(valid[: self._prefix_cursor, column])
            if len(indices):
                index = int(indices[-1])
                self._prefix_index[column] = index
                self._prefix_value[column] = values[index, column]

    def receive(self, event: SensorEvent, *, cutoff_minutes: float) -> bool:
        if not math.isfinite(cutoff_minutes) or cutoff_minutes < self._clock:
            raise DataContractError("Delivery clock must be finite and monotone")
        self._clock = cutoff_minutes
        if type(event.event_index) is not int or not 0 <= event.event_index < len(self._delivered):
            raise DataContractError("Event index outside observer grid")
        if event.station_id not in self._columns:
            raise DataContractError("Event station outside fitted cohort")
        if not isinstance(
            event.value, (float, np.float16, np.float32, np.float64)
        ) or not math.isfinite(event.value):
            raise DataContractError("Event value must be a finite float16/32/64 value")
        if not math.isfinite(event.arrival_minutes):
            raise DataContractError("Event arrival must be finite")
        if max(event.arrival_minutes, event.interval_end_minutes) > cutoff_minutes:
            return False  # Do not admit future arrivals or unfinished intervals.
        column = self._columns[event.station_id]
        previous = self._delivered[event.event_index, column]
        if np.isfinite(previous):
            if previous != event.value:
                raise DataContractError("Conflicting duplicate event value")
            return False
        self._delivered[event.event_index, column] = event.value
        self.delivered_count += 1
        if (
            event.event_index < self._prefix_cursor
            and event.event_index > self._prefix_index[column]
        ):
            self._prefix_index[column] = event.event_index
            self._prefix_value[column] = event.value
        return True

    def forecast(self, target_index: int, *, cutoff_minutes: float) -> FeatureFrame:
        started = perf_counter()
        self.grid.point(target_index)
        if target_index < self._last_target or cutoff_minutes < self._clock:
            raise DataContractError(
                "Forecast target/cutoff must be monotone; issued snapshots are immutable"
            )
        if target_index < self._minimum_target:
            raise DataContractError("Forecast precedes the primed history boundary")
        if not math.isfinite(cutoff_minutes) or cutoff_minutes < target_index * 60:
            raise DataContractError("Forecast cutoff precedes its origin")
        self._clock, self._last_target = cutoff_minutes, target_index
        left = max(0, target_index - 24)
        for index in range(self._prefix_cursor, left):
            row = self._delivered[index]
            good = np.isfinite(row)
            self._prefix_value = np.where(good, row, self._prefix_value)
            self._prefix_index = np.where(good, index, self._prefix_index)
        self._prefix_cursor = left
        last_value, last_index = self._prefix_value.copy(), self._prefix_index.copy()
        filled = np.empty((24, len(self._columns)), dtype=np.float64)
        available = np.zeros_like(filled, dtype=bool)
        for position, index in enumerate(range(target_index - 24, target_index)):
            if index >= 0:
                raw = self._delivered[index]
                good = np.isfinite(raw)
                last_value = np.where(good, raw, last_value)
                last_index = np.where(good, index, last_index)
                available[position] = good
            filled[position] = last_value
        gap = np.where(last_index >= 0, target_index - 1 - last_index, target_index)
        return frame_from_history(
            self.fitted,
            self.grid,
            target_index,
            filled,
            available,
            last_value,
            last_index,
            gap,
            started=started,
        )
