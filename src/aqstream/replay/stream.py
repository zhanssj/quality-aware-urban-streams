"""Schedule forecasts on virtual time independently of message arrivals."""

import hashlib
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from aqstream.data import DataContractError
from aqstream.features import FallbackState, FeatureFrame, YearGrid
from aqstream.replay.calendars import Calendar
from aqstream.replay.delivery import DeliveryHistory, SensorEvent


@dataclass(frozen=True)
class ReplaySnapshot:
    wait_minutes: int
    features: FeatureFrame
    origin_minutes: int
    cutoff_minutes: int
    target_end_minutes: int
    remaining_lead_minutes: int
    receive_seconds: float


def replay_snapshots(
    values: np.ndarray,
    source_valid: np.ndarray,
    fitted: FallbackState,
    grid: YearGrid,
    calendar: Calendar,
    *,
    target_start: int,
    target_stop: int,
    waits: tuple[int, ...],
):
    if target_start < 1 or target_stop <= target_start or target_stop > len(values):
        raise DataContractError("Replay targets must lie within cached source grid")
    if source_valid.shape != values.shape or source_valid.dtype != bool:
        raise DataContractError("Replay source truth dimensions/validity must match values")
    if calendar.station_ids != fitted.station_ids or calendar.input_start != target_start - 1:
        raise DataContractError("Replay calendar identity/scope differs from target plan")
    if calendar.dropped.shape != (target_stop - target_start, len(fitted.station_ids)):
        raise DataContractError("Replay calendar shape differs from target plan")
    if (
        not waits
        or tuple(sorted(set(waits))) != waits
        or any(wait not in (0, 5, 15) for wait in waits)
    ):
        raise DataContractError("Replay waits must be unique sorted 0/5/15 policies")
    history = DeliveryHistory(fitted, grid)
    scope = slice(calendar.input_start, calendar.input_start + len(calendar.dropped))
    if calendar.source_valid_sha256 != hashlib.sha256(source_valid[scope].tobytes()).hexdigest():
        raise DataContractError("Replay source validity differs from saved calendar")
    history.prime(
        values[: calendar.input_start],
        source_valid[: calendar.input_start],
        first_target=target_start,
    )
    eligible = source_valid[scope] & ~calendar.dropped
    events = []
    for local, column in np.argwhere(eligible):
        index = calendar.input_start + int(local)
        events.append(
            SensorEvent(
                index,
                fitted.station_ids[column],
                float(values[index, column]),
                (index + 1) * 60 + float(calendar.delay_minutes[local, column]),
            )
        )
    events.sort(key=lambda event: (event.arrival_minutes, event.event_index, event.station_id))
    cursor = 0
    for target in range(target_start, target_stop):
        origin = target * 60
        for wait in waits:
            cutoff = origin + wait
            started = perf_counter()
            while cursor < len(events) and events[cursor].arrival_minutes <= cutoff:
                history.receive(events[cursor], cutoff_minutes=cutoff)
                cursor += 1
            receive_seconds = perf_counter() - started
            frame = history.forecast(target, cutoff_minutes=cutoff)
            yield ReplaySnapshot(
                wait, frame, origin, cutoff, (target + 1) * 60, 60 - wait, receive_seconds
            )
