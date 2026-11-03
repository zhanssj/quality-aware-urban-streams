"""Explicit weighted training sample views; B02 does not fit augmented models."""

from dataclasses import dataclass

import numpy as np

from aqstream.data import DataContractError
from aqstream.features import CausalFeatures, FallbackState, FeatureFrame, TruthFrame, YearGrid
from aqstream.features.causal import readonly
from aqstream.replay.calendars import ScenarioSpec, training_mixture
from aqstream.replay.stream import replay_snapshots


@dataclass(frozen=True)
class WeightedTrainingSample:
    calendar_sha256: str
    weight: float
    features: FeatureFrame
    truth: TruthFrame


def weighted_training_samples(
    values: np.ndarray,
    source_valid: np.ndarray,
    fitted: FallbackState,
    grid: YearGrid,
    *,
    specs: tuple[ScenarioSpec, ...],
    weights: tuple[float, ...],
    train_seed: int,
):
    if any(year not in (2021, 2022, 2023) for year in grid.years):
        raise DataContractError("Training mixture cannot read validation/test values")
    if values.shape != source_valid.shape or len(values) != sum(grid.lengths) or len(values) < 2:
        raise DataContractError("Training mixture needs the complete nonempty training grid")
    views = training_mixture(
        source_valid[:-1], fitted.station_ids, specs=specs, weights=weights, train_seed=train_seed
    )

    def record(calendar_hash, weight, frame):
        target = frame.sample_ids[0].target_index
        valid = source_valid[target] & np.isfinite(values[target])
        truth = TruthFrame(
            frame.sample_ids,
            readonly(np.where(valid, values[target], np.nan)),
            readonly(valid.copy()),
        )
        return WeightedTrainingSample(calendar_hash, weight, frame, truth)

    for calendar, weight in views:
        calendar_hash = calendar.fingerprint
        yield record(calendar_hash, weight, CausalFeatures(fitted, grid).forecast())
        for snapshot in replay_snapshots(
            values,
            source_valid,
            fitted,
            grid,
            calendar,
            target_start=1,
            target_stop=len(values),
            waits=(0,),
        ):
            yield record(calendar_hash, weight, snapshot.features)
