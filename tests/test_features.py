import h5py
import numpy as np
import pytest

from aqstream.data import DataContractError, HDF5Reader
from aqstream.features import (
    COMMON_NAMES,
    QUALITY_NAMES,
    CausalFeatures,
    FallbackState,
    YearGrid,
    fit_fallback,
    scheduled_samples,
)


def fitted(ids=("s0",), medians=(4.0,)):
    return FallbackState(ids, medians, tuple(1 for _ in ids), 4.0, (2023,))


def write_year(root, year, data, masks=None):
    data = np.asarray(data, dtype=np.float64).reshape(len(data), -1, 1)
    with h5py.File(root / f"aq_compact_{year}.h5", "w") as handle:
        handle["values"] = data
        handle["masks"] = (
            np.isfinite(data) if masks is None else np.asarray(masks).reshape(data.shape)
        )
        handle["station_ids"] = np.array([f"s{i}".encode() for i in range(data.shape[1])])
        handle["params"] = [b"pm25"]


def history(values, available=None, state=None):
    builder = CausalFeatures(state or fitted(), YearGrid((2023,), (len(values) + 1,)))
    if available is None:
        available = np.isfinite(values)
    for index, (value, good) in enumerate(zip(values, available, strict=True)):
        builder.observe(index, np.array([value]), np.array([good]))
    return builder.forecast()


def test_hand_calculated_lags_rolling_and_quality():
    frame = history([2, np.nan, 6, np.nan])
    common = dict(zip(COMMON_NAMES, frame.common[0], strict=True))
    quality = dict(zip(QUALITY_NAMES, frame.quality[0], strict=True))
    assert [common[f"lag_{i}"] for i in (1, 2, 3, 6, 12, 24)] == [6, 6, 2, 4, 4, 4]
    assert common["mean_6"] == 4
    assert common["std_6"] == pytest.approx(np.sqrt(16 / 6))
    assert quality["age"] == quality["current_gap"] == 1
    assert quality["missing_ratio_6"] == pytest.approx(4 / 6)
    assert quality["missing_ratio_24"] == pytest.approx(22 / 24)
    assert quality["has_history"] == 1
    assert [quality[f"observed_{i}"] for i in (1, 2, 3)] == [0, 1, 0]


def test_long_gap_preserves_positions_and_age():
    frame = history([10] + [np.nan] * 100)
    quality = dict(zip(QUALITY_NAMES, frame.quality[0], strict=True))
    assert frame.sample_ids[0].origin_index == 100
    assert frame.sample_ids[0].target_index == 101
    assert quality["age"] == quality["current_gap"] == 100
    assert quality["missing_ratio_6"] == quality["missing_ratio_24"] == 1
    np.testing.assert_array_equal(frame.common[0, :6], np.full(6, 10))


def test_no_history_and_common_pipeline_identity():
    frame = history([np.nan] * 3)
    quality = dict(zip(QUALITY_NAMES, frame.quality[0], strict=True))
    assert quality["has_history"] == 0 and quality["age"] == -1
    assert quality["current_gap"] == 3
    np.testing.assert_array_equal(frame.common[0, :6], np.full(6, 4))
    assert frame.persistence[0] == 4
    np.testing.assert_array_equal(frame.standard, frame.quality_aware[:, : len(COMMON_NAMES)])
    assert frame.sample_ids[0].key == "s0:2:3"
    with pytest.raises(ValueError):
        frame.common[0, 0] = 99


def test_hidden_values_do_not_affect_features():
    first = history([2, 999999, 6], [1, 0, 1])
    second = history([2, -999999, 6], [1, 0, 1])
    np.testing.assert_array_equal(first.common, second.common)
    np.testing.assert_array_equal(first.quality, second.quality)


def test_target_and_future_invariance_with_frozen_fit(tmp_path):
    state = fitted()
    write_year(tmp_path, 2023, [2, 3, 4, 5, 6, 7])
    reader = HDF5Reader(tmp_path)
    grid = YearGrid.from_reader(reader, (2023,))
    before = list(scheduled_samples(reader, state, grid))
    write_year(tmp_path, 2023, [2, 3, 4, 999, -999, 999])
    after = list(scheduled_samples(reader, state, grid))
    for index in range(4):  # Includes the changed target at index 3.
        np.testing.assert_array_equal(before[index][0].common, after[index][0].common)
        np.testing.assert_array_equal(before[index][0].quality, after[index][0].quality)
    assert before[3][1].values[0] != after[3][1].values[0]


def test_split_uses_target_and_history_crosses_year(tmp_path):
    write_year(tmp_path, 2023, [1, 2, 3])
    write_year(tmp_path, 2024, [4, np.nan])
    reader = HDF5Reader(tmp_path)
    state = fit_fallback(reader, ["s0"], years=(2023,))
    frames = list(scheduled_samples(reader, state, YearGrid.from_reader(reader, (2023, 2024))))
    boundary = frames[3][0]
    identity = boundary.sample_ids[0]
    assert (identity.origin_index, identity.target_index, identity.target_year) == (2, 3, 2024)
    assert identity.split == "validation"
    assert frames[2][0].sample_ids[0].split == "train"
    assert boundary.common[0, 0] == 3
    assert len(frames) == 5  # Missing target is scheduled too.
    assert not frames[-1][1].source_valid[0]
    assert np.isnan(frames[-1][1].values[0])


def test_fallback_fit_train_only_and_global_for_empty_station(tmp_path, monkeypatch):
    write_year(tmp_path, 2023, [[2, np.nan], [4, np.nan], [0, np.nan]])
    reader = HDF5Reader(tmp_path)
    calls = []
    original = reader.iter_blocks

    def train_only(**kwargs):
        calls.append(kwargs["year"])
        assert kwargs["year"] == 2023
        return original(**kwargs)

    monkeypatch.setattr(reader, "iter_blocks", train_only)
    state = fit_fallback(reader, ["s0", "s1"], years=(2023,))
    assert state.medians == (2, 2) and state.global_median == 2
    assert state.train_counts == (3, 0)
    calls.clear()
    for years in ((2024,), (2023, 2024), (2025,)):
        with pytest.raises(DataContractError, match="restricted to train"):
            fit_fallback(reader, ["s0"], years=years)
    assert calls == []


def test_fit_is_explicit_and_no_observations_is_error(tmp_path):
    with pytest.raises(DataContractError, match="fitted"):
        CausalFeatures(None, YearGrid((2023,), (2,)))
    write_year(tmp_path, 2023, [np.nan, np.nan])
    with pytest.raises(DataContractError, match="No source-valid training"):
        fit_fallback(HDF5Reader(tmp_path), ["s0"], years=(2023,))


def test_issued_frame_immutable_and_grid_order_enforced():
    builder = CausalFeatures(fitted(), YearGrid((2023,), (3,)))
    issued = builder.forecast()
    snapshot = issued.quality_aware.copy()
    builder.observe(0, np.array([10]), np.array([True]))
    np.testing.assert_array_equal(issued.quality_aware, snapshot)
    assert issued.persistence[0] == 4
    with pytest.raises(DataContractError, match="every grid position"):
        builder.observe(2, np.array([10]), np.array([True]))


def test_grid_rejects_test_before_any_open(tmp_path, monkeypatch):
    reader = HDF5Reader(tmp_path)
    monkeypatch.setattr(reader, "inspect", lambda year: pytest.fail("must not inspect test"))
    with pytest.raises(DataContractError, match="2025 is CLOSED"):
        YearGrid.from_reader(reader, (2023, 2024, 2025))


def test_multiple_ring_wraps_and_asynchronous_stations_match_independent_reference():
    rng = np.random.default_rng(57)
    source = rng.normal(size=(151, 2))
    source[3, 0] = 0
    available = rng.random(source.shape) > 0.4
    available[3, 0] = True
    state = fitted(("s0", "s1"), (4.0, 8.0))
    builder = CausalFeatures(state, YearGrid((2023,), (152,)))
    filled = [np.array(state.medians) for _ in range(24)]
    masks = [np.zeros(2, dtype=bool) for _ in range(24)]
    last_value = np.array(state.medians)
    last_index = np.full(2, -1)
    gap = np.zeros(2)
    for target in range(151):
        frame = builder.forecast()
        lags = np.stack([filled[-lag] for lag in (1, 2, 3, 6, 12, 24)], axis=1)
        observed = np.stack([masks[-lag] for lag in (1, 2, 3, 6, 12, 24)], axis=1)
        expected_common = np.column_stack(
            (
                lags,
                np.mean(filled[-6:], axis=0),
                np.std(filled[-6:], axis=0),
                np.mean(filled[-24:], axis=0),
                np.std(filled[-24:], axis=0),
            )
        )
        expected_quality = np.column_stack(
            (
                observed,
                np.where(last_index >= 0, target - 1 - last_index, -1),
                1 - np.mean(masks[-6:], axis=0),
                1 - np.mean(masks[-24:], axis=0),
                gap,
                last_index >= 0,
            )
        )
        np.testing.assert_allclose(frame.common, expected_common, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(frame.quality, expected_quality)
        np.testing.assert_array_equal(frame.persistence, last_value)
        last_value = np.where(available[target], source[target], last_value)
        last_index = np.where(available[target], target, last_index)
        gap = np.where(available[target], 0, gap + 1)
        filled.append(last_value.copy())
        masks.append(available[target].copy())
        builder.observe(target, source[target], available[target])
