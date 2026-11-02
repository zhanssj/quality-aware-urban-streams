import numpy as np
import pytest

from aqstream.data import DataContractError
from aqstream.features import FallbackState, YearGrid
from aqstream.replay.calendars import ScenarioSpec, make_calendar, training_mixture
from aqstream.replay.mixture import weighted_training_samples


def test_calendar_reproducibility_and_explicit_denominators(tmp_path):
    valid = np.ones((100, 4), dtype=bool)
    valid[::5] = False
    spec = ScenarioSpec("random20", random_drop_rate=0.2, delay_fraction=0.3)
    first = make_calendar(valid, ("a", "b", "c", "d"), input_start=9, spec=spec, seed=3101)
    second = make_calendar(valid, ("a", "b", "c", "d"), input_start=9, spec=spec, seed=3101)
    assert first.fingerprint == second.fingerprint
    np.testing.assert_array_equal(first.dropped, second.dropped)
    assert not (first.dropped & ~valid).any()
    assert first.statistics["source_valid_input_cells"] == 320
    assert first.statistics["artificially_lost_source_valid"] == 64
    assert first.statistics["loss_fraction_source_valid"] == 0.2
    assert first.statistics["loss_fraction_all_cells"] == 0.16
    assert first.statistics["total_unavailable_after_drop_fraction"] == 0.36
    assert first.statistics["delay_eligible_messages"] == 256
    assert first.statistics["delayed_messages"] == round(0.3 * 256)
    assert not (first.dropped & (first.delay_minutes > 0)).any()
    assert not ((first.delay_minutes > 0) & ~valid).any()
    first.save(tmp_path / "calendar")
    assert (tmp_path / "calendar/calendar.json").is_file()
    with pytest.raises(ValueError):
        first.dropped[0, 0] = True


@pytest.mark.parametrize("duration", [6, 24])
@pytest.mark.parametrize("group", [False, True])
def test_exact_station_group_and_intervals(duration, group):
    valid = np.ones((336, 8), dtype=bool)
    valid[:, 0] = False
    ids = tuple(f"s{i}" for i in range(8))
    spec = ScenarioSpec(
        "blocks",
        block_duration=duration,
        block_station_count=0 if group else 1,
        block_station_fraction=0.25 if group else 0,
    )
    calendar = make_calendar(valid, ids, input_start=10, spec=spec, seed=20)
    expected = np.zeros_like(valid)
    assert len(calendar.blocks) == 2
    for block in calendar.blocks:
        assert block["stop"] - block["start"] == duration
        assert len(block["station_ids"]) == (2 if group else 1)
        columns = [ids.index(station) for station in block["station_ids"]]
        expected[block["start"] - 10 : block["stop"] - 10, columns] = True
    np.testing.assert_array_equal(calendar.dropped, expected & valid)


def test_no_natural_target_is_created_and_empty_source_rates_are_undefined():
    valid = np.zeros((10, 2), dtype=bool)
    truth = valid.copy()
    calendar = make_calendar(
        valid,
        ("a", "b"),
        input_start=0,
        spec=ScenarioSpec("empty", random_drop_rate=0.2, delay_fraction=0.3),
        seed=4,
    )
    assert calendar.statistics["artificially_lost_source_valid"] == 0
    assert calendar.statistics["loss_fraction_source_valid"] is None
    np.testing.assert_array_equal(valid, truth)


def test_weighted_training_views_explicit_and_independent_seed():
    valid = np.ones((100, 2), dtype=bool)
    specs = (ScenarioSpec("clean"), ScenarioSpec("random10", random_drop_rate=0.1))
    views = training_mixture(valid, ("a", "b"), specs=specs, weights=(0.25, 0.75), train_seed=2101)
    assert [weight for _, weight in views] == [0.25, 0.75]
    assert all(calendar.seed == 2101 for calendar, _ in views)
    validation = make_calendar(valid, ("a", "b"), input_start=0, spec=specs[1], seed=3101)
    assert validation.fingerprint != views[1][0].fingerprint
    with pytest.raises(DataContractError, match="sum to one"):
        training_mixture(valid, ("a", "b"), specs=specs, weights=(1.0, 1.0), train_seed=2101)


def test_weighted_training_generator_preserves_truth_and_total_weight_per_sample():
    values = np.arange(12, dtype=float).reshape(-1, 1)
    valid = np.ones_like(values, dtype=bool)
    valid[3] = False
    state = FallbackState(("s0",), (4.0,), (1,), 4.0, (2023,))
    rows = list(
        weighted_training_samples(
            values,
            valid,
            state,
            YearGrid((2023,), (12,)),
            specs=(ScenarioSpec("clean"), ScenarioSpec("drop", random_drop_rate=1.0)),
            weights=(0.25, 0.75),
            train_seed=2101,
        )
    )
    sums = {}
    for row in rows:
        key = row.features.sample_ids[0].key
        sums[key] = sums.get(key, 0) + row.weight
        target = row.features.sample_ids[0].target_index
        assert row.truth.source_valid[0] == valid[target, 0]
        if valid[target, 0]:
            assert row.truth.values[0] == values[target, 0]
        else:
            assert np.isnan(row.truth.values[0])
    assert len(rows) == 24 and len(sums) == 12 and set(sums.values()) == {1.0}
    with pytest.raises(DataContractError, match="cannot read validation"):
        list(
            weighted_training_samples(
                values,
                valid,
                state,
                YearGrid((2024,), (12,)),
                specs=(ScenarioSpec("clean"),),
                weights=(1.0,),
                train_seed=2101,
            )
        )
