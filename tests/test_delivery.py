import numpy as np
import pytest

from aqstream.data import DataContractError
from aqstream.features import CausalFeatures, FallbackState, YearGrid
from aqstream.replay.delivery import DeliveryHistory, SensorEvent


def observer(length=200):
    state = FallbackState(("s0",), (4.0,), (1,), 4.0, (2023,))
    grid = YearGrid((2024,), (length,))
    return DeliveryHistory(state, grid), state, grid


def test_cutoff_tie_epsilon_and_interval_end():
    history, _, _ = observer()
    assert not history.receive(SensorEvent(0, "s0", 2.0, 5.0), cutoff_minutes=59.999)
    before = history.forecast(1, cutoff_minutes=60.0)
    assert before.persistence[0] == 4
    assert not history.receive(SensorEvent(0, "s0", 2.0, 65.000000001), cutoff_minutes=65.0)
    assert history.receive(SensorEvent(0, "s0", 2.0, 65.0), cutoff_minutes=65.0)
    after = history.forecast(1, cutoff_minutes=65.0)
    assert after.persistence[0] == 2
    assert before.persistence[0] == 4  # Previously issued snapshot is not rewritten.


def test_duplicates_out_of_order_and_stale_event_do_not_replace_last_known():
    frames = []
    for order in ((0, 2, 1), (2, 1, 0)):
        history, _, _ = observer()
        events = [SensorEvent(i, "s0", float(i + 1), 180.0) for i in range(3)]
        for index in order:
            assert history.receive(events[index], cutoff_minutes=180.0)
        assert not history.receive(events[0], cutoff_minutes=180.0)
        assert history.delivered_count == 3
        frame = history.forecast(3, cutoff_minutes=180.0)
        assert frame.persistence[0] == 3
        assert frame.common[0, :3].tolist() == [3, 2, 1]
        frames.append(frame)
    np.testing.assert_array_equal(frames[0].common, frames[1].common)
    np.testing.assert_array_equal(frames[0].quality, frames[1].quality)


def test_target_and_future_intervals_are_never_inputs_even_if_arrival_is_early():
    history, _, _ = observer()
    history.receive(SensorEvent(1, "s0", 999.0, 0.0), cutoff_minutes=120.0)
    frame = history.forecast(1, cutoff_minutes=120.0)
    assert frame.persistence[0] == 4
    assert frame.quality[0, -1] == 0


def test_old_late_message_updates_future_history_but_not_last_known_or_issued_snapshot():
    history, _, _ = observer()
    history.receive(SensorEvent(1, "s0", 10.0, 120.0), cutoff_minutes=120.0)
    first = history.forecast(2, cutoff_minutes=120.0)
    old_common = first.common.copy()
    history.receive(SensorEvent(0, "s0", 3.0, 125.0), cutoff_minutes=125.0)
    second = history.forecast(2, cutoff_minutes=125.0)
    assert second.persistence[0] == 10
    assert second.common[0, 1] == 3  # lag 2 is now visible in this new snapshot.
    np.testing.assert_array_equal(first.common, old_common)


def test_very_late_prefix_message_and_full_empty_gap():
    history, _, _ = observer()
    empty = history.forecast(101, cutoff_minutes=6060.0)
    assert empty.quality[0, -1] == 0 and empty.quality[0, -2] == 101
    history.receive(SensorEvent(0, "s0", 7.0, 6065.0), cutoff_minutes=6065.0)
    late = history.forecast(101, cutoff_minutes=6065.0)
    assert late.persistence[0] == 7
    assert late.quality[0, 6] == 100
    history.receive(SensorEvent(5, "s0", 9.0, 6070.0), cutoff_minutes=6070.0)
    history.receive(SensorEvent(2, "s0", 1.0, 6070.0), cutoff_minutes=6070.0)
    assert history.forecast(101, cutoff_minutes=6070.0).persistence[0] == 9


def test_zero_delay_multiple_wraps_match_original_common_kernel():
    history, state, grid = observer()
    original = CausalFeatures(state, grid)
    rng = np.random.default_rng(41)
    values = rng.normal(size=150)
    valid = rng.random(150) > 0.3
    for target in range(150):
        if target and valid[target - 1]:
            history.receive(
                SensorEvent(target - 1, "s0", float(values[target - 1]), target * 60.0),
                cutoff_minutes=target * 60.0,
            )
        baseline = original.forecast()
        frame = history.forecast(target, cutoff_minutes=target * 60.0)
        np.testing.assert_array_equal(frame.common, baseline.common)
        np.testing.assert_array_equal(frame.quality, baseline.quality)
        np.testing.assert_array_equal(frame.persistence, baseline.persistence)
        assert frame.sample_ids == baseline.sample_ids
        original.observe(target, values[target : target + 1], valid[target : target + 1])


def test_prime_and_waits_have_same_target_and_remaining_lead():
    history, state, grid = observer()
    values = np.arange(70, dtype=float).reshape(-1, 1)
    history.prime(values, np.ones_like(values, dtype=bool), first_target=71)
    frames = [history.forecast(71, cutoff_minutes=71 * 60 + wait) for wait in (0, 5, 15)]
    assert all(frame.sample_ids == frames[0].sample_ids for frame in frames)
    for frame in frames[1:]:
        np.testing.assert_array_equal(frame.common, frames[0].common)
    assert [(71 + 1) * 60 - (71 * 60 + wait) for wait in (0, 5, 15)] == [60, 55, 45]
    with pytest.raises(DataContractError, match="monotone"):
        history.forecast(70, cutoff_minutes=70 * 60.0)


def test_priming_cannot_be_used_for_a_forecast_before_its_history_boundary():
    history, _, _ = observer()
    values = np.arange(70, dtype=float).reshape(-1, 1)
    history.prime(values, np.ones_like(values, dtype=bool), first_target=71)
    with pytest.raises(DataContractError, match="primed history boundary"):
        history.forecast(20, cutoff_minutes=1200.0)


def test_conflicting_duplicate_is_error_and_original_message_is_unchanged():
    history, _, _ = observer()
    history.receive(SensorEvent(0, "s0", 2.0, 60.0), cutoff_minutes=60.0)
    with pytest.raises(DataContractError, match="Conflicting duplicate"):
        history.receive(SensorEvent(0, "s0", 3.0, 60.0), cutoff_minutes=60.0)
    assert history.forecast(1, cutoff_minutes=60.0).persistence[0] == 2
