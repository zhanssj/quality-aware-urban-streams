import json

import numpy as np
import pytest
import yaml
from test_experiment import experiment_inputs as experiment_inputs

from aqstream.artifacts import file_sha256
from aqstream.cli import main
from aqstream.data import DataContractError, HDF5Reader
from aqstream.experiment import run_experiment
from aqstream.features import FallbackState, YearGrid
from aqstream.models import Baselines
from aqstream.replay.calendars import ScenarioSpec, make_calendar
from aqstream.replay.config import load_replay_config
from aqstream.replay.runner import run_replay
from aqstream.replay.stream import replay_snapshots


def replay_config(path, baseline):
    config = {
        "task_id": "B02",
        "baseline_run": str(baseline),
        "baseline_manifest_sha256": file_sha256(baseline / "manifest.json"),
        "validation_year": 2024,
        "validation_steps": 12,
        "validation_seeds": [3101],
        "train_mixture_seeds": [2101],
        "scenarios": [
            "clean_source",
            "random_drop10",
            "group_block6",
            "delay30",
            "random10_delay30",
        ],
        "waits": [0, 5, 15],
        "training_mixture_weights": {"clean_source": 0.5, "random_drop10": 0.5},
        "full_zero_control": True,
        "chunk_size": 4,
        "batch_steps": 3,
        "threads": 1,
        "max_rss_mb": 2048,
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_frozen_replay_end_to_end_zero_parity_truth_and_shared_targets(
    experiment_inputs, monkeypatch
):
    baseline = run_experiment(experiment_inputs)
    source_model_sha = file_sha256(baseline / "models.joblib")
    source_csv_sha = file_sha256(baseline / "predictions_validation.csv")
    path = replay_config(experiment_inputs.parent / "replay.yaml", baseline)
    monkeypatch.setattr(Baselines, "fit", lambda *args: pytest.fail("B02 must never fit models"))
    original = HDF5Reader.iter_blocks
    opened = []

    def track(reader, **kwargs):
        opened.append(kwargs["year"])
        assert kwargs["year"] in (2023, 2024)
        return original(reader, **kwargs)

    monkeypatch.setattr(HDF5Reader, "iter_blocks", track)
    output = run_replay(path)
    assert opened == [2023, 2024]
    result = json.loads((output / "summary_metrics.json").read_text(encoding="utf-8"))
    assert result["full_zero_controls"][0]["baseline_parity"]["compared"] == 30
    assert all(
        value == 0
        for value in result["full_zero_controls"][0]["baseline_parity"][
            "max_absolute_prediction_difference"
        ].values()
    )
    cases = result["pilot_cases"]
    assert len(cases) == 11
    assert all(
        case["metrics"]["planned"] == 24 and case["metrics"]["schedule_validated"] for case in cases
    )
    shared = {
        metric["scored_target_ids_sha256"]
        for case in cases
        for metric in case["metrics"]["models"].values()
    }
    assert len(shared) == 1
    assert all(
        metric["scored"] == 15 for case in cases for metric in case["metrics"]["models"].values()
    )
    delayed = [case for case in cases if case["scenario_name"] == "delay30"]
    assert len({case["calendar_sha256"] for case in delayed}) == 1
    assert [case["remaining_lead_minutes"] for case in delayed] == [60, 55, 45]
    assert delayed[0]["lag1_available_at_cutoff"] <= delayed[-1]["lag1_available_at_cutoff"]
    assert source_model_sha == file_sha256(baseline / "models.joblib")
    assert source_csv_sha == file_sha256(baseline / "predictions_validation.csv")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["augmented_model_fit"] == "NOT_RUN"
    assert manifest["test_gate"] == "CLOSED"
    one = output / "cases" / cases[0]["case_id"]
    assert (
        main(
            [
                "evaluate",
                "--predictions",
                str(one / "predictions.csv"),
                "--plan",
                str(one / "evaluation_plan.json"),
                "--out",
                str(one / "recomputed.json"),
            ]
        )
        == 0
    )


def test_zero_messages_still_forecasts_and_drop_does_not_change_truth():
    values = np.arange(12, dtype=float).reshape(-1, 1)
    valid = np.ones_like(values, dtype=bool)
    truth = values.copy()
    fitted = FallbackState(("s0",), (4.0,), (1,), 4.0, (2023,))
    calendar = make_calendar(
        valid[:10],
        ("s0",),
        input_start=0,
        spec=ScenarioSpec("all_dropped", random_drop_rate=1.0),
        seed=5,
    )
    snapshots = list(
        replay_snapshots(
            values,
            valid,
            fitted,
            YearGrid((2024,), (12,)),
            calendar,
            target_start=1,
            target_stop=11,
            waits=(0, 5, 15),
        )
    )
    assert len(snapshots) == 30
    assert all(snapshot.features.persistence[0] == 4 for snapshot in snapshots)
    assert all(snapshot.features.quality[0, -1] == 0 for snapshot in snapshots)
    for index in range(0, len(snapshots), 3):
        assert snapshots[index].features.sample_ids == snapshots[index + 2].features.sample_ids
    np.testing.assert_array_equal(values, truth)
    assert valid.all()  # Artificially dropped inputs remain source-valid targets for the evaluator.


def test_blocks_recover_age_and_history_on_event_time():
    values = np.arange(100, dtype=float).reshape(-1, 1)
    valid = np.ones_like(values, dtype=bool)
    fitted = FallbackState(("s0",), (4.0,), (1,), 4.0, (2023,))
    calendar = make_calendar(
        valid[:80],
        ("s0",),
        input_start=0,
        spec=ScenarioSpec("blocks", block_duration=6, block_period=40, block_station_count=1),
        seed=20,
    )
    snapshots = list(
        replay_snapshots(
            values,
            valid,
            fitted,
            YearGrid((2024,), (100,)),
            calendar,
            target_start=1,
            target_stop=81,
            waits=(0,),
        )
    )
    block = calendar.blocks[0]
    recovered = snapshots[block["stop"]]  # target=stop+1, observes first input after the block.
    assert recovered.features.quality[0, 0] == 1
    assert recovered.features.quality[0, 6] == 0  # age
    assert recovered.features.persistence[0] == block["stop"]


@pytest.mark.parametrize(
    "change,message",
    [
        ({"validation_year": 2025}, "2025 is CLOSED"),
        ({"validation_seeds": [2101]}, "independent"),
        ({"waits": [0, 60]}, "waits"),
        ({"baseline_manifest_sha256": "bad"}, "SHA256"),
    ],
)
def test_replay_config_rejects_unsafe_scope_before_data_access(tmp_path, change, message):
    # Only config syntax is checked; no baseline pickle or HDF5 exists here.
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text("{}", encoding="utf-8")
    path = replay_config(tmp_path / "config.yaml", baseline)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config.update(change)
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(DataContractError, match=message):
        load_replay_config(path)
