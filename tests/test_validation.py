import json

import numpy as np
import pytest
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from test_experiment import experiment_inputs as experiment_inputs

from aqstream.artifacts import file_sha256
from aqstream.cli import REPOSITORY, main
from aqstream.data import DataContractError, HDF5Reader
from aqstream.experiment import run_experiment
from aqstream.prediction_plan import PredictionPlan
from aqstream.replay.calendars import ScenarioSpec, make_calendar
from aqstream.validation.benchmark import run_benchmark
from aqstream.validation.config import load_config
from aqstream.validation.metrics import (
    block_bootstrap,
    evaluate_npz,
    save_predictions,
    score_arrays,
    select_regime,
    slice_labels,
)
from aqstream.validation.models import MODEL_NAMES, predictors
from aqstream.validation.runner import run_validation

BOOTSTRAP = {"block_steps": 2, "replicates": 100, "seed": 5101}


def test_feature_ablations_and_native_control_keep_common_and_hide_unavailable_values():
    common = np.arange(20, dtype=float).reshape(2, 10)
    quality = np.ones((2, 11))
    quality[:, 6] = [0, 19]
    quality[:, 7:9] = 0
    assert predictors(common, quality, "masks").shape == (2, 16)
    age = predictors(common, quality, "age")
    np.testing.assert_array_equal(age[:, :10], common)
    np.testing.assert_array_equal(age[:, -1], [0, 19])
    np.testing.assert_array_equal(predictors(common, quality, "native_missing"), common)
    quality[0, 0] = 0
    quality[0, 7] = 1 / 6
    native = predictors(common, quality, "native_missing")
    assert np.isnan(native[0, 0]) and np.isnan(native[0, 6:8]).all()
    np.testing.assert_array_equal(native[0, 8:], common[0, 8:])
    np.testing.assert_array_equal(common, np.arange(20).reshape(2, 10))


def test_bootstrap_pairs_stations_and_weights_partial_blocks():
    delta = np.array([[1.0, 1.0], [1.0, 1.0], [10.0, 10.0]])
    valid = np.ones_like(delta, dtype=bool)
    result = block_bootstrap(delta, valid, **BOOTSTRAP)
    assert result["delta_mae"] == 4
    assert result["ci95"] == [1, 10]
    assert result["blocks"] == 2 and not result["stations_resampled_independently"]
    np.testing.assert_array_equal(
        result["ci95"], block_bootstrap(delta, valid, **BOOTSTRAP)["ci95"]
    )
    paired = np.tile([10.0, -10.0], (8, 1))
    assert block_bootstrap(paired, np.ones_like(paired, dtype=bool), **BOOTSTRAP)["ci95"] == [0, 0]
    assert block_bootstrap(delta, ~valid, **BOOTSTRAP)["delta_mae"] is None


def tensor_fixture(tmp_path):
    plan = PredictionPlan(("a", "b"), 10, 12, 0, "run", "scenario", 5)
    plan.save(tmp_path / "evaluation_plan.json")
    truth = np.array([[1.0, np.nan], [3.0, 4.0]])
    valid = np.isfinite(truth)
    predictions = np.zeros((2, 2, len(MODEL_NAMES)))
    predictions[:] = np.nan_to_num(truth)[:, :, None] + 2
    labels = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    save_predictions(tmp_path / "predictions.npz", predictions, plan, labels)
    np.savez(
        tmp_path / "truth.npz",
        values=truth,
        valid=valid,
        station_ids=np.array(plan.station_ids),
        target_indices=[10, 11],
    )
    return plan, predictions, truth, valid, labels


def test_tensor_hand_metrics_shared_missing_predictions_slices_and_missing_truth(tmp_path):
    plan, predictions, truth, valid, labels = tensor_fixture(tmp_path)
    predictions[0, 0, 1] = np.nan
    result = score_arrays(predictions, truth, valid, plan, labels, bootstrap=BOOTSTRAP)
    assert result["planned"] == result["present_rows"] == 4
    assert all(m["scored"] == 2 and m["eligible_truth"] == 3 for m in result["models"].values())
    assert result["models"]["clean/standard"]["prediction_coverage"] == 0.75
    for method in result["models"].values():
        assert method["mae"] == method["rmse"] == method["macro_station_mae"] == 2
        assert sum(s["scheduled"] for s in method["slices"].values()) == 4
        assert sum(s["scored"] for s in method["slices"].values()) == 2
    save_predictions(tmp_path / "predictions.npz", predictions, plan, labels)
    assert (
        evaluate_npz(tmp_path / "predictions.npz", tmp_path / "truth.npz", bootstrap=BOOTSTRAP)
        == result
    )


def test_invalid_truth_is_not_scored_or_filled_and_empty_selection_is_rejected(tmp_path):
    plan, predictions, truth, valid, labels = tensor_fixture(tmp_path)
    truth[0, 0] = np.inf
    with pytest.raises(DataContractError, match="truth"):
        score_arrays(predictions, truth, valid, plan, labels, bootstrap=BOOTSTRAP)
    with pytest.raises(DataContractError, match="scored"):
        select_regime([])


def test_benchmark_changed_model_is_rejected_before_unpickle(tmp_path, monkeypatch):
    import joblib

    (tmp_path / "models.joblib").write_bytes(b"changed")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "task_id": "T06",
                "test_gate": "CLOSED",
                "artifacts": {"models.joblib": {"sha256": "0" * 64}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("changed model must not load"))
    with pytest.raises(DataContractError, match="Changed trusted"):
        run_benchmark(tmp_path, file_sha256(tmp_path / "manifest.json"), tmp_path / "out")


@pytest.mark.parametrize(
    "tamper",
    ["row", "station", "target", "duplicate_target", "model", "run", "scenario", "wait", "hash"],
)
def test_tensor_expected_plan_rejects_incomplete_replaced_or_mixed_data(tmp_path, tamper):
    tensor_fixture(tmp_path)
    path = tmp_path / "predictions.npz"
    with np.load(path, allow_pickle=False) as data:
        payload = dict(data)
    if tamper == "row":
        payload["predictions"] = payload["predictions"][:-1]
    elif tamper == "station":
        payload["station_ids"] = np.array(["a", "z"])
    elif tamper in ("target", "duplicate_target"):
        payload["target_indices"] = [10, 12 if tamper == "target" else 10]
    elif tamper == "model":
        payload["model_names"] = payload["model_names"][::-1]
    elif tamper == "wait":
        payload["wait_minutes"] = 15
    else:
        payload[{"run": "run_id", "scenario": "scenario_id", "hash": "plan_sha256"}[tamper]] = (
            "wrong"
        )
    np.savez(path, **payload)
    with pytest.raises(DataContractError):
        evaluate_npz(path, tmp_path / "truth.npz", bootstrap=BOOTSTRAP)


def test_slice_partition_uses_only_past_disruptions_and_fixed_recovery_window():
    valid = np.ones((12, 2), dtype=bool)
    calendar = make_calendar(
        valid, ("a", "b"), input_start=0, spec=ScenarioSpec("drop", random_drop_rate=0.2), seed=21
    )
    labels = slice_labels(calendar, 0, 2)
    for step in range(12):
        for column in range(2):
            disrupted = calendar.dropped[:, column]
            expected = (
                1 if disrupted[step] else 2 if disrupted[max(0, step - 2) : step].any() else 0
            )
            assert labels[step, column] == expected
    assert set(np.unique(labels)) <= {0, 1, 2}


def test_regime_selection_weights_scenarios_equally_not_number_of_policies():
    def case(scenario, augmented):
        return {
            "scenario_name": scenario,
            "metrics": {
                "models": {
                    f"{regime}/{name}": {"mae": value}
                    for regime, value in (("clean", 3), ("augmented", augmented))
                    for name in ("standard", "quality_aware")
                }
            },
        }

    selection = select_regime([*[case("many", 0) for _ in range(5)], case("one", 10)])
    assert selection["selected_regime"] == "clean"
    assert selection["scores"] == {"clean": 3, "augmented": 5}


def validation_config(path, baseline):
    config = yaml.safe_load((REPOSITORY / "configs/t06.yaml").read_text(encoding="utf-8"))
    config.update(
        baseline_run=str(baseline),
        baseline_manifest_sha256=file_sha256(baseline / "manifest.json"),
        validation_steps=15,
        pilot_steps=10,
        scenarios=["clean_source", "random_drop10", "delay30"],
        train_stride=2,
        batch_steps=4,
        threads=1,
        bootstrap_block_steps=3,
        bootstrap_replicates=20,
        benchmark_steps=4,
        benchmark_warmup=1,
        benchmark_repeats=2,
    )
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_validation_end_to_end_train_only_paired_cases_frozen_baseline_and_cli(
    experiment_inputs, monkeypatch
):
    baseline = run_experiment(experiment_inputs)
    original_hash = file_sha256(baseline / "models.joblib")
    config = validation_config(experiment_inputs.parent / "t06.yaml", baseline)
    opened = []
    real_reader = HDF5Reader.iter_blocks
    real_fit = HistGradientBoostingRegressor.fit
    fits = []

    def reader(self, **kwargs):
        opened.append(kwargs["year"])
        assert kwargs["year"] != 2025
        return real_reader(self, **kwargs)

    def fit(self, features, labels, **kwargs):
        assert 2024 not in opened
        assert np.max(labels) < 100
        assert self.early_stopping is False
        fits.append(len(labels))
        return real_fit(self, features, labels, **kwargs)

    monkeypatch.setattr(HDF5Reader, "iter_blocks", reader)
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", fit)
    output = run_validation(config)
    assert len(fits) == 8  # 3 extra clean ablations and 5 augmented estimators.
    assert opened == [2023, 2023, 2024]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["test_gate"] == "CLOSED"
    assert manifest["pilot_cases"] == 15 and manifest["final_cases"] == 23
    assert manifest["zero_control"]["final"]["compared_per_method"] == 30
    assert all(
        x == 0 for x in manifest["zero_control"]["final"]["max_absolute_difference"].values()
    )
    cases = json.loads((output / "final_metrics.json").read_text(encoding="utf-8"))
    assert len({c["metrics"]["scored_target_ids_sha256"] for c in cases}) == 1
    assert all(m["scored"] == 20 for c in cases for m in c["metrics"]["models"].values())
    assert file_sha256(baseline / "models.joblib") == original_hash
    plan = json.loads((output / "fit_plan.json").read_text(encoding="utf-8"))
    assert plan["total_weight"] == pytest.approx(51)
    assert plan["augmented_rows"] == 510 and len(plan["views"]) == 10
    freeze = json.loads((output / "protocol_freeze.json").read_text(encoding="utf-8"))
    assert freeze["owner_test_authorization"] is None and not freeze["units_verified"]
    assert freeze["model_bank_sha256"] == file_sha256(output / "models.joblib")
    case = output / "final_cases/clean_source_seed0_wait0"
    recomputed = output.parent / "recomputed.json"
    assert (
        main(
            [
                "evaluate-tensor",
                "--predictions",
                str(case / "predictions.npz"),
                "--truth",
                str(output / "truth.npz"),
                "--config",
                str(config),
                "--out",
                str(recomputed),
            ]
        )
        == 0
    )
    assert json.loads(recomputed.read_text(encoding="utf-8")) == json.loads(
        (case / "metrics.json").read_text(encoding="utf-8")
    )
    benchmark = output.parent / "extra_benchmark"
    assert (
        main(
            [
                "benchmark",
                "--run",
                str(output),
                "--manifest-sha256",
                file_sha256(output / "manifest.json"),
                "--out",
                str(benchmark),
            ]
        )
        == 0
    )
    result = json.loads((benchmark / "benchmark.json").read_text(encoding="utf-8"))
    assert len(result["warmups"]) == 1 and len(result["measured"]) == 2
    assert result["observations_per_repeat"] == 8
    assert all(r["samples_per_second"] == 8 / r["job_seconds"] for r in result["measured"])


@pytest.mark.parametrize(
    "change",
    [
        {"validation_year": 2025},
        {"validation_seeds": [2101, 4102, 4103, 4104, 4105]},
        {"waits": [0, 60]},
        {"model_seed": True},
        {"training_mixture_weights": {"clean_source": 1}},
    ],
)
def test_config_rejects_unsafe_scope_without_source_access(tmp_path, change):
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (baseline / "manifest.json").write_text("{}", encoding="utf-8")
    path = validation_config(tmp_path / "config.yaml", baseline)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.update(change)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(DataContractError):
        load_config(path)
