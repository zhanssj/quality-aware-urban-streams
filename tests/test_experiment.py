import csv
import json
import zipfile

import h5py
import numpy as np
import pytest
import yaml

from aqstream.artifacts import export_review
from aqstream.baseline_config import load_baseline_config
from aqstream.cli import main
from aqstream.data import DataContractError, HDF5Reader
from aqstream.evaluation import evaluate_predictions
from aqstream.experiment import choose_cohort, run_experiment
from aqstream.models import Baselines
from aqstream.prediction_plan import PredictionPlan


@pytest.fixture
def experiment_inputs(tmp_path, monkeypatch):
    for year, length in ((2023, 72), (2024, 15)):
        values = 10 + np.sin(np.arange(length)[:, None] / 3) + np.arange(2)[None, :] * 3
        values = values[:, :, None]
        mask = np.ones_like(values, dtype=np.uint8)
        mask[2::4, 0] = 0
        mask[5:11, 1] = 0
        values[mask == 0] = 999999
        with h5py.File(tmp_path / f"aq_compact_{year}.h5", "w") as handle:
            handle["values"], handle["masks"] = values, mask
            handle["station_ids"] = [b"s0", b"s1"]
            handle["params"] = [b"pm25"]
    cohort = tmp_path / "candidates.csv"
    cohort.write_text("station_id\ns0\ns1\n", encoding="utf-8")
    config = {
        "task_id": "B01",
        "train_years": [2023],
        "validation_year": 2024,
        "cohort_candidates": str(cohort),
        "cohort_size": 2,
        "seed": 2026,
        "train_stride": 2,
        "max_train_samples": 100,
        "chunk_size": 7,
        "predict_batch_steps": 4,
        "threads": 1,
        "max_rss_mb": 2048,
        "model_params": {
            "max_iter": 6,
            "max_leaf_nodes": 5,
            "min_samples_leaf": 2,
            "learning_rate": 0.1,
            "l2_regularization": 1.0,
        },
    }
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("AQ_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("AQ_ARTIFACT_ROOT", str(tmp_path / "outputs"))
    return config_path


def test_synthetic_end_to_end_reproducible_and_evaluable(experiment_inputs, monkeypatch):
    opened_years = []
    original = HDF5Reader.iter_blocks
    fit = Baselines.fit

    def track(reader, **kwargs):
        opened_years.append(kwargs["year"])
        return original(reader, **kwargs)

    def fit_train_only(models, common, quality, labels):
        assert 2024 not in opened_years
        assert np.max(labels) < 100  # Hidden source numbers do not enter training labels.
        return fit(models, common, quality, labels)

    monkeypatch.setattr(HDF5Reader, "iter_blocks", track)
    monkeypatch.setattr(Baselines, "fit", fit_train_only)
    first = run_experiment(experiment_inputs)
    opened_years.clear()
    second = run_experiment(experiment_inputs)
    first_metrics = json.loads((first / "metrics.json").read_text(encoding="utf-8"))
    assert first_metrics == evaluate_predictions(first / "predictions_validation.csv")
    second_metrics = json.loads((second / "metrics.json").read_text(encoding="utf-8"))
    # Runtime identity differs; every numerical metric/count remains exactly reproducible.
    assert {
        k: v for k, v in first_metrics.items() if k not in ("context", "expected_plan_sha256")
    } == {k: v for k, v in second_metrics.items() if k not in ("context", "expected_plan_sha256")}
    assert first_metrics["context"]["run_id"] != second_metrics["context"]["run_id"]
    with (first / "predictions_validation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 30
    assert any(row["source_valid"] == "0" and row["target"] == "" for row in rows)
    assert all(
        all(row[name] for name in ("persistence", "standard", "quality_aware")) for row in rows
    )
    assert rows[0]["origin_index"] == "71" and rows[0]["target_index"] == "72"
    assert (
        len({value["scored_target_ids_sha256"] for value in first_metrics["models"].values()}) == 1
    )
    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PASS" and manifest["test_gate"] == "CLOSED"
    assert manifest["train_scheduled"] == 72
    assert manifest["train_scored_for_fit"] == 51
    assert all(
        item["scored"] == 20 and item["skipped_missing_truth"] == 10
        for item in first_metrics["models"].values()
    )
    assert float(rows[0]["persistence"]) == pytest.approx(10 + np.sin(71 / 3))
    assert manifest["model_params"]["early_stopping"] is False
    assert manifest["peak_rss_sampled_bytes"] > 0
    assert (first / "models.joblib").is_file()
    assert (first / "train_inputs.npz").is_file()
    assert (
        main(
            [
                "evaluate",
                "--predictions",
                str(first / "predictions_validation.csv"),
                "--out",
                str(first / "recomputed.json"),
            ]
        )
        == 0
    )
    assert json.loads((first / "recomputed.json").read_text(encoding="utf-8")) == first_metrics


def test_train_cli_and_test_gate(experiment_inputs, tmp_path, monkeypatch):
    assert main(["train", "--config", str(experiment_inputs)]) == 0
    config = yaml.safe_load(experiment_inputs.read_text(encoding="utf-8"))
    config["validation_year"] = 2025
    experiment_inputs.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(h5py, "File", lambda *a, **kw: pytest.fail("test must not open HDF5"))
    with pytest.raises(DataContractError, match="2025 is CLOSED"):
        load_baseline_config(experiment_inputs)
    assert main(["train", "--config", str(experiment_inputs)]) == 2


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.1"])
def test_model_config_rejects_nonfinite_or_nonnumeric(experiment_inputs, value):
    config = yaml.safe_load(experiment_inputs.read_text(encoding="utf-8"))
    config["model_params"]["learning_rate"] = value
    experiment_inputs.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(DataContractError, match="finite number"):
        load_baseline_config(experiment_inputs)


def test_predeclared_training_cap_stops_without_resampling(experiment_inputs):
    config = yaml.safe_load(experiment_inputs.read_text(encoding="utf-8"))
    config["max_train_samples"] = 1
    experiment_inputs.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(DataContractError, match="exceeds predeclared"):
        run_experiment(experiment_inputs)


def predictions_fixture(tmp_path):
    path = tmp_path / "predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "station_id",
                "origin_index",
                "target_index",
                "target_year",
                "split",
                "source_valid",
                "target",
                "persistence",
                "standard",
                "quality_aware",
                "target_local_index",
                "run_id",
                "scenario_id",
                "wait_minutes",
            ],
        )
        writer.writeheader()
        for index, truth, valid, persistence, standard, quality in (
            (0, 1, 1, 2, 1, 0),
            (1, "", 0, 999, 999, 999),
            (2, 3, 1, 1, 3, 5),
        ):
            writer.writerow(
                {
                    "sample_id": f"s0:{index - 1}:{index}",
                    "station_id": "s0",
                    "origin_index": index - 1,
                    "target_index": index,
                    "target_year": 2024,
                    "split": "validation",
                    "source_valid": valid,
                    "target": truth,
                    "persistence": persistence,
                    "standard": standard,
                    "quality_aware": quality,
                    "target_local_index": index,
                    "run_id": "fixture",
                    "scenario_id": "natural_missingness",
                    "wait_minutes": 0,
                }
            )
    PredictionPlan(("s0",), 0, 3, 0, "fixture", "natural_missingness", 0).save(
        tmp_path / "evaluation_plan.json"
    )
    return path


def test_hand_metrics_ignore_missing_truth_and_share_targets(tmp_path):
    path = predictions_fixture(tmp_path)
    metrics = evaluate_predictions(path)["models"]
    assert metrics["persistence"]["mae"] == 1.5
    assert metrics["persistence"]["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["standard"]["mae"] == 0
    assert metrics["quality_aware"]["mae"] == 1.5
    assert all(row["scheduled"] == 3 and row["scored"] == 2 for row in metrics.values())
    text = path.read_text(encoding="utf-8").replace(
        "s0:1:2,s0,1,2,2024,validation,1,3,1,3,5",
        "s0:1:2,s0,1,2,2024,validation,1,3,1,,5",
    )
    path.write_text(text, encoding="utf-8")
    metrics = evaluate_predictions(path)["models"]
    assert all(row["scored"] == 1 for row in metrics.values())
    assert metrics["standard"]["missing_predictions"] == 1
    assert metrics["standard"]["skipped_missing_any_prediction"] == 1


def test_cohort_deterministic_from_ids_only(tmp_path):
    path = tmp_path / "cohort.csv"
    path.write_text("station_id,coverage_2024\ns2,1\ns0,0\ns1,0\n", encoding="utf-8")
    first = choose_cohort(path, 2, 2026)
    path.write_text("station_id,coverage_2024\ns1,1\ns2,0\ns0,1\n", encoding="utf-8")
    assert choose_cohort(path, 2, 2026) == first


def test_review_export_whitelist_hashes_and_secret_rejection(tmp_path):
    (tmp_path / "src/aqstream").mkdir(parents=True)
    (tmp_path / "src/aqstream/safe.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts/models.joblib").write_bytes(b"model")
    (tmp_path / "artifacts/aq_compact_2025.h5").write_bytes(b"not actual HDF5")
    (tmp_path / ".env").write_text("secret", encoding="utf-8")
    output = tmp_path / "review.zip"
    export_review(tmp_path, output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"src/aqstream/safe.py", "bundle_sha256.json"}
    (tmp_path / "src/aqstream/safe.py").write_text("ghp_" + "x" * 30, encoding="utf-8")
    with pytest.raises(DataContractError, match="Possible secret"):
        export_review(tmp_path, tmp_path / "rejected.zip")
