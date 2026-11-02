import csv

import pytest

from aqstream.data import DataContractError
from aqstream.evaluation import evaluate_predictions
from aqstream.prediction_plan import PredictionPlan


def write_rows(tmp_path, mutation=None):
    rows = [
        {
            "sample_id": f"s0:{i - 1}:{i}",
            "station_id": "s0",
            "origin_index": i - 1,
            "target_index": i,
            "target_year": 2024,
            "target_local_index": i,
            "split": "validation",
            "run_id": "run",
            "scenario_id": "clean",
            "wait_minutes": 0,
            "source_valid": 1,
            "target": i + 1.0,
            "persistence": 0.0,
            "standard": 0.0,
            "quality_aware": 0.0,
        }
        for i in range(3)
    ]
    if mutation:
        mutation(rows)
    path = tmp_path / "predictions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return path


def plan():
    return PredictionPlan(("s0",), 0, 3, 0, "run", "clean", 0)


def test_complete_expected_schedule_and_coverage(tmp_path):
    result = evaluate_predictions(write_rows(tmp_path), plan=plan())
    assert result["schedule_validated"] and result["planned"] == result["present_rows"] == 3
    assert result["models"]["persistence"]["mae"] == 2
    assert result["models"]["persistence"]["prediction_coverage"] == 1


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda rows: rows.pop(), "Incomplete expected schedule"),
        (lambda rows: rows.append(rows[0].copy()), "Duplicate"),
        (
            lambda rows: rows[0].update(station_id="other", sample_id="other:-1:0"),
            "Unexpected sample",
        ),
        (lambda rows: rows[0].update(run_id="other"), "run/scenario/wait"),
        (lambda rows: rows[0].update(scenario_id="other"), "run/scenario/wait"),
        (lambda rows: rows[0].update(wait_minutes=5), "run/scenario/wait"),
        (lambda rows: rows[0].update(target_local_index=8), "local index"),
        (
            lambda rows: rows[0].update(target_index=3, origin_index=2, sample_id="s0:2:3"),
            "Unexpected sample",
        ),
    ],
)
def test_missing_replaced_duplicate_and_mixed_rows_rejected(tmp_path, mutation, message):
    with pytest.raises(DataContractError, match=message):
        evaluate_predictions(write_rows(tmp_path, mutation), plan=plan())


def test_csv_without_expected_plan_is_not_validated(tmp_path):
    with pytest.raises(DataContractError, match="expected prediction plan"):
        evaluate_predictions(write_rows(tmp_path))


def test_finite_prediction_coverage_uses_planned_count(tmp_path):
    result = evaluate_predictions(
        write_rows(tmp_path, lambda rows: rows[0].update(standard="")), plan=plan()
    )
    assert result["models"]["standard"]["prediction_coverage"] == pytest.approx(2 / 3)
    assert all(value["scored"] == 2 for value in result["models"].values())


@pytest.mark.parametrize(
    "field", ["origin_minutes", "cutoff_minutes", "target_end_minutes", "remaining_lead_minutes"]
)
def test_inconsistent_clock_metadata_rejected(tmp_path, field):
    with pytest.raises(DataContractError, match="clock differs"):
        evaluate_predictions(
            write_rows(tmp_path, lambda rows: rows[0].update({field: -99})), plan=plan()
        )
