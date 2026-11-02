import json

import pytest
import yaml

from aqstream.cli import main
from aqstream.config import load_prepare_config
from aqstream.data import DataContractError


def config_file(tmp_path, **updates):
    config = {
        "task_id": "T01",
        "year": 2023,
        "station_ids": ["30", "20"],
        "start": 0,
        "stop": 6,
        "chunk_size": 2,
        "parameter": "PM2.5",
    }
    config.update(updates)
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_prepare_synthetic(source, monkeypatch, capsys):
    config = config_file(source)
    artifacts = source / "outputs"
    monkeypatch.setenv("AQ_DATA_ROOT", str(source))
    monkeypatch.setenv("AQ_ARTIFACT_ROOT", str(artifacts))
    assert main(["prepare", "--config", str(config)]) == 0
    report = json.loads(next(artifacts.glob("*/manifest.json")).read_text(encoding="utf-8"))
    assert report["output_shape"] == [6, 2]
    assert report["available_elements"] == 8
    assert report["blocks_read"] == 3
    assert report["test_gate"] == "CLOSED"
    assert report["model_training"] == "NOT_RUN"
    assert report["input"]["sha256_verified_in_this_run"] is False
    assert report["selection"]["station_ids"] == ["30", "20"]
    assert "PASS bounded loader smoke" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"year": 2025}, "training years"),
        ({"year": 2024}, "training years"),
        ({"stop": 169}, "1–168"),
        ({"start": -1}, "start < stop"),
        ({"chunk_size": 0}, "between 1 and 168"),
        ({"stop": True}, "must be an integer"),
        ({"station_ids": ["20"]}, "2–4"),
        ({"station_ids": ["20", "20"]}, "duplicate"),
        ({"station_ids": [True, "20"]}, "strings or integers"),
        ({"task_id": "T03"}, "only T01 prepare"),
        ({"unexpected": 3}, "exactly these keys"),
    ],
)
def test_config_restrictions(tmp_path, updates, message):
    path = config_file(tmp_path, **updates)
    with pytest.raises(DataContractError, match=message):
        load_prepare_config(path)


def test_cli_missing_environment_is_error(source, monkeypatch, capsys):
    monkeypatch.delenv("AQ_DATA_ROOT", raising=False)
    assert main(["prepare", "--config", str(config_file(source))]) == 2
    captured = capsys.readouterr()
    assert "AQ_DATA_ROOT" in captured.err
    assert "PASS" not in captured.out


@pytest.mark.parametrize("command", ["train", "replay", "evaluate", "benchmark", "export-review"])
def test_unimplemented_commands_fail(command):
    with pytest.raises(SystemExit) as error:
        main([command])
    assert error.value.code == 2


def test_bad_config_is_clear_error(tmp_path, capsys):
    missing = tmp_path / "absent.yaml"
    assert main(["prepare", "--config", str(missing)]) == 2
    assert str(missing) in capsys.readouterr().err
    invalid = tmp_path / "bad.yaml"
    invalid.write_text("[", encoding="utf-8")
    assert main(["prepare", "--config", str(invalid)]) == 2
    assert "Cannot load config" in capsys.readouterr().err
