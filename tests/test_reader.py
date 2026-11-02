import hashlib

import h5py
import numpy as np
import pytest

from aqstream.data import DataContractError, HDF5Reader


def read(source, **kwargs):
    request = {"year": 2023, "station_ids": ["20", "10", "30"], "chunk_size": 2}
    request.update(kwargs)
    return list(HDF5Reader(source).iter_blocks(**request))


def test_validity_grid_and_read_only(source):
    path = source / "aq_compact_2023.h5"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    blocks = read(source)
    values = np.concatenate([block.values for block in blocks])
    valid = np.concatenate([block.source_valid for block in blocks])
    expected = np.array(
        [
            [0, 20, 30],
            [np.nan, 21, 31],
            [np.nan, np.nan, -5],
            [np.nan, np.nan, np.nan],
            [5, np.nan, 35],
            [6, 26, 36],
        ]
    )
    np.testing.assert_array_equal(values, expected)
    np.testing.assert_array_equal(valid, np.isfinite(expected))
    assert values.shape == (6, 3)  # The fully empty row remains present.
    assert [(b.start, b.stop) for b in blocks] == [(0, 2), (2, 4), (4, 6)]
    with pytest.raises(ValueError):
        blocks[0].values[0, 0] = 99
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_selection_uses_ids_and_restores_requested_order(source):
    block = read(source, station_ids=["30", "20", "10"], start=0, stop=1)[0]
    assert block.station_ids == ("30", "20", "10")
    np.testing.assert_array_equal(block.values, [[30, 0, 20]])
    assert block.parameter == "pm25"


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 100])
def test_chunks_match_whole_selection(source, chunk_size):
    blocks = read(source, start=1, stop=6, chunk_size=chunk_size)
    whole = read(source, start=1, stop=6, chunk_size=100)[0]
    np.testing.assert_array_equal(np.concatenate([b.values for b in blocks]), whole.values)
    np.testing.assert_array_equal(
        np.concatenate([b.source_valid for b in blocks]), whole.source_valid
    )
    assert max(b.stop - b.start for b in blocks) <= chunk_size


@pytest.mark.parametrize("parameter", ["PM2.5", "pm25", "PM₂.₅", "pm2p5"])
def test_parameter_aliases(source, parameter):
    assert read(source, parameter=parameter, stop=1)[0].values[0, 0] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"station_ids": ["missing"]}, "unknown station IDs"),
        ({"station_ids": ["20", "20"]}, "unique IDs"),
        ({"station_ids": "20"}, "sequence"),
        ({"station_ids": []}, "nonempty"),
        ({"parameter": "ozone"}, "parameter.*not found"),
        ({"chunk_size": 0}, "positive integer"),
        ({"chunk_size": True}, "positive integer"),
        ({"start": -1}, "invalid range"),
        ({"start": 2, "stop": 2}, "invalid range"),
        ({"stop": 7}, "invalid range"),
        ({"stop": 1.5}, "invalid range"),
    ],
)
def test_bad_requests(source, kwargs, message):
    with pytest.raises(DataContractError, match=message):
        read(source, **kwargs)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_params", "params.*missing"),
        ("shape", "incompatible.*shapes"),
        ("ids_shape", "station_ids shape"),
        ("duplicate_ids", "duplicate station_ids"),
        ("duplicate_params", "duplicate normalized params"),
        ("dtype", "numeric/boolean dtype"),
        ("group", "must be a dataset"),
        ("soft_link", "required local dataset"),
    ],
)
def test_bad_schemas(source, mutation, message):
    with h5py.File(source / "aq_compact_2023.h5", "r+") as handle:
        if mutation == "missing_params":
            del handle["params"]
        elif mutation == "shape":
            del handle["masks"]
            handle["masks"] = np.zeros((2, 3, 2))
        elif mutation == "ids_shape":
            del handle["station_ids"]
            handle["station_ids"] = np.array([b"20"])
        elif mutation == "duplicate_ids":
            handle["station_ids"][:] = np.array([b"20", b"20", b"30"])
        elif mutation == "duplicate_params":
            handle["params"][:] = np.array([b"pm25", b"pm25"])
        elif mutation == "dtype":
            del handle["masks"]
            handle["masks"] = np.full((6, 3, 2), b"x")
        elif mutation == "group":
            del handle["params"]
            handle.create_group("params")
        elif mutation == "soft_link":
            del handle["params"]
            handle["params"] = h5py.SoftLink("/station_ids")
    with pytest.raises(DataContractError, match=message):
        read(source)


def test_missing_files_environment_and_closed_test_gate(tmp_path, monkeypatch):
    monkeypatch.delenv("AQ_DATA_ROOT", raising=False)
    with pytest.raises(DataContractError, match="AQ_DATA_ROOT is not set"):
        HDF5Reader()
    with pytest.raises(DataContractError, match="Data directory does not exist"):
        HDF5Reader(tmp_path / "absent")
    reader = HDF5Reader(tmp_path)
    with pytest.raises(DataContractError, match="Missing HDF5 file") as error:
        reader.inspect(2023)
    assert str(tmp_path / "aq_compact_2023.h5") in str(error.value)
    # The gate is checked before even looking for a test file.
    with pytest.raises(DataContractError, match="2025 is CLOSED"):
        reader.inspect(2025)
    monkeypatch.setenv("AQ_DATA_ROOT", str(tmp_path))
    assert HDF5Reader().data_root == tmp_path


def test_only_selected_slices_are_read(source, monkeypatch):
    original = h5py.Dataset.__getitem__
    selections = []

    def track(dataset, selection):
        if dataset.name in ("/values", "/masks"):
            selections.append((dataset.name, selection))
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", track)
    read(source, station_ids=["30"], start=1, stop=5, chunk_size=2)
    assert len(selections) == 4
    for _, (rows, columns, channel) in selections:
        assert rows.stop - rows.start <= 2
        assert columns == [2]
        assert channel == 1
