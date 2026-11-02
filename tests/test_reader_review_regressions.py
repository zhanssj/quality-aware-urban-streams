"""Independent T01 review checks. Use only temporary synthetic data.

Run from the project root on its supported environment:
    python -m pytest /path/to/test_reader_review.py -q
Two conversion regression tests intentionally fail on commit 103a6ef.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from aqstream.data import DataContractError, HDF5Reader


def write_source(root: Path, values: np.ndarray, masks=None, ids=None):
    path = root / "aq_compact_2023.h5"
    with h5py.File(path, "w") as f:
        f["values"] = values
        f["masks"] = np.ones(values.shape, dtype=np.uint8) if masks is None else masks
        f["station_ids"] = np.asarray(ids or [f"s{i}" for i in range(values.shape[1])], dtype="S")
        f["params"] = np.asarray([b"pm25", b"pm10"][: values.shape[2]])
    return path


@pytest.mark.parametrize("seed", range(10))
def test_random_subsets_match_independent_reference(tmp_path, seed):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(23, 7, 2)).astype(np.float32)
    masks = rng.integers(0, 3, size=values.shape, dtype=np.uint8)
    values[2, 1, 0] = np.nan
    values[3, 2, 0] = np.inf
    values[4, 3, 0] = 0
    masks[4, 3, 0] = 1
    selected = rng.choice(7, 4, replace=False)
    ids = [f"s{i}" for i in selected]
    write_source(tmp_path, values, masks)
    blocks = list(
        HDF5Reader(tmp_path).iter_blocks(year=2023, station_ids=ids, start=1, stop=21, chunk_size=3)
    )
    raw = values[1:21, :, 0][:, selected]
    valid = (masks[1:21, :, 0][:, selected] == 1) & np.isfinite(raw)
    np.testing.assert_array_equal(np.concatenate([b.source_valid for b in blocks]), valid)
    np.testing.assert_array_equal(
        np.concatenate([b.values for b in blocks]), np.where(valid, raw.astype(np.float64), np.nan)
    )
    assert all(b.station_ids == tuple(ids) for b in blocks)


def test_external_link_rejected(tmp_path):
    path = write_source(tmp_path, np.ones((2, 1, 1), dtype=np.float32))
    with h5py.File(path, "r+") as f:
        del f["values"]
        f["values"] = h5py.ExternalLink(str(tmp_path / "absent.h5"), "/values")
    with pytest.raises(DataContractError):
        HDF5Reader(tmp_path).inspect(2023)


def test_virtual_dataset_rejected(tmp_path):
    path = write_source(tmp_path, np.ones((2, 1, 1), dtype=np.float32))
    with h5py.File(path, "r+") as f:
        del f["values"]
        layout = h5py.VirtualLayout(shape=(2, 1, 1), dtype="f4")
        layout[:] = h5py.VirtualSource(str(tmp_path / "absent.h5"), "values", shape=(2, 1, 1))
        f.create_virtual_dataset("values", layout)
    with pytest.raises(DataContractError, match="external/virtual"):
        HDF5Reader(tmp_path).inspect(2023)


def test_external_storage_rejected(tmp_path):
    path = write_source(tmp_path, np.ones((2, 1, 1), dtype=np.float32))
    with h5py.File(path, "r+") as f:
        del f["values"]
        f.create_dataset(
            "values",
            shape=(2, 1, 1),
            dtype="f4",
            external=[(str(tmp_path / "raw.bin"), 0, h5py.h5f.UNLIMITED)],
        )
    with pytest.raises(DataContractError, match="external/virtual"):
        HDF5Reader(tmp_path).inspect(2023)


def test_2025_gate_precedes_hdf5_open(tmp_path, monkeypatch):
    def forbidden_open(*args, **kwargs):
        raise AssertionError("h5py.File must not be called for 2025")

    monkeypatch.setattr(h5py, "File", forbidden_open)
    with pytest.raises(DataContractError, match="2025 is CLOSED"):
        list(HDF5Reader(tmp_path).iter_blocks(year=2025, station_ids=["s0"]))


def test_corrupt_hdf5_has_contract_error(tmp_path):
    (tmp_path / "aq_compact_2023.h5").write_bytes(b"not HDF5")
    with pytest.raises(DataContractError, match="Cannot read HDF5"):
        HDF5Reader(tmp_path).inspect(2023)


def test_single_step_channel_and_station(tmp_path):
    write_source(tmp_path, np.array([[[0.0]]], dtype=np.float32))
    b = next(HDF5Reader(tmp_path).iter_blocks(year=2023, station_ids=["s0"], chunk_size=1))
    assert b.values.shape == (1, 1)
    assert b.source_valid[0, 0] and b.values[0, 0] == 0
    assert (b.start, b.stop) == (0, 1)


def test_finite_extended_float_must_not_become_valid_infinity(tmp_path):
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        pytest.skip("Platform has no wider-than-float64 longdouble")
    values = np.full((1, 1, 1), np.longdouble("1e400"), dtype=np.longdouble)
    try:
        write_source(tmp_path, values)
    except (TypeError, ValueError) as exc:
        pytest.skip(f"This HDF5 build does not support extended floats: {exc}")
    try:
        b = next(HDF5Reader(tmp_path).iter_blocks(year=2023, station_ids=["s0"]))
    except DataContractError:
        return  # Explicitly rejecting unsupported numeric data is a valid policy.
    assert np.isfinite(b.values[b.source_valid]).all(), (
        "Finite source value became inf while source_valid remained True"
    )


def test_large_integer_must_not_silently_change_value(tmp_path):
    value = 2**53 + 1
    write_source(tmp_path, np.full((1, 1, 1), value, dtype=np.uint64))
    try:
        b = next(HDF5Reader(tmp_path).iter_blocks(year=2023, station_ids=["s0"]))
    except DataContractError:
        return  # Rejecting integers that cannot be represented exactly is acceptable.
    assert b.source_valid[0, 0]
    assert int(b.values[0, 0]) == value, "Accepted source integer was silently rounded"
