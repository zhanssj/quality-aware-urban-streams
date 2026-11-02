import h5py
import numpy as np
import pytest

from aqstream.data import DataContractError, HDF5Reader


def write_values(root, values):
    path = root / "aq_compact_2023.h5"
    with h5py.File(path, "w") as handle:
        handle["values"] = values.reshape(-1, 1, 1)
        handle["masks"] = np.ones((values.size, 1, 1), dtype=np.uint8)
        handle["station_ids"] = [b"s0"]
        handle["params"] = [b"pm25"]
    return path


@pytest.mark.parametrize(
    "dtype_name",
    [
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "uint8",
        "uint16",
        "uint32",
        ">f4",
        ">f8",
        ">i4",
        ">u4",
    ],
)
def test_supported_dtype_extremes_are_exact_and_finite(tmp_path, dtype_name):
    dtype = np.dtype(dtype_name)
    limits = np.finfo(dtype) if dtype.kind == "f" else np.iinfo(dtype)
    source = np.array([limits.min, 0, limits.max], dtype=dtype)
    write_values(tmp_path, source)

    blocks = list(
        HDF5Reader(tmp_path).iter_blocks(
            year=2023,
            station_ids=["s0"],
            chunk_size=2,
        )
    )
    values = np.concatenate([block.values[:, 0] for block in blocks])
    assert values.dtype == np.float64
    assert np.isfinite(values).all()
    assert all(block.source_valid.all() for block in blocks)
    np.testing.assert_array_equal(values.astype(dtype), source)
    if dtype.kind in "iu":
        assert [int(value) for value in values] == [int(value) for value in source]


@pytest.mark.parametrize("dtype_name", ["int64", "uint64"])
@pytest.mark.parametrize("value", [1, 2**53 + 1])
def test_unsupported_integer_dtype_rejected_before_numeric_reads(
    tmp_path,
    monkeypatch,
    dtype_name,
    value,
):
    path = write_values(tmp_path, np.array([value], dtype=dtype_name))
    original = h5py.Dataset.__getitem__

    def forbid_numeric_read(dataset, selection):
        if dataset.name in ("/values", "/masks"):
            pytest.fail("Unsupported values dtype must be rejected before reading numeric blocks")
        return original(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", forbid_numeric_read)
    reader = HDF5Reader(tmp_path)
    with pytest.raises(DataContractError, match=f"unsupported values dtype {dtype_name}") as error:
        reader.inspect(2023)
    assert str(path) in str(error.value)
    with pytest.raises(DataContractError, match=f"unsupported values dtype {dtype_name}") as error:
        list(reader.iter_blocks(year=2023, station_ids=["s0"]))
    assert str(path) in str(error.value)
