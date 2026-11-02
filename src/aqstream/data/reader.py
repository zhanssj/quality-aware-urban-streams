"""Read selected time/station/channel blocks; never expose masked source numbers."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


class DataContractError(ValueError):
    """Invalid source layout or request, with a useful path in source errors."""


def _parameter_key(value: str) -> str:
    value = value.lower().translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    key = re.sub(r"[^a-z0-9]", "", value)
    return "pm25" if key == "pm2p5" else key


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict")
    if isinstance(value, (str, int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(value)
    raise DataContractError(f"Expected a string or integer identifier; got {type(value).__name__}")


@dataclass(frozen=True)
class SourceInfo:
    path: Path
    shape: tuple[int, int, int]
    station_ids: tuple[str, ...]
    params: tuple[str, ...]
    values_dtype: str
    masks_dtype: str


@dataclass(frozen=True)
class DataBlock:
    year: int
    start: int
    stop: int
    station_ids: tuple[str, ...]
    parameter: str
    values: np.ndarray
    source_valid: np.ndarray


class HDF5Reader:
    """No download, imputation, scaler, model fitting, or full-value audit.

    Time ranges are half-open row indices within the named year. Metadata is small
    and read once per opened file; tensors are read only for the selected rows,
    stations, and parameter. All source files are opened in mode ``r``.
    """

    def __init__(self, data_root: str | Path | None = None):
        root = data_root if data_root is not None else os.environ.get("AQ_DATA_ROOT")
        if root is None or not str(root).strip():
            raise DataContractError(
                "AQ_DATA_ROOT is not set. Point it to the existing AirQualityBench directory; "
                "no files are downloaded automatically."
            )
        self.data_root = Path(root).expanduser().resolve()
        if not self.data_root.is_dir():
            raise DataContractError(f"Data directory does not exist: {self.data_root}")

    def path_for(self, year: int) -> Path:
        if type(year) is not int or year not in (2021, 2022, 2023, 2024):
            raise DataContractError(
                f"Year {year!r} is unavailable in this stage. Test gate 2025 is CLOSED; "
                "T07 requires protocol freeze and explicit owner authorization."
            )
        path = self.data_root / f"aq_compact_{year}.h5"
        if not path.is_file():
            raise DataContractError(f"Missing HDF5 file: {path}")
        return path

    @staticmethod
    def _inspect(handle: h5py.File, path: Path) -> SourceInfo:
        for name in ("values", "masks", "station_ids", "params"):
            # Reject indirection before dereferencing links; do not read external files.
            if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                raise DataContractError(f"{path}: required local dataset {name!r} is missing")
            if not isinstance(handle[name], h5py.Dataset):
                raise DataContractError(f"{path}: {name!r} must be a dataset")
            if handle[name].is_virtual or handle[name].external:
                raise DataContractError(f"{path}: external/virtual dataset {name!r} is unsupported")
        values, masks = handle["values"], handle["masks"]
        if values.ndim != 3 or masks.shape != values.shape or min(values.shape) == 0:
            raise DataContractError(
                f"{path}: incompatible values/masks shapes: {values.shape}, {masks.shape}; "
                "expected equal nonempty (time, station, parameter) tensors"
            )
        # A dtype-level guarantee avoids overflow/rounding without scanning values.
        # All float16/32/64 and <=32-bit integers are exactly representable in float64.
        safe_float = values.dtype.kind == "f" and values.dtype.itemsize in (2, 4, 8)
        safe_integer = values.dtype.kind in "iu" and values.dtype.itemsize in (1, 2, 4)
        if not (safe_float or safe_integer):
            raise DataContractError(
                f"{path}: unsupported values dtype {values.dtype}; expected float16/32/64 "
                "or int8/16/32 or uint8/16/32 for lossless float64 conversion"
            )
        if masks.dtype.kind not in "bfiu":
            raise DataContractError(f"{path}: values/masks must have real numeric/boolean dtype")
        for name, length in (("station_ids", values.shape[1]), ("params", values.shape[2])):
            if handle[name].shape != (length,):
                raise DataContractError(f"{path}: {name} shape must be ({length},)")
        try:
            ids = tuple(_text(item) for item in handle["station_ids"][:])
            params = tuple(_text(item) for item in handle["params"][:])
        except (ValueError, UnicodeError) as exc:
            raise DataContractError(f"{path}: invalid station_ids/params: {exc}") from exc
        if any(not item.strip() for item in ids) or len(set(ids)) != len(ids):
            raise DataContractError(f"{path}: empty or duplicate station_ids")
        keys = tuple(_parameter_key(param) for param in params)
        if any(not key for key in keys) or len(set(keys)) != len(keys):
            raise DataContractError(f"{path}: empty or duplicate normalized params")
        return SourceInfo(path, values.shape, ids, params, str(values.dtype), str(masks.dtype))

    def inspect(self, year: int) -> SourceInfo:
        path = self.path_for(year)
        try:
            with h5py.File(path, "r") as handle:
                return self._inspect(handle, path)
        except OSError as exc:
            raise DataContractError(f"Cannot read HDF5 file {path}: {exc}") from exc

    def iter_blocks(
        self,
        *,
        year: int,
        station_ids: Sequence[str],
        start: int = 0,
        stop: int | None = None,
        chunk_size: int = 168,
        parameter: str = "PM2.5",
    ) -> Iterator[DataBlock]:
        if type(chunk_size) is not int or chunk_size <= 0:
            raise DataContractError("chunk_size must be a positive integer")
        if isinstance(station_ids, (str, bytes)):
            raise DataContractError("station_ids must be a nonempty sequence of unique IDs")
        selected_ids = tuple(_text(item) for item in station_ids)
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise DataContractError("station_ids must be a nonempty sequence of unique IDs")
        if not isinstance(parameter, str) or not _parameter_key(parameter):
            raise DataContractError("parameter must be a nonempty pollutant name")
        path = self.path_for(year)
        try:
            with h5py.File(path, "r") as handle:
                info = self._inspect(handle, path)
                end = info.shape[0] if stop is None else stop
                if (
                    type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= info.shape[0]
                ):
                    raise DataContractError(
                        f"{path}: invalid range [{start}, {end}); expected "
                        f"0 <= start < stop <= {info.shape[0]}"
                    )
                lookup = {station: i for i, station in enumerate(info.station_ids)}
                missing = [station for station in selected_ids if station not in lookup]
                if missing:
                    raise DataContractError(f"{path}: unknown station IDs: {missing}")
                keys = [_parameter_key(param) for param in info.params]
                if _parameter_key(parameter) not in keys:
                    raise DataContractError(
                        f"{path}: parameter {parameter!r} not found; available: {info.params}"
                    )
                channel = keys.index(_parameter_key(parameter))
                # h5py fancy indexing requires strictly increasing column indices.
                requested = np.array([lookup[station] for station in selected_ids])
                order = np.argsort(requested)
                columns = requested[order].tolist()
                restore = np.argsort(order)
                for left in range(start, end, chunk_size):
                    right = min(left + chunk_size, end)
                    raw = handle["values"][left:right, columns, channel][:, restore]
                    mask = handle["masks"][left:right, columns, channel][:, restore]
                    valid = (mask == 1) & np.isfinite(raw)
                    clean = np.where(valid, raw.astype(np.float64, casting="safe"), np.nan)
                    clean.setflags(write=False)
                    valid.setflags(write=False)
                    yield DataBlock(
                        year, left, right, selected_ids, info.params[channel], clean, valid
                    )
        except OSError as exc:
            raise DataContractError(f"Cannot read HDF5 file {path}: {exc}") from exc
