from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def source(tmp_path: Path) -> Path:
    values = np.zeros((6, 3, 2), dtype=np.float32)
    values[:, :, 0] = 900  # PM10 must never be mistaken for PM2.5.
    values[:, :, 1] = [
        [0, 20, 30],
        [123456, 21, 31],
        [np.nan, np.inf, -5],
        [4, 24, 34],
        [5, 25, 35],
        [6, 26, 36],
    ]
    masks = np.ones_like(values)
    masks[1, 0, 1] = 0
    masks[3, :, 1] = 0
    masks[4, 1, 1] = 2  # Only exactly 1 is source-valid.
    with h5py.File(tmp_path / "aq_compact_2023.h5", "w") as handle:
        handle["values"] = values
        handle["masks"] = masks
        handle["station_ids"] = np.array([b"20", b"10", b"30"])
        handle["params"] = np.array([b"pm10", b"pm25"])
    return tmp_path
