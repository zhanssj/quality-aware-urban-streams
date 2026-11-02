"""Synthetic tests only. These are NOT findings about the real dataset."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from audit_data import audit_one, jsonable


class AuditTests(unittest.TestCase):
    def make_file(self, path):
        values = np.full((12, 3, 2), 10.0, dtype=np.float32)
        masks = np.ones_like(values, dtype=np.uint8)
        masks[:, 0, 0] = [1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
        masks[:, 1, 0] = 0
        values[4, 2, 0] = np.nan
        values[8, 2, 0] = -1
        values[9, 2, 0] = 0  # A real zero must remain usable.
        masks[2, 0, 1] = 2  # Invalid mask is not silently cast to True.
        with h5py.File(path, "w") as f:
            f.create_dataset("values", data=values)
            f.create_dataset("masks", data=masks)
            f.create_dataset("station_ids", data=np.array([b"1", b"2", b"3"]))
            f.create_dataset("params", data=np.array([b"pm25", b"no2"]))
            f.attrs["synthetic_test_only"] = True
        return values, masks

    def test_scan_and_gap_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aq_compact_2021.h5"
            values, masks = self.make_file(path)
            summary, pollutants, stations = audit_one(path, block_rows=3)
            self.assertEqual(stations[0]["longest_unusable_run_steps"], 5)
            self.assertEqual(stations[0]["longest_internal_unusable_run_steps"], 3)
            self.assertEqual(stations[0]["trailing_unusable_steps"], 5)
            self.assertEqual(stations[1]["longest_unusable_run_steps"], 12)
            self.assertEqual(stations[1]["longest_internal_unusable_run_steps"], 0)
            self.assertIsNone(stations[1]["first_usable_row"])
            self.assertEqual(stations[2]["pm25_mask_true_nonfinite"], 1)
            self.assertEqual(stations[2]["pm25_usable"], 11)
            self.assertEqual(stations[2]["pm25_negative"], 1)
            self.assertEqual(stations[2]["pm25_zero"], 1)
            for j in range(2):
                good = (masks[:, :, j] == 1) & np.isfinite(values[:, :, j])
                self.assertEqual(pollutants[j]["usable"], int(good.sum()))
                self.assertAlmostEqual(pollutants[j]["mean_observed_finite"], float(values[:, :, j][good].astype(float).mean()))
            self.assertEqual(pollutants[1]["invalid_mask"], 1)
            self.assertFalse(summary["time_axis_validated"])

    def test_bad_shape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aq_compact_2021.h5"
            self.make_file(path)
            with h5py.File(path, "a") as f:
                del f["masks"]
                f.create_dataset("masks", data=np.ones((12, 2, 2), dtype=bool))
            with self.assertRaises(ValueError):
                audit_one(path)

    def test_cli_packages_report_and_signals_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            self.make_file(data / "aq_compact_2021.h5")
            process = subprocess.run([sys.executable, str(Path(__file__).with_name("audit_data.py")), "--data", str(data), "--out", str(root / "audit")], capture_output=True, text=True)
            self.assertEqual(process.returncode, 1, process.stderr)
            self.assertTrue((root / "audit" / "audit_results.zip").exists())
            report = json.loads((root / "audit" / "summary.json").read_text())
            self.assertEqual(len(report["files"]), 1)
            self.assertEqual(report["files"][0]["pm25_usable_cells"], 15)

    def test_json_scalar_array(self):
        self.assertEqual(jsonable(np.array(1)), 1)


if __name__ == "__main__":
    unittest.main()
