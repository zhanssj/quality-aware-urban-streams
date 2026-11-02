import tempfile
import subprocess
import sys
import zipfile
import json
import unittest
from pathlib import Path
import h5py
import numpy as np
from deep_audit_pm25 import scan_file, runs, consecutive_count

class DeepAuditTests(unittest.TestCase):
    def test_runs_and_windows(self):
        self.assertEqual(runs(np.array([True, True, False, True])), [(0, 2), (3, 4)])
        self.assertEqual(consecutive_count(np.array([True, True, False, True, True, True]), 2), 3)
        self.assertEqual(consecutive_count(np.array([False, False]), 3), 0)

    def make_file(self, root):
        path = root / 'aq_compact_2021.h5'
        v = np.ones((40, 3, 2), dtype=np.float32)
        m = np.ones_like(v, dtype=bool)
        # PM2.5 deliberately second, to test parameter-axis mapping.
        m[5:30, :2, 1] = False
        m[:, 2, 1] = False
        v[4, 0, 1] = 0
        v[6, 0, 1] = 500
        v[34, 1, 1] = np.nan
        with h5py.File(path, 'w') as f:
            f['values'] = v; f['masks'] = m
            f['params'] = np.array([b'no2', b'pm25'])
            f['station_ids'] = np.array([b'a', b'b', b'c'])
        return path

    def test_diagnostics(self):
        with tempfile.TemporaryDirectory() as td:
            path = self.make_file(Path(td))
            r = scan_file(path, ['a', 'b'])
            self.assertEqual(r['summary']['mask_1_nonfinite_count'], 1)
            self.assertEqual(r['summary']['steps_with_zero_usable_fixed_cohort'], 25)
            self.assertEqual(r['summary']['longest_below_10pct_fixed_cohort_steps'], 25)
            g = r['repeated_internal_gaps'][0]
            self.assertEqual((g['start_row_inclusive'], g['stop_row_exclusive']), (5, 30))
            self.assertEqual(g['stations_with_exact_same_internal_gap'], 2)
            self.assertEqual(r['value_distributions'][0]['count'], 29)
            self.assertEqual(r['value_distributions'][0]['zero_count'], 1)
            self.assertEqual(r['value_distributions'][1]['max'], 500)
            self.assertEqual(r['coverage_by_step'][6]['usable_all_station_count'], 0)
            self.assertEqual(r['station_continuity'][0]['consecutive_observed_pair_count'], 13)

    def test_cli_package(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / 'data'; data.mkdir()
            self.make_file(data)
            cohort = root / 'cohort.csv'
            cohort.write_text('station_id\na\nb\n', encoding='utf-8')
            process = subprocess.run([sys.executable,
                str(Path(__file__).with_name('deep_audit_pm25.py')),
                '--data', str(data), '--out', str(root / 'out'),
                '--years', '2021', '--cohort', str(cohort)],
                capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            with zipfile.ZipFile(root / 'out' / 'audit_deep_results.zip') as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(len(z.namelist()), 8)
                result = json.loads(z.read('deep_summary.json'))
                self.assertEqual(result['files'][0]['steps_with_zero_usable_fixed_cohort'], 25)

    def test_missing_cohort_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                scan_file(self.make_file(Path(td)), ['unknown'])

if __name__ == '__main__':
    unittest.main()
