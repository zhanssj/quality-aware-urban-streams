"""Observe the CSV-only denominator limitation on synthetic rows.

Run from the project with its dependencies installed:
    python reproduce_coverage_gap.py
This is a diagnostic reproduction, not an assertion that pilot rows are missing.
"""
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from aqstream.evaluation import MODEL_NAMES, evaluate_predictions


def main():
    fields = ['sample_id', 'station_id', 'origin_index', 'target_index',
              'target_year', 'split', 'source_valid', 'target', *MODEL_NAMES]
    rows = [dict(sample_id=f's0:{i-1}:{i}', station_id='s0', origin_index=i-1,
                 target_index=i, target_year=2024, split='validation', source_valid=1,
                 target=float(i+1), **{model:0.0 for model in MODEL_NAMES})
            for i in range(3)]
    evidence = {'expected_scheduled': 3, 'real_pilot_data_used': False}
    with TemporaryDirectory() as folder:
        for name, subset in [('full', rows), ('truncated', rows[:-1])]:
            path = Path(folder) / f'{name}.csv'
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(subset)
            metrics = evaluate_predictions(path)['models']['persistence']
            evidence[name] = {key:metrics[key] for key in
                              ('scheduled','predicted','prediction_coverage','mae')}
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
