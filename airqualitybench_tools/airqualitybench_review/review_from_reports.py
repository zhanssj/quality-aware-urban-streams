#!/usr/bin/env python3
"""Reproduce aggregate checks and train-only diagnostic cohort from audit reports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit', type=Path, default=Path(__file__).with_name('source_audit'))
    p.add_argument('--out', type=Path, default=Path(__file__).parent)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    stations = read_csv(a.audit/'pm25_station_year.csv')
    pollutants = read_csv(a.audit/'pollutant_year_summary.csv')
    source = json.loads((a.audit/'summary.json').read_text(encoding='utf-8'))
    index = {}
    for r in stations:
        key = (int(r['year']), r['station_id'])
        if key in index: raise ValueError(f'Duplicate {key}')
        index[key] = r
    annual = []
    for f in source['files']:
        y = f['year_from_filename']
        rows = [r for r in stations if int(r['year']) == y]
        active = [r for r in rows if int(r['pm25_usable']) > 0]
        usable = sum(int(r['pm25_usable']) for r in rows)
        expected = next(r for r in pollutants if int(r['year']) == y and r['pollutant'] == 'pm25')
        assert usable == f['pm25_usable_cells'] == int(expected['usable'])
        assert set(r['station_id'] for r in rows) == set(f['station_ids'])
        assert len(rows) == f['shape'][1]
        counts = {str(t): sum(float(r['pm25_usable_fraction']) >= t for r in rows) for t in [.5, .7, .8, .9, .95]}
        assert counts['0.8'] == f['pm25_stations_coverage_ge_80pct']
        mode = Counter(int(r['longest_internal_unusable_run_steps']) for r in active)
        annual.append({'year': y, 'rows': f['shape'][0], 'stations': len(rows),
            'usable': usable, 'coverage_all': usable/(len(rows)*f['shape'][0]),
            'stations_with_any_pm25': len(active), 'stations_no_pm25': len(rows)-len(active),
            'coverage_annual_active': usable/(len(active)*f['shape'][0]),
            'max_station_coverage': max(float(r['pm25_usable_fraction']) for r in rows),
            'coverage_threshold_counts': counts, 'common_longest_internal_gap_lengths': mode.most_common(10)})
    ids = sorted(set(r['station_id'] for r in stations))
    years = sorted(set(int(r['year']) for r in stations))
    never = sum(all(int(index[(y,sid)]['pm25_usable']) == 0 for y in years) for sid in ids)
    # Diagnostic cohort is defined only from 2023 training-period masks.
    cohort = [sid for sid in ids if float(index[(2023, sid)]['pm25_usable_fraction']) >= .8]
    with (a.out/'training_candidates_669.csv').open('w', encoding='utf-8-sig', newline='') as out:
        columns = ['station_id'] + [f'coverage_{y}' for y in [2021,2022,2023]] + [f'longest_internal_gap_steps_{y}' for y in [2021,2022,2023]]
        writer = csv.DictWriter(out, fieldnames=columns); writer.writeheader()
        for sid in cohort:
            row = {'station_id': sid}
            for y in [2021,2022,2023]:
                row[f'coverage_{y}'] = float(index[(y,sid)]['pm25_usable_fraction'])
                row[f'longest_internal_gap_steps_{y}'] = int(index[(y,sid)]['longest_internal_unusable_run_steps'])
            writer.writerow(row)
    train_50 = sum(all(float(index[(y,sid)]['pm25_usable_fraction']) >= .5 for y in [2021,2022,2023]) for sid in ids)
    pm = [r for r in pollutants if r['pollutant'] == 'pm25']
    result = {'input_checksums': {name:hashlib.sha256((a.audit/name).read_bytes()).hexdigest() for name in ['summary.json','report.md','pm25_station_year.csv','pollutant_year_summary.csv']},
        'summary_csv_consistency_checks_passed': True, 'annual': annual,
        'pm25_usable_total': sum(int(r['usable']) for r in pm),
        'pm25_grid_cells_total': sum(int(r['total_cells']) for r in pm),
        'pm25_mask_false_finite_total': sum(int(r['masked_finite']) for r in pm),
        'all_pollutants_mask_false_finite_total': sum(int(r['masked_finite']) for r in pollutants),
        'stations_never_pm25_descriptive_only': never,
        'cohort_rule': 'PM2.5 usable_fraction >= 0.80 in 2023 only; diagnostic pool, not final study selection',
        'cohort_count': len(cohort),
        'cohort_2023_min_longest_internal_gap_steps': min(int(index[(2023,sid)]['longest_internal_unusable_run_steps']) for sid in cohort),
        'stations_coverage_ge_50pct_each_training_year': train_50,
        'warning': 'Raw HDF5 files not available to this review. 2025 summaries are descriptive only; no cohort choice uses them.'}
    (a.out/'derived_findings.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['annual','input_checksums']}, ensure_ascii=False, indent=2))

if __name__ == '__main__': main()
