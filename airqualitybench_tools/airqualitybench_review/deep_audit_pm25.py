#!/usr/bin/env python3
"""Read-only second-stage PM2.5 diagnostics. No imputation, training or inferred dates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def text(v: Any) -> str:
    return v.decode('utf-8') if isinstance(v, bytes) else str(v)


def normalize(v: str) -> str:
    return re.sub(r'[^a-z0-9]', '', v.lower())


def runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """True intervals as [start, stop), including leading/trailing intervals."""
    edges = np.diff(np.r_[False, flag, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def consecutive_count(good: np.ndarray, width: int) -> int:
    if len(good) < width:
        return 0
    prefix = np.r_[0, np.cumsum(good, dtype=np.int64)]
    return int(np.count_nonzero(prefix[width:] - prefix[:-width] == width))


def distribution(x: np.ndarray) -> dict[str, Any]:
    """Exact quantiles of finite values. Do not interpret these as verified units."""
    if not x.size:
        return {'count': 0, 'min': None, 'p50': None, 'p90': None,
                'p99': None, 'p999': None, 'max': None, 'zero_count': 0,
                'negative_count': 0}
    q = np.quantile(x, [0, .5, .9, .99, .999, 1])
    return {'count': int(x.size), **dict(zip(
        ['min', 'p50', 'p90', 'p99', 'p999', 'max'], map(float, q))),
        'zero_count': int(np.count_nonzero(x == 0)),
        'negative_count': int(np.count_nonzero(x < 0))}


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='raise')
        writer.writeheader()
        writer.writerows(rows)


def scan_file(path: Path, cohort_ids: list[str]) -> dict[str, Any]:
    match = re.fullmatch(r'aq_compact_(\d{4})\.h5', path.name)
    if not match:
        raise ValueError(f'Unexpected filename: {path.name}')
    year = int(match[1])
    with h5py.File(path, 'r') as f:
        for name in ('values', 'masks', 'params', 'station_ids'):
            if name not in f or not isinstance(f[name], h5py.Dataset):
                raise ValueError(f'{path.name}: missing dataset {name}')
        if f['params'].ndim != 1 or f['station_ids'].ndim != 1:
            raise ValueError('params/station_ids must be one-dimensional')
        params = [text(v) for v in f['params'][:]]
        ids = [text(v) for v in f['station_ids'][:]]
        indices = [i for i, p in enumerate(params) if normalize(p) in ('pm25', 'pm2p5')]
        if len(indices) != 1 or len(set(ids)) != len(ids):
            raise ValueError('PM2.5 not unique or duplicate station IDs')
        if (f['values'].ndim != 3 or f['values'].shape != f['masks'].shape
                or f['values'].shape[1:] != (len(ids), len(params))):
            raise ValueError('Incompatible tensor dimensions')
        missing_ids = sorted(set(cohort_ids) - set(ids))
        if missing_ids:
            raise ValueError(f'Cohort IDs missing in {path.name}: {missing_ids[:10]}')
        # Load one pollutant from one year, not all years/channels at once.
        values = f['values'][:, :, indices[0]]
        masks = f['masks'][:, :, indices[0]]
    if not values.size:
        raise ValueError('Empty data array')
    invalid_masks = int(np.count_nonzero(~((masks == 0) | (masks == 1))))
    if invalid_masks:
        raise ValueError(f'{invalid_masks} non-binary mask cells')
    finite = np.isfinite(values)
    good = (masks == 1) & finite
    t, n = good.shape
    active = good.any(axis=0)
    active_n = int(active.sum())
    index = {sid: i for i, sid in enumerate(ids)}
    ci = np.asarray([index[sid] for sid in cohort_ids], dtype=np.int64)
    counts = good.sum(axis=1)
    cohort_counts = good[:, ci].sum(axis=1)
    fractions = cohort_counts / len(cohort_ids)
    coverage = [{'year': year, 'row_index': i,
                 'usable_all_station_count': int(counts[i]),
                 'all_station_count': n,
                 'annual_active_station_count': active_n,
                 'usable_fraction_among_annual_active': float(counts[i] / active_n) if active_n else None,
                 'fixed_train_cohort_count': len(cohort_ids),
                 'usable_fixed_cohort_count': int(cohort_counts[i]),
                 'usable_fixed_cohort_fraction': float(fractions[i])} for i in range(t)]
    low_intervals = []
    for threshold in (.10, .50):
        for start, stop in runs(fractions < threshold):
            low_intervals.append({'year': year, 'threshold': threshold,
                'start_row_inclusive': start, 'stop_row_exclusive': stop,
                'length_steps': stop - start,
                'min_cohort_fraction': float(fractions[start:stop].min()),
                'mean_cohort_fraction': float(fractions[start:stop].mean())})
    distributions = []
    for label, flag in (('mask_1_finite', good),
                        ('mask_0_finite_not_to_be_restored', (masks == 0) & finite)):
        distributions.append({'year': year, 'population': label,
                              **distribution(values[flag])})
    longest_gaps = []
    repeated_internal: Counter[tuple[int, int]] = Counter()
    continuity = []
    cohort_set = set(cohort_ids)
    for j, sid in enumerate(ids):
        valid = good[:, j]
        gs = runs(~valid)
        for start, stop in gs:
            if start > 0 and stop < t and stop - start >= 24:
                repeated_internal[(start, stop)] += 1
        for start, stop in sorted(gs, key=lambda pair: (-(pair[1] - pair[0]), pair[0]))[:3]:
            kind = ('all_year_missing' if start == 0 and stop == t else
                    'leading' if start == 0 else 'trailing' if stop == t else 'internal')
            longest_gaps.append({'year': year, 'station_id': sid,
                'gap_kind': kind, 'start_row_inclusive': start,
                'stop_row_exclusive': stop, 'length_steps': stop - start})
        continuity.append({'year': year, 'station_id': sid,
            'in_fixed_train_cohort': int(sid in cohort_set),
            'usable': int(valid.sum()),
            'consecutive_observed_pair_count': consecutive_count(valid, 2),
            'fully_observed_6_history_plus_target_count': consecutive_count(valid, 7),
            'fully_observed_24_history_plus_target_count': consecutive_count(valid, 25)})
    repeated = [{'year': year, 'start_row_inclusive': a, 'stop_row_exclusive': b,
                 'length_steps': b - a, 'stations_with_exact_same_internal_gap': count}
                for (a, b), count in sorted(repeated_internal.items(),
                    key=lambda item: (-item[1], item[0]))[:100]]
    examples = []
    for label, flag in (('mask_1_finite', good), ('mask_0_finite', (masks == 0) & finite)):
        # Top 10 values per population, selected only for descriptive inspection.
        flat_idx = np.flatnonzero(flag)
        if not len(flat_idx):
            continue
        selected_values = values.ravel()[flat_idx]
        k = min(10, len(flat_idx))
        top = np.argpartition(selected_values, len(selected_values) - k)[-k:]
        top = top[np.argsort(-selected_values[top], kind='stable')]
        for idx in flat_idx[top]:
            row, col = np.unravel_index(int(idx), values.shape)
            examples.append({'year': year, 'population': label,
                'row_index': int(row), 'station_id': ids[col],
                'raw_value_units_unverified': float(values[row, col]),
                'mask': int(masks[row, col])})
        del selected_values, flat_idx
    summary = {'year': year, 'rows': t, 'stations': n,
        'annual_active_stations': active_n, 'fixed_train_cohort_stations': len(cohort_ids),
        'mask_1_nonfinite_count': int(np.count_nonzero((masks == 1) & ~finite)),
        'steps_with_zero_usable_all_stations': int(np.count_nonzero(counts == 0)),
        'steps_with_zero_usable_fixed_cohort': int(np.count_nonzero(cohort_counts == 0)),
        'longest_below_10pct_fixed_cohort_steps': max((r['length_steps'] for r in low_intervals if r['threshold'] == .1), default=0),
        'timestamp_semantics_verified': False}
    return {'summary': summary, 'coverage_by_step': coverage,
            'low_coverage_intervals': low_intervals, 'value_distributions': distributions,
            'station_longest_gaps': longest_gaps, 'repeated_internal_gaps': repeated,
            'station_continuity': continuity, 'extreme_examples': examples}


COLUMNS = {
    'coverage_by_step': ['year', 'row_index', 'usable_all_station_count', 'all_station_count',
        'annual_active_station_count', 'usable_fraction_among_annual_active',
        'fixed_train_cohort_count', 'usable_fixed_cohort_count', 'usable_fixed_cohort_fraction'],
    'low_coverage_intervals': ['year', 'threshold', 'start_row_inclusive', 'stop_row_exclusive',
        'length_steps', 'min_cohort_fraction', 'mean_cohort_fraction'],
    'value_distributions': ['year', 'population', 'count', 'min', 'p50', 'p90', 'p99', 'p999', 'max', 'zero_count', 'negative_count'],
    'station_longest_gaps': ['year', 'station_id', 'gap_kind', 'start_row_inclusive', 'stop_row_exclusive', 'length_steps'],
    'repeated_internal_gaps': ['year', 'start_row_inclusive', 'stop_row_exclusive', 'length_steps', 'stations_with_exact_same_internal_gap'],
    'station_continuity': ['year', 'station_id', 'in_fixed_train_cohort', 'usable',
        'consecutive_observed_pair_count', 'fully_observed_6_history_plus_target_count',
        'fully_observed_24_history_plus_target_count'],
    'extreme_examples': ['year', 'population', 'row_index', 'station_id', 'raw_value_units_unverified', 'mask'],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('audit_deep'))
    parser.add_argument('--years', type=int, nargs='+', default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument('--cohort', type=Path, default=Path(__file__).with_name('training_candidates_669.csv'))
    args = parser.parse_args()
    try:
        with args.cohort.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        cohort = [r['station_id'] for r in rows]
        if not cohort or len(cohort) != len(set(cohort)):
            raise ValueError('Cohort must contain unique, nonempty station IDs')
        paths = [args.data / f'aq_compact_{y}.h5' for y in sorted(set(args.years))]
        absent = [str(p) for p in paths if not p.is_file()]
        if absent:
            raise FileNotFoundError('Missing: ' + ', '.join(absent))
        args.out.mkdir(parents=True, exist_ok=True)
        combined = {k: [] for k in COLUMNS}
        summaries = []
        for path in paths:
            print(f'Scanning PM2.5: {path.name}', flush=True)
            result = scan_file(path, cohort)
            summaries.append(result.pop('summary'))
            for key, records in result.items():
                combined[key].extend(records)
        for key, columns in COLUMNS.items():
            write_csv(args.out / (key + '.csv'), combined[key], columns)
        summary = {'scope': 'Read-only descriptive PM2.5 diagnostics; no model training or test-based selection',
            'cohort_file': args.cohort.name,
            'cohort_sha256': hashlib.sha256(args.cohort.read_bytes()).hexdigest(),
            'files': summaries,
            'limitations': [
                'Row indexes only: no inferred dates, timezone or hours.',
                'Calibration, units and mask origins require source documentation.',
                'Coincident gaps do not establish a collection failure or its cause.',
                'Annual-active denominators are descriptive, not station-selection criteria.',
                'Continuity/gaps are evaluated within each year, not across year boundaries.',
                'Fully observed windows are counts for diagnosis, not a proposed filter on model targets.',
                'First-stage file checksums should be retained for version tracking.',
                'Only the three longest gaps per station/year and top 100 repeated internal intervals are exported.',
                'No recovery of true arrival timestamps is attempted.',
                'Do not tune the cohort or model based on 2025 diagnostic values.']}
        (args.out / 'deep_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        archive = args.out / 'audit_deep_results.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for name in [k + '.csv' for k in COLUMNS] + ['deep_summary.json']:
                z.write(args.out / name, name)
        print(f'Done: {archive.resolve()}')
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
