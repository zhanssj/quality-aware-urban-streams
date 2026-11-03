"""Additional B01 review checks; only synthetic data, no real HDF5 or network.

The coverage truncation observation is recorded by a separate script, not
silently converted into a passing correctness test here.
"""

import csv
import math

import h5py
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from aqstream.data import DataContractError, HDF5Reader
from aqstream.evaluation import MODEL_NAMES, evaluate_predictions
from aqstream.features import YearGrid, fit_fallback, scheduled_samples
from aqstream.models import Baselines


def write_source(root, year, values, valid, permutation):
    with h5py.File(root / f"aq_compact_{year}.h5", "w") as handle:
        handle["values"] = values[:, permutation, None]
        handle["masks"] = valid[:, permutation, None]
        handle["station_ids"] = [f"s{i}".encode() for i in permutation]
        handle["params"] = [b"pm25"]


@pytest.mark.parametrize("chunk_size", [1, 3, 24, 168])
@pytest.mark.parametrize("seed", [7, 19])
def test_scheduled_features_match_independent_multiyear_reference(tmp_path, chunk_size, seed):
    rng = np.random.default_rng(seed)
    lengths = (17, 31, 49, 29)
    values = rng.normal(20, 4, size=(sum(lengths), 3))
    valid = rng.random(values.shape) > 0.35
    valid[:5, 0] = False
    valid[30:85, 1] = False
    # Invalid but finite source values must not enter fallback, features or labels.
    raw = np.where(valid, values, 999999.)
    offset = 0
    for year, length in zip(range(2021, 2025), lengths, strict=True):
        order = rng.permutation(3)
        write_source(tmp_path, year, raw[offset:offset+length], valid[offset:offset+length], order)
        offset += length
    reader = HDF5Reader(tmp_path)
    state = fit_fallback(reader, ('s2', 's0', 's1'), chunk_size=chunk_size)
    grid = YearGrid.from_reader(reader, (2021, 2022, 2023, 2024))
    order = [2, 0, 1]
    expected_fallback = np.array([
        np.median(values[:97, col][valid[:97, col]]) for col in order
    ])
    np.testing.assert_array_equal(state.medians, expected_fallback)
    last = expected_fallback.copy()
    last_indices = np.full(3, -1)
    history = [last.copy() for _ in range(24)]
    masks = [np.zeros(3, bool) for _ in range(24)]
    gap = np.zeros(3)
    snapshots = []
    for target, (frame, truth) in enumerate(scheduled_samples(reader, state, grid, chunk_size=chunk_size)):
        common = np.column_stack((
            *(history[-lag] for lag in (1, 2, 3, 6, 12, 24)),
            np.mean(history[-6:], axis=0), np.std(history[-6:], axis=0),
            np.mean(history[-24:], axis=0), np.std(history[-24:], axis=0),
        ))
        quality = np.column_stack((
            *(masks[-lag] for lag in (1, 2, 3, 6, 12, 24)),
            np.where(last_indices >= 0, target-1-last_indices, -1),
            1-np.mean(masks[-6:], axis=0), 1-np.mean(masks[-24:], axis=0),
            gap, last_indices >= 0,
        ))
        np.testing.assert_allclose(frame.common, common, rtol=0, atol=1e-13)
        np.testing.assert_array_equal(frame.quality, quality)
        np.testing.assert_array_equal(frame.persistence, last)
        np.testing.assert_array_equal(truth.source_valid, valid[target, order])
        np.testing.assert_array_equal(truth.values, np.where(valid[target, order], values[target, order], np.nan))
        assert all(s.split == ('train' if target < 97 else 'validation') for s in frame.sample_ids)
        assert all(s.origin_index == target-1 and s.target_index == target for s in frame.sample_ids)
        snapshots.append((frame, frame.common.copy()))
        flags = valid[target, order]
        last = np.where(flags, values[target, order], last)
        last_indices = np.where(flags, target, last_indices)
        gap = np.where(flags, 0, gap+1)
        history.append(last.copy())
        masks.append(flags.copy())
    assert len(snapshots) == sum(lengths)
    for frame, original in snapshots:
        np.testing.assert_array_equal(frame.common, original)


def rows_fixture():
    result=[]
    for station, target, truth, predictions in (
        ('s0', 100, 4., (3., 2., 4.)),
        ('s0', 101, None, (500., 500., 500.)),
        ('s0', 102, 8., (11., 8., 10.)),
        ('s1', 100, 10., (8., 14., 11.)),
        ('s1', 101, 9., (10., '', 11.)),
    ):
        result.append(dict(
            sample_id=f'{station}:{target-1}:{target}', station_id=station,
            origin_index=target-1, target_index=target, target_year=2024,
            split='validation', source_valid=int(truth is not None),
            target='' if truth is None else truth,
            **dict(zip(MODEL_NAMES, predictions, strict=True))
        ))
    return result


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_independent_hand_metrics_and_missing_predictions(tmp_path):
    metrics=evaluate_predictions(write_csv(tmp_path/'pred.csv', rows_fixture()))['models']
    expected_errors={'persistence':[-1.,3.,-2.], 'standard':[-2.,0.,4.], 'quality_aware':[0.,2.,1.]}
    for name, errors in expected_errors.items():
        item=metrics[name]
        assert item['mae'] == pytest.approx(sum(map(abs, errors))/3)
        assert item['rmse'] == pytest.approx(math.sqrt(sum(e*e for e in errors)/3))
        assert item['macro_station_mae'] == pytest.approx(((abs(errors[0])+abs(errors[1]))/2+abs(errors[2]))/2)
        assert (item['scheduled'],item['eligible_truth'],item['scored']) == (5,4,3)
        assert item['skipped_missing_truth'] == item['skipped_missing_any_prediction'] == 1
    assert metrics['standard']['prediction_coverage'] == .8
    assert len({r['scored_target_ids_sha256'] for r in metrics.values()})==1


@pytest.mark.parametrize('fault', ['duplicate','identity','filled_missing_truth','test_year'])
def test_evaluator_rejects_corrupt_inputs(tmp_path, fault):
    rows=rows_fixture()
    if fault=='duplicate': rows.append(rows[0].copy())
    elif fault=='identity': rows[0]['sample_id']='s0:1:3'
    elif fault=='filled_missing_truth': rows[1]['target']=42.
    elif fault=='test_year': rows[0]['target_year']=2025
    with pytest.raises(DataContractError):
        evaluate_predictions(write_csv(tmp_path/'bad.csv',rows))


def test_prediction_batching_does_not_change_fitted_model_outputs():
    rng=np.random.default_rng(35)
    x=rng.normal(size=(140,10))
    q=rng.integers(0,2,size=(140,11)).astype(float)
    labels=x[:,0]*.7 + q[:,0]*.3
    with threadpool_limits(limits=1):
        models=Baselines(seed=5,model_params={'max_iter':8,'max_leaf_nodes':5,'min_samples_leaf':3})
        models.fit(x,q,labels)
        whole,_=models.predict(x,q,x[:,0])
        for chunk in (1,7,50):
            parts={name:[] for name in MODEL_NAMES}
            for left in range(0,len(x),chunk):
                right=min(left+chunk,len(x))
                predicted,_=models.predict(x[left:right],q[left:right],x[left:right,0])
                for name in MODEL_NAMES: parts[name].extend(predicted[name])
            for name in MODEL_NAMES:
                np.testing.assert_allclose(parts[name], whole[name], rtol=0, atol=1e-12)
        assert models.standard.get_params()==models.quality_aware.get_params()
        assert models.standard.get_params()['early_stopping'] is False
        np.testing.assert_array_equal(whole['persistence'],x[:,0])
