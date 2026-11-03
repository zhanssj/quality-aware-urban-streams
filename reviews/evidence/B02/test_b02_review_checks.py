"""Independent B02 tests. Run inside the real repo with PYTHONPATH=src.

These tests do not open HDF5, load pickle models, or use network services.
The local review used exact core source copies and import-only IO adapters.
"""
import csv
from dataclasses import replace

import numpy as np
import pytest

from aqstream.data import DataContractError
from aqstream.features import FallbackState, YearGrid
from aqstream.evaluation import MODEL_NAMES, evaluate_predictions
from aqstream.prediction_plan import PredictionPlan
from aqstream.replay.calendars import ScenarioSpec, make_calendar
from aqstream.replay.stream import replay_snapshots
from aqstream.replay.mixture import weighted_training_samples

LAGS = (1, 2, 3, 6, 12, 24)


def reference(values, valid, fitted, calendar, target, cutoff):
    """Recompute history from scratch, not using the production kernel/buffer."""
    n = values.shape[1]
    rows = [np.array(fitted.medians, dtype=float) for _ in range(24)]
    masks = [np.zeros(n, dtype=bool) for _ in range(24)]
    last = np.array(fitted.medians, dtype=float)
    last_idx = np.full(n, -1)
    for event in range(target):
        present = valid[event].copy() & np.isfinite(values[event])
        if event >= calendar.input_start:
            local = event - calendar.input_start
            present &= ~calendar.dropped[local]
            present &= (event + 1) * 60 + calendar.delay_minutes[local] <= cutoff
        last = np.where(present, values[event], last)
        last_idx = np.where(present, event, last_idx)
        rows.append(last.copy())
        masks.append(present.copy())
    filled = np.array(rows[-24:])
    available = np.array(masks[-24:])
    common = np.column_stack([
        *[filled[-lag] for lag in LAGS],
        filled[-6:].mean(axis=0), filled[-6:].std(axis=0),
        filled.mean(axis=0), filled.std(axis=0),
    ])
    age = np.where(last_idx >= 0, target - 1 - last_idx, -1)
    quality = np.column_stack([
        *[available[-lag] for lag in LAGS], age,
        1 - available[-6:].mean(axis=0), 1 - available.mean(axis=0),
        np.where(last_idx >= 0, target - 1 - last_idx, target), last_idx >= 0,
    ])
    return common, quality, last


def fixture(seed=2, start=37, delay=30):
    rng = np.random.default_rng(seed)
    values = rng.normal(20, 5, (150, 3))
    valid = rng.random(values.shape) > 0.25
    valid[65:115, 1] = False
    values[~valid] = 9e8  # Must not influence features.
    fitted = FallbackState(('a','b','c'), (7.,8.,9.), (1,1,1), 8., (2023,))
    grid = YearGrid((2023,2024), (37,113))
    stop = 145
    spec = ScenarioSpec('review', random_drop_rate=0.2, block_duration=6,
                        block_period=25, block_station_count=1,
                        delay_fraction=0.6, max_delay_minutes=delay)
    calendar = make_calendar(valid[start-1:stop-1], fitted.station_ids,
                             input_start=start-1, spec=spec, seed=seed)
    return values,valid,fitted,grid,calendar,start,stop


@pytest.mark.parametrize('seed',[1,7,19])
@pytest.mark.parametrize('start',[1,37])
@pytest.mark.parametrize('delay',[30,2400])
def test_replay_matches_independent_asof_reference(seed,start,delay):
    values,valid,fitted,grid,calendar,start,stop = fixture(seed,start,delay)
    original = values.copy()
    previous=[]
    for snapshot in replay_snapshots(values,valid,fitted,grid,calendar,
            target_start=start,target_stop=stop,waits=(0,5,15)):
        target=snapshot.features.sample_ids[0].target_index
        x,q,p = reference(values,valid,fitted,calendar,target,snapshot.cutoff_minutes)
        np.testing.assert_allclose(snapshot.features.common,x,rtol=0,atol=1e-13)
        np.testing.assert_array_equal(snapshot.features.quality,q)
        np.testing.assert_array_equal(snapshot.features.persistence,p)
        assert snapshot.target_end_minutes-snapshot.cutoff_minutes == snapshot.remaining_lead_minutes
        assert not snapshot.features.common.flags.writeable
        previous.append((snapshot.features, snapshot.features.common.copy()))
    for frame,copy in previous:
        np.testing.assert_array_equal(frame.common,copy)
    np.testing.assert_array_equal(values,original)


@pytest.mark.parametrize('wait',[0,5,15])
def test_policy_is_independent_of_other_waits(wait):
    values,valid,fitted,grid,calendar,start,stop=fixture(delay=2400)
    combined=[s for s in replay_snapshots(values,valid,fitted,grid,calendar,
              target_start=start,target_stop=stop,waits=(0,5,15)) if s.wait_minutes==wait]
    isolated=list(replay_snapshots(values,valid,fitted,grid,calendar,
                 target_start=start,target_stop=stop,waits=(wait,)))
    for a,b in zip(combined,isolated,strict=True):
        assert a.features.sample_ids == b.features.sample_ids
        np.testing.assert_array_equal(a.features.common,b.features.common)
        np.testing.assert_array_equal(a.features.quality,b.features.quality)


def test_future_and_target_mutation_with_frozen_calendar():
    v,m,f,g,c,start,stop=fixture()
    before=list(replay_snapshots(v,m,f,g,c,target_start=start,target_stop=stop,waits=(0,5,15)))
    changed=v.copy(); changed[85:]=123456
    after=list(replay_snapshots(changed,m,f,g,c,target_start=start,target_stop=stop,waits=(0,5,15)))
    for a,b in zip(before,after,strict=True):
        if a.features.sample_ids[0].target_index<=85:
            np.testing.assert_array_equal(a.features.common,b.features.common)
            np.testing.assert_array_equal(a.features.quality,b.features.quality)


def test_calendar_nested_drops_counts_and_truth_unchanged():
    rng=np.random.default_rng(5); valid=rng.random((83,4))>0.4; copy=valid.copy()
    ids=('a','b','c','d')
    low=make_calendar(valid,ids,input_start=8,spec=ScenarioSpec('low',random_drop_rate=.1),seed=99)
    high=make_calendar(valid,ids,input_start=8,spec=ScenarioSpec('high',random_drop_rate=.2),seed=99)
    delayed=make_calendar(valid,ids,input_start=8,spec=ScenarioSpec('low',random_drop_rate=.1,delay_fraction=.3),seed=99)
    assert not np.any(low.dropped & ~high.dropped)
    assert not np.any(low.dropped & ~valid)
    assert int(high.dropped.sum()) == round(.2*valid.sum())
    np.testing.assert_array_equal(low.dropped,delayed.dropped)
    assert not np.any((delayed.delay_minutes>0) & (~valid|delayed.dropped))
    np.testing.assert_array_equal(copy,valid)


def records():
    plan=PredictionPlan(('a','b'),10,13,0,'run','scenario',5)
    rows=[]
    for t in range(10,13):
        for s in plan.station_ids:
            rows.append(dict(sample_id=f'{s}:{t-1}:{t}',station_id=s,
              origin_index=t-1,target_index=t,target_year=2024,target_local_index=t-10,
              run_id='run',scenario_id='scenario',wait_minutes=5,split='validation',
              source_valid=1,target=t,origin_minutes=t*60,cutoff_minutes=t*60+5,
              target_end_minutes=(t+1)*60,remaining_lead_minutes=55,
              persistence=t+1,standard=t+2,quality_aware=t+3))
    return plan,rows


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)


@pytest.mark.parametrize('tamper',['delete','duplicate','station','target','wait','scenario','run','clock'])
def test_n1_rejects_incomplete_or_mixed_schedule(tmp_path,tamper):
    plan,rows=records()
    if tamper=='delete': rows.pop()
    elif tamper=='duplicate': rows[-1]=rows[0].copy()
    elif tamper=='station':
        rows[-1]['station_id']='unknown'; rows[-1]['sample_id']='unknown:11:12'
    elif tamper=='target':
        rows[-1].update(origin_index=12,target_index=13,sample_id='b:12:13',target_local_index=3)
    elif tamper=='wait': rows[-1]['wait_minutes']=15
    elif tamper=='scenario': rows[-1]['scenario_id']='other'
    elif tamper=='run': rows[-1]['run_id']='other'
    elif tamper=='clock': rows[-1]['remaining_lead_minutes']=60
    path=tmp_path/'p.csv';write(path,rows)
    with pytest.raises(DataContractError): evaluate_predictions(path,plan=plan)


def test_complete_schedule_missing_prediction_and_truth_denominators(tmp_path):
    plan,rows=records();rows[0]['standard']='';rows[1].update(source_valid=0,target='')
    path=tmp_path/'p.csv';write(path,rows)
    result=evaluate_predictions(path,plan=plan)
    assert result['planned']==result['present_rows']==6
    for name in MODEL_NAMES:
        m=result['models'][name]
        assert m['scored']==4 and m['eligible_truth']==5
        assert m['skipped_missing_truth']==1 and m['skipped_missing_any_prediction']==1
    assert result['models']['standard']['prediction_coverage']==5/6
    assert result['models']['persistence']['prediction_coverage']==1
    assert result['models']['standard']['mae']==2


def test_training_mixture_preserves_truth_and_unit_weight_per_sample():
    values=np.arange(160,dtype=float).reshape(80,2); valid=np.ones_like(values,dtype=bool)
    valid[15:20,0]=False
    fitted=FallbackState(('a','b'),(7.,8.),(1,1),7.5,(2023,))
    specs=(ScenarioSpec('clean'),ScenarioSpec('random',random_drop_rate=.3),ScenarioSpec('delay',delay_fraction=.3))
    sums={}
    for sample in weighted_training_samples(values,valid,fitted,YearGrid((2023,),(80,)),
            specs=specs,weights=(.2,.3,.5),train_seed=2101):
        for col,identity in enumerate(sample.features.sample_ids):
            sums[identity.key]=sums.get(identity.key,0)+sample.weight
            t=identity.target_index
            assert sample.truth.source_valid[col]==valid[t,col]
            if valid[t,col]: assert sample.truth.values[col]==values[t,col]
            else: assert np.isnan(sample.truth.values[col])
    assert len(sums)==160
    assert all(abs(w-1)<1e-12 for w in sums.values())


def test_training_mixture_rejects_validation_grid():
    fitted=FallbackState(('a',),(7.,),(1,),7.,(2023,))
    with pytest.raises(DataContractError,match='validation/test'):
        list(weighted_training_samples(np.ones((2,1)),np.ones((2,1),bool),fitted,
            YearGrid((2024,),(2,)),specs=(ScenarioSpec('clean'),),weights=(1.,),train_seed=2101))
