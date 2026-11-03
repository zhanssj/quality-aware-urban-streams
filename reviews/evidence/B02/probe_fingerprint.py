"""Count full-calendar hashing; no wall-clock performance claims or raw inputs."""
import json
from unittest.mock import patch
import numpy as np
from aqstream.features import FallbackState, YearGrid
from aqstream.replay.calendars import Calendar, ScenarioSpec
from aqstream.replay.mixture import weighted_training_samples

def main():
    calls = 0
    original = Calendar.fingerprint.fget
    def counted(calendar):
        nonlocal calls
        calls += 1
        return original(calendar)
    values=np.arange(160,dtype=float).reshape(80,2)
    valid=np.ones_like(values,dtype=bool)
    fitted=FallbackState(('a','b'),(7.,8.),(1,1),7.5,(2023,))
    with patch.object(Calendar,'fingerprint',property(counted)):
        samples=list(weighted_training_samples(values,valid,fitted,YearGrid((2023,),(80,)),
                     specs=(ScenarioSpec('clean'),),weights=(1.,),train_seed=2101))
    print(json.dumps({'calendar_count':1,'grid_steps':80,'stations':2,
                      'yielded_feature_frames':len(samples),'full_fingerprint_calls':calls,
                      'unique_returned_calendar_hashes':len({s.calendar_sha256 for s in samples}),
                      'timing_benchmark':False},indent=2))
if __name__=='__main__': main()
