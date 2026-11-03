"""Run unchanged core modules on the review interpreter, not a full project install.

Python 3.13 needs postponed annotations for PredictionPlan.load -> PredictionPlan,
which is valid under the project's Python 3.14 requirement. Only compilation of
annotations is adapted; source bytes and algorithms are unchanged. IO imports
are explicit placeholders, never called. No HDF5, pickle, fit or CLI tests here.
"""
import __future__
import importlib.util
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
sys.path.insert(0,str(root/'core'))
if sys.version_info < (3,14):
    path=root/'core/aqstream/prediction_plan.py'
    spec=importlib.util.spec_from_file_location('aqstream.prediction_plan',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    exec(compile(path.read_bytes(),str(path),'exec',flags=__future__.annotations.compiler_flag),module.__dict__)
import pytest
raise SystemExit(pytest.main(['-q',str(root/'tests/test_b02_review_checks.py')]))
