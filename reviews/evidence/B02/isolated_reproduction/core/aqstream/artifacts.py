"""Review-only unused IO adapter for array-only calendar tests."""
def write_json(*args, **kwargs):
    raise RuntimeError('Calendar.save is not exercised in the isolated review harness')
