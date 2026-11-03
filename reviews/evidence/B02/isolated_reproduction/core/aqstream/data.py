"""Review-only import adapter. HDF5 is deliberately unavailable in isolated array tests."""
class DataContractError(ValueError):
    pass
class HDF5Reader:
    def __init__(self, *args, **kwargs):
        raise RuntimeError('HDF5 reader is not loaded in this isolated review harness')
