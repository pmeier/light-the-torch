import light_the_torch.computation_backend as cb
import pytest


class GenericComputationBackend(cb.ComputationBackend):
    @property
    def local_specifier(self):
        return "generic"

    def __lt__(self, other):
        return NotImplemented


@pytest.fixture
def generic_backend():
    return GenericComputationBackend()
