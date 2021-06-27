import pytest

import light_the_torch.computation_backend as cb


class GenericComputationBackend(cb.ComputationBackend):
    @property
    def local_specifier(self):
        return "generic"

    def __lt__(self, other):
        return NotImplemented


@pytest.fixture
def generic_backend():
    return GenericComputationBackend()
