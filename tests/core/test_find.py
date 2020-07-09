import pytest

import ltt
from ltt import computation_backend as cb
from ltt._core.common import InternalLTTError
from ltt._core.find import PytorchCandidatePreferences


def test_resolve_dists_internal_error(mocker):
    mocker.patch("ltt._core.find.run")

    with pytest.raises(InternalLTTError):
        ltt.find_links(["foo"])


def test_PytorchCandidatePreferences_detect_computation_backend(mocker):
    class GenericComputationBackend(cb.ComputationBackend):
        @property
        def local_specifier(self):
            return "generic"

    computation_backend = GenericComputationBackend()
    mocker.patch(
        "ltt._core.find.detect_computation_backend", return_value=computation_backend,
    )

    candidate_prefs = PytorchCandidatePreferences()
    assert candidate_prefs.computation_backend is computation_backend


@pytest.mark.slow
def test_find_links_smoke(subtests):
    computation_backends = [
        cb.ComputationBackend.from_str(string)
        for string in ("cpu", "cu92", "cu101", "cu102")
    ]
    dists = ["torch", "torchaudio", "torchtext", "torchvision"]
    for computation_backend in computation_backends:
        with subtests.test(computation_backend=computation_backend):
            links = ltt.find_links(dists, computation_backend=computation_backend)
            assert len(links) == len(dists)
