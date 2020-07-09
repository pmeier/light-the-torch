import pytest

import ltt
from ltt._core import resolve
from ltt._core.common import InternalLTTError


def test_resolve_dists_internal_error(mocker):
    mocker.patch("ltt._core.resolve.run")

    with pytest.raises(InternalLTTError):
        ltt.resolve_dists(["foo"])


def test_StopAfterPytorchDistsFoundResolver_no_torch(mocker):
    mocker.patch("ltt._core.resolve.PatchedResolverBase.__init__", return_value=None)
    resolver = resolve.StopAfterPytorchDistsFoundResolver()
    resolver._required_pytorch_dists = ["torchaudio", "torchtext", "torchvision"]
    assert "torch" in resolver.required_pytorch_dists


# @pytest.mark.slow
# def test_resolve_dists_lighter():
#     assert ltt.resolve_dists(["ltt"]) == []


@pytest.mark.large_download
@pytest.mark.slow
def test_resolve_dists_pystiche(subtests):
    pystiche = "git+https://github.com/pmeier/pystiche@v{}"
    reqs_and_dists = (
        (pystiche.format("0.4.0"), {"torch>=1.4.0", "torchvision>=0.5.0"}),
        (pystiche.format("0.5.0"), {"torch>=1.5.0", "torchvision>=0.6.0"}),
    )
    for req, dists in reqs_and_dists:
        with subtests.test(req=req):
            assert set(ltt.resolve_dists([req])) == dists


@pytest.mark.large_download
@pytest.mark.slow
def test_resolve_dists_kornia(subtests):
    kornia = "kornia=={}"
    reqs_and_dists = (
        (kornia.format("0.2.2"), {"torch<=1.4.0,>=1.0.0"}),
        (kornia.format("0.3.1"), {"torch==1.5.0"}),
    )
    for req, dists in reqs_and_dists:
        with subtests.test(req=req):
            assert set(ltt.resolve_dists([req])) == dists
