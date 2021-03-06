import pytest

import light_the_torch as ltt
from light_the_torch._pip import extract
from light_the_torch._pip.common import InternalLTTError


def test_StopAfterPytorchDistsFoundResolver_no_torch(mocker):
    mocker.patch(
        "light_the_torch._pip.extract.PatchedResolverBase.__init__", return_value=None
    )
    resolver = extract.StopAfterPytorchDistsFoundResolver()
    resolver._required_pytorch_dists = ["torchaudio", "torchtext", "torchvision"]
    assert "torch" in resolver.required_pytorch_dists


def test_StopAfterPytorchDistsFoundResolver_torch_compatibility(mocker):
    mocker.patch(
        "light_the_torch._pip.extract.PatchedResolverBase.__init__", return_value=None
    )
    resolver = extract.StopAfterPytorchDistsFoundResolver()
    resolver._required_pytorch_dists = ["torchvision==0.7"]
    assert "torch==1.6.0" in resolver.required_pytorch_dists


def test_extract_pytorch_internal_error(mocker):
    mocker.patch("light_the_torch._pip.extract.run")

    with pytest.raises(InternalLTTError):
        ltt.extract_dists(["foo"])


@pytest.mark.slow
def test_extract_dists_ltt():
    assert ltt.extract_dists(["light-the-torch"]) == []


@pytest.mark.slow
def test_extract_dists_pystiche(subtests):
    pystiche = "git+https://github.com/pmeier/pystiche@v{}"
    reqs_and_dists = (
        (pystiche.format("0.4.0"), {"torch>=1.4.0", "torchvision>=0.5.0"}),
        (pystiche.format("0.5.0"), {"torch>=1.5.0", "torchvision>=0.6.0"}),
    )
    for req, dists in reqs_and_dists:
        with subtests.test(req=req):
            assert set(ltt.extract_dists([req])) == dists


@pytest.mark.slow
def test_extract_dists_kornia(subtests):
    kornia = "kornia=={}"
    reqs_and_dists = (
        (kornia.format("0.2.2"), {"torch<=1.4.0,>=1.0.0"}),
        (kornia.format("0.3.1"), {"torch==1.5.0"}),
    )
    for req, dists in reqs_and_dists:
        with subtests.test(req=req):
            assert set(ltt.extract_dists([req])) == dists
