import itertools

import pytest

import light_the_torch as ltt
from light_the_torch._pip.common import InternalLTTError
from light_the_torch._pip.find import PytorchCandidatePreferences, maybe_add_option
from light_the_torch.computation_backend import ComputationBackend


@pytest.fixture
def patch_extract_dists(mocker):
    def patch_extract_dists_(return_value=None):
        if return_value is None:
            return_value = []
        return mocker.patch(
            "light_the_torch._pip.find.extract_dists", return_value=return_value
        )

    return patch_extract_dists_


@pytest.fixture
def patch_run(mocker):
    def patch_run_():
        return mocker.patch("light_the_torch._pip.find.run")

    return patch_run_


@pytest.fixture
def computation_backends():
    return ("cpu", "cu92", "cu101", "cu102")


@pytest.fixture
def platforms():
    return ("linux_x86_64", "macosx_10_9_x86_64", "win_amd64")


@pytest.fixture
def python_versions():
    return ("3.6", "3.7", "3.8")


@pytest.fixture
def wheel_properties(computation_backends, platforms, python_versions):
    properties = []
    for properties_ in itertools.product(
        computation_backends, platforms, python_versions
    ):
        # macOS binaries don't support CUDA
        computation_backend, platform, _ = properties_
        if platform.startswith("macosx") and computation_backend != "cpu":
            continue

        properties.append(
            dict(
                zip(("computation_backend", "platform", "python_version"), properties_)
            )
        )
    return tuple(properties)


def test_maybe_add_option_already_set(subtests):
    args = ["--foo", "bar"]
    assert maybe_add_option(args, "--foo",) == args
    assert maybe_add_option(args, "-f", aliases=("--foo",)) == args


def test_PytorchCandidatePreferences_detect_computation_backend(mocker):
    class GenericComputationBackend(ComputationBackend):
        @property
        def local_specifier(self):
            return "generic"

    computation_backend = GenericComputationBackend()
    mocker.patch(
        "light_the_torch._pip.find.detect_computation_backend",
        return_value=computation_backend,
    )

    candidate_prefs = PytorchCandidatePreferences()
    assert candidate_prefs.computation_backend is computation_backend


def test_find_links_internal_error(patch_extract_dists, patch_run):
    patch_extract_dists()
    patch_run()

    with pytest.raises(InternalLTTError):
        ltt.find_links([])


def test_find_links_computation_backend(
    subtests, patch_extract_dists, patch_run, computation_backends
):
    patch_extract_dists()
    run = patch_run()

    for computation_backend in computation_backends:
        with subtests.test(computation_backend=computation_backend):
            run.reset()

            with pytest.raises(InternalLTTError):
                ltt.find_links([], computation_backend=computation_backend)

            args, _ = run.call_args
            cmd = args[0]
            assert cmd.computation_backend == ComputationBackend.from_str(
                computation_backend
            )


def test_find_links_platform(subtests, patch_extract_dists, patch_run, platforms):
    patch_extract_dists()
    run = patch_run()

    for platform in platforms:
        with subtests.test(platform=platform):
            run.reset()

            with pytest.raises(InternalLTTError):
                ltt.find_links([], platform=platform)

            args, _ = run.call_args
            options = args[2]
            assert options.platform == platform


def test_find_links_python_version(
    subtests, patch_extract_dists, patch_run, python_versions
):
    patch_extract_dists()
    run = patch_run()

    for python_version in python_versions:
        python_version_tuple = tuple(int(v) for v in python_version.split("."))
        with subtests.test(python_version=python_version):
            run.reset()

            with pytest.raises(InternalLTTError):
                ltt.find_links([], python_version=python_version)

            args, _ = run.call_args
            options = args[2]
            assert options.python_version == python_version_tuple


@pytest.mark.slow
def test_find_links_torch_smoke(subtests, wheel_properties):
    dist = "torch"

    for properties in wheel_properties:
        with subtests.test(**properties):
            assert ltt.find_links([dist], **properties)


@pytest.mark.slow
def test_find_links_torchaudio_smoke(subtests, wheel_properties):
    dist = "torchaudio"

    for properties in wheel_properties:
        # torchaudio has no published releases for Windows
        if properties["platform"].startswith("win"):
            continue
        with subtests.test(**properties):
            a = ltt.find_links([dist], **properties)
            assert a


@pytest.mark.slow
def test_find_links_torchtext_smoke(subtests, wheel_properties):
    dist = "torchtext"

    for properties in wheel_properties:
        with subtests.test(**properties):
            assert ltt.find_links([dist], **properties)


@pytest.mark.slow
def test_find_links_torchvision_smoke(subtests, wheel_properties):
    dist = "torchvision"

    for properties in wheel_properties:
        with subtests.test(**properties):
            assert ltt.find_links([dist], **properties)
