import itertools

import pytest
from pip._internal.models.wheel import Wheel
from pip._vendor.packaging.version import Version

import light_the_torch as ltt
import light_the_torch.computation_backend as cb
from light_the_torch._pip.common import InternalLTTError
from light_the_torch._pip.find import maybe_add_option


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


CHANNELS = ("stable", "test", "nightly")
PLATFORMS = ("linux_x86_64", "macosx_10_9_x86_64", "win_amd64")
PLATFORM_MAP = dict(zip(PLATFORMS, ("Linux", "Darwin", "Windows")))


SUPPORTED_PYTHON_VERSIONS = {
    Version("11.1"): tuple(f"3.{minor}" for minor in (6, 7, 8, 9)),
    Version("11.0"): tuple(f"3.{minor}" for minor in (6, 7, 8, 9)),
    Version("10.2"): tuple(f"3.{minor}" for minor in (6, 7, 8, 9)),
    Version("10.1"): tuple(f"3.{minor}" for minor in (6, 7, 8, 9)),
    Version("10.0"): tuple(f"3.{minor}" for minor in (6, 7, 8)),
    Version("9.2"): tuple(f"3.{minor}" for minor in (6, 7, 8, 9)),
    Version("9.1"): tuple(f"3.{minor}" for minor in (6,)),
    Version("9.0"): tuple(f"3.{minor}" for minor in (6, 7)),
    Version("8.0"): tuple(f"3.{minor}" for minor in (6, 7)),
    Version("7.5"): tuple(f"3.{minor}" for minor in (6,)),
}
PYTHON_VERSIONS = set(itertools.chain(*SUPPORTED_PYTHON_VERSIONS.values()))


def test_maybe_add_option_already_set():
    args = ["--foo", "bar"]
    assert maybe_add_option(args, "--foo",) == args
    assert maybe_add_option(args, "-f", aliases=("--foo",)) == args


def test_find_links_internal_error(patch_extract_dists, patch_run):
    patch_extract_dists()
    patch_run()

    with pytest.raises(InternalLTTError):
        ltt.find_links([])


def test_find_links_computation_backend_detect(
    mocker, patch_extract_dists, patch_run, generic_backend
):
    computation_backends = {generic_backend}
    mocker.patch(
        "light_the_torch.computation_backend.detect_compatible_computation_backends",
        return_value=computation_backends,
    )

    patch_extract_dists()
    run = patch_run()

    with pytest.raises(InternalLTTError):
        ltt.find_links([], computation_backends=None)

    args, _ = run.call_args
    cmd = args[0]
    assert cmd.computation_backends == computation_backends


def test_find_links_unknown_channel():
    with pytest.raises(ValueError):
        ltt.find_links([], channel="channel")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_find_links_platform(patch_extract_dists, patch_run, platform):
    patch_extract_dists()
    run = patch_run()

    with pytest.raises(InternalLTTError):
        ltt.find_links([], platform=platform)

    args, _ = run.call_args
    options = args[2]
    assert options.platform == platform


@pytest.mark.parametrize("python_version", PYTHON_VERSIONS)
def test_find_links_python_version(patch_extract_dists, patch_run, python_version):
    patch_extract_dists()
    run = patch_run()

    python_version_tuple = tuple(int(v) for v in python_version.split("."))

    with pytest.raises(InternalLTTError):
        ltt.find_links([], python_version=python_version)

    args, _ = run.call_args
    options = args[2]
    assert options.python_version == python_version_tuple


def wheel_properties():
    params = []
    for platform in PLATFORMS:
        params.extend(
            [
                (platform, cb.CPUBackend(), python_version)
                for python_version in PYTHON_VERSIONS
            ]
        )

        system = PLATFORM_MAP[platform]
        cuda_versions = cb._MINIMUM_DRIVER_VERSIONS.get(system, {}).keys()
        if not cuda_versions:
            continue

        params.extend(
            [
                (
                    platform,
                    cb.CUDABackend(cuda_version.major, cuda_version.minor),
                    python_version,
                )
                for cuda_version in cuda_versions
                for python_version in SUPPORTED_PYTHON_VERSIONS[cuda_version]
                if not (
                    platform == "win_amd64"
                    and (
                        (cuda_version == Version("7.5") and python_version == "3.6")
                        or (cuda_version == Version("9.2") and python_version == "3.9")
                        or (cuda_version == Version("10.0") and python_version == "3.8")
                    )
                )
            ]
        )

    return pytest.mark.parametrize(
        ("platform", "computation_backend", "python_version"), params, ids=str,
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "pytorch_dist", ["torch", "torchaudio", "torchtext", "torchvision"]
)
@wheel_properties()
def test_find_links_stable_smoke(
    pytorch_dist, platform, computation_backend, python_version
):
    assert ltt.find_links(
        [pytorch_dist],
        computation_backends=computation_backend,
        platform=platform,
        python_version=python_version,
    )


@pytest.mark.slow
@pytest.mark.parametrize("channel", CHANNELS)
def test_find_links_channel_smoke(channel):
    assert ltt.find_links(
        ["torch"], computation_backends={cb.CPUBackend()}, channel=channel
    )


@pytest.mark.parametrize("python_version", PYTHON_VERSIONS)
def test_patch_mac_local_specifier_lt_1_0_0(
    patch_extract_dists, patch_run, python_version
):
    # See https://github.com/pmeier/light-the-torch/issues/34
    dists = ["torch"]
    patch_extract_dists(return_value=dists)

    links = ltt.find_links(
        dists, python_version=python_version, platform="macosx_10_9_x86_64"
    )
    version = Version(Wheel(links[0]).version)

    assert version >= Version("1.0.0")
