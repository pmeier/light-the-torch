import subprocess

import pytest

from light_the_torch import _cb as cb


class TestComputationBackend:
    def test_eq(self, generic_backend):
        assert generic_backend == generic_backend
        assert generic_backend == generic_backend.local_specifier
        assert generic_backend != 0

    def test_hash_smoke(self, generic_backend):
        assert isinstance(hash(generic_backend), int)

    def test_repr_smoke(self, generic_backend):
        assert isinstance(repr(generic_backend), str)

    def test_from_str_cpu(self):
        string = "cpu"
        backend = cb.ComputationBackend.from_str(string)
        assert isinstance(backend, cb.CPUBackend)

    @pytest.mark.parametrize(
        ("major", "minor", "string"),
        [
            pytest.param(major, minor, string, id=string)
            for major, minor, string in (
                (12, 3, "cu123"),
                (12, 3, "cu12.3"),
                (12, 3, "cuda123"),
                (12, 3, "cuda12.3"),
            )
        ],
    )
    def test_from_str_cuda(self, major, minor, string):
        backend = cb.ComputationBackend.from_str(string)
        assert isinstance(backend, cb.CUDABackend)
        assert backend.major == major
        assert backend.minor == minor

    @pytest.mark.parametrize("string", (("unknown", "cudnn")))
    def test_from_str_unknown(self, string):
        with pytest.raises(cb.ParseError):
            cb.ComputationBackend.from_str(string)


class TestCPUBackend:
    def test_eq(self):
        backend = cb.CPUBackend()
        assert backend == "cpu"


class TestCUDABackend:
    def test_eq(self):
        major = 42
        minor = 21
        backend = cb.CUDABackend(major, minor)
        assert backend == f"cu{major}{minor}"


class TestOrdering:
    def test_cpu(self):
        assert cb.CPUBackend() < cb.CUDABackend(0, 0)

    def test_cuda(self):
        assert cb.CUDABackend(0, 0) > cb.CPUBackend()
        assert cb.CUDABackend(1, 2) < cb.CUDABackend(2, 1)
        assert cb.CUDABackend(2, 1) < cb.CUDABackend(10, 0)


try:
    subprocess.check_call(
        "nvidia-smi",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    NVIDIA_DRIVER_AVAILABLE = True
except subprocess.CalledProcessError:
    NVIDIA_DRIVER_AVAILABLE = False


skip_if_nvidia_driver_unavailable = pytest.mark.skipif(
    not NVIDIA_DRIVER_AVAILABLE, reason="Requires nVidia driver."
)


@pytest.fixture
def patch_nvidia_driver_version(mocker):
    def factory(version):
        return mocker.patch(
            "light_the_torch.computation_backend.subprocess.check_output",
            return_value=f"driver_version\n{version}".encode("utf-8"),
        )

    return factory


def cuda_backends_params():
    params = []
    for system, minimum_driver_versions in cb._MINIMUM_DRIVER_VERSIONS.items():
        cuda_versions, driver_versions = zip(*sorted(minimum_driver_versions.items()))
        cuda_backends = tuple(
            cb.CUDABackend(version.major, version.minor) for version in cuda_versions
        )

        # latest driver supports every backend
        params.append(
            pytest.param(
                system,
                str(driver_versions[-1]),
                set(cuda_backends),
                id=f"{system.lower()}-latest",
            )
        )

        # outdated driver supports no backend
        params.append(
            pytest.param(
                system,
                str(driver_versions[0].major - 1),
                {},
                id=f"{system.lower()}-outdated",
            )
        )

        # "normal" driver supports some backends
        idx = len(cuda_versions) // 2
        params.append(
            pytest.param(
                system,
                str(driver_versions[idx]),
                set(
                    cuda_backends[: idx + 1],
                ),
                id=f"{system.lower()}-normal",
            )
        )

    return pytest.mark.parametrize(
        ("system", "nvidia_driver_version", "compatible_cuda_backends"), params
    )


class TestDetectCompatibleComputationBackends:
    def test_no_nvidia_driver(self, mocker):
        mocker.patch(
            "light_the_torch.computation_backend.subprocess.check_output",
            side_effect=subprocess.CalledProcessError(1, ""),
        )

        assert cb.detect_compatible_computation_backends() == {cb.CPUBackend()}

    @cuda_backends_params()
    def test_cuda_backends(
        self,
        mocker,
        patch_nvidia_driver_version,
        system,
        nvidia_driver_version,
        compatible_cuda_backends,
    ):
        mocker.patch(
            "light_the_torch.computation_backend.platform.system", return_value=system
        )
        patch_nvidia_driver_version(nvidia_driver_version)

        backends = cb.detect_compatible_computation_backends()
        assert backends == {cb.CPUBackend(), *compatible_cuda_backends}

    @skip_if_nvidia_driver_unavailable
    def test_cuda_backend(self):
        backend_types = {
            type(backend) for backend in cb.detect_compatible_computation_backends()
        }
        assert cb.CUDABackend in backend_types
