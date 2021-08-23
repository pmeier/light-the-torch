import platform
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Optional, Set

from pip._vendor.packaging.version import InvalidVersion, Version

__all__ = [
    "ComputationBackend",
    "CPUBackend",
    "CUDABackend",
    "detect_compatible_computation_backends",
]


class ParseError(ValueError):
    def __init__(self, string: str) -> None:
        super().__init__(f"Unable to parse {string} into a computation backend")


class ComputationBackend(ABC):
    @property
    @abstractmethod
    def local_specifier(self) -> str:
        pass

    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        pass

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ComputationBackend):
            return self.local_specifier == other.local_specifier
        elif isinstance(other, str):
            return self.local_specifier == other
        else:
            return False

    def __hash__(self) -> int:
        return hash(self.local_specifier)

    def __repr__(self) -> str:
        return self.local_specifier

    @classmethod
    def from_str(cls, string: str) -> "ComputationBackend":
        parse_error = ParseError(string)
        string = string.lower()
        if string == "cpu":
            return CPUBackend()
        elif string.startswith("cu"):
            match = re.match(r"^cu(da)?(?P<version>[\d.]+)$", string)
            if match is None:
                raise parse_error

            version = match.group("version")
            if "." in version:
                major, minor = version.split(".")
            else:
                major = version[:-1]
                minor = version[-1]

            return CUDABackend(int(major), int(minor))
        else:
            raise parse_error


class CPUBackend(ComputationBackend):
    @property
    def local_specifier(self) -> str:
        return "cpu"

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, ComputationBackend):
            return NotImplemented

        return True


class CUDABackend(ComputationBackend):
    def __init__(self, major: int, minor: int) -> None:
        self.major = major
        self.minor = minor

    @property
    def local_specifier(self) -> str:
        return f"cu{self.major}{self.minor}"

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, CPUBackend):
            return False
        elif not isinstance(other, CUDABackend):
            return NotImplemented

        return (self.major, self.minor) < (other.major, other.minor)


def _detect_nvidia_driver_version() -> Optional[Version]:
    cmd = "nvidia-smi --query-gpu=driver_version --format=csv"
    try:
        output = (
            subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        return Version(output.splitlines()[-1])
    except (subprocess.CalledProcessError, InvalidVersion):
        return None


# Table 3 from https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
_MINIMUM_DRIVER_VERSIONS = {
    "Linux": {
        Version("11.1"): Version("455.32"),
        Version("11.0"): Version("450.51.06"),
        Version("10.2"): Version("440.33"),
        Version("10.1"): Version("418.39"),
        Version("10.0"): Version("410.48"),
        Version("9.2"): Version("396.26"),
        Version("9.1"): Version("390.46"),
        Version("9.0"): Version("384.81"),
        Version("8.0"): Version("375.26"),
        Version("7.5"): Version("352.31"),
    },
    "Windows": {
        Version("11.1"): Version("456.81"),
        Version("11.0"): Version("451.82"),
        Version("10.2"): Version("441.22"),
        Version("10.1"): Version("418.96"),
        Version("10.0"): Version("411.31"),
        Version("9.2"): Version("398.26"),
        Version("9.1"): Version("391.29"),
        Version("9.0"): Version("385.54"),
        Version("8.0"): Version("376.51"),
        Version("7.5"): Version("353.66"),
    },
}


def _detect_compatible_cuda_backends() -> Set[CUDABackend]:
    driver_version = _detect_nvidia_driver_version()
    if not driver_version:
        return set()

    minimum_driver_versions = _MINIMUM_DRIVER_VERSIONS.get(platform.system())
    if not minimum_driver_versions:
        return set()

    return {
        CUDABackend(cuda_version.major, cuda_version.minor)
        for cuda_version, minimum_driver_version in minimum_driver_versions.items()
        if driver_version >= minimum_driver_version
    }


def detect_compatible_computation_backends() -> Set[ComputationBackend]:
    return {CPUBackend(), *_detect_compatible_cuda_backends()}
