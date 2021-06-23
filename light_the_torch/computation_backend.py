import os
import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any, List, Optional

__all__ = [
    "ComputationBackend",
    "CPUBackend",
    "CUDABackend",
    "detect_computation_backend",
]


class ParseError(ValueError):
    def __init__(self, string: str) -> None:
        super().__init__(f"Unable to parse {string} into a computation backend")


class ComputationBackend(ABC):
    @property
    @abstractmethod
    def local_specifier(self) -> str:
        ...

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


class CUDABackend(ComputationBackend):
    def __init__(self, major: int, minor: int) -> None:
        self.major = major
        self.minor = minor

    @property
    def local_specifier(self) -> str:
        return f"cu{self.major}{self.minor}"


def detect_nvidia_driver() -> Optional[str]:
    driver: Optional[str]
    try:
        output = subprocess.check_output(
            "nvidia-smi --query-gpu=driver_version --format=csv",
            shell=True,
            stderr=subprocess.DEVNULL,
        )
        driver = output.decode("utf-8").splitlines()[-1]
        pattern = re.compile(r"(\d+\.\d+)")  # match at least major and minor
        if not pattern.match(driver):
            driver = None
    except subprocess.CalledProcessError:
        driver = None
    return driver


def get_supported_cuda_version() -> Optional[str]:
    def split(version_string: str) -> List[int]:
        return [int(n) for n in version_string.split(".")]

    nvidia_driver = detect_nvidia_driver()
    if nvidia_driver is None:
        return None

    nvidia_driver = split(nvidia_driver)
    cuda_version = None
    if os.name == "nt":  # windows
        # Table 3 from https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
        if nvidia_driver >= split("456.38"):
            cuda_version = "11.1"
        elif nvidia_driver >= split("451.22"):
            cuda_version = "11.0"
        elif nvidia_driver >= split("441.22"):
            cuda_version = "10.2"
        elif nvidia_driver >= split("418.96"):
            cuda_version = "10.1"
        elif nvidia_driver >= split("398.26"):
            cuda_version = "9.2"
    else:  # linux
        # Table 1 from https://docs.nvidia.com/deploy/cuda-compatibility/index.html
        if nvidia_driver >= split("450.80.02"):
            cuda_version = "11.1"
        elif nvidia_driver >= split("450.36.06"):
            cuda_version = "11.0"
        elif nvidia_driver >= split("440.33"):
            cuda_version = "10.2"
        elif nvidia_driver >= split("418.39"):
            cuda_version = "10.1"
        elif nvidia_driver >= split("396.26"):
            cuda_version = "9.2"
    return cuda_version


def detect_computation_backend() -> ComputationBackend:
    cuda_version = get_supported_cuda_version()
    if cuda_version is None:
        return CPUBackend()
    else:
        major, minor = cuda_version.split(".")
        return CUDABackend(int(major), int(minor))
