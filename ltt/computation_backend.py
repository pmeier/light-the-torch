import re
import subprocess
from abc import ABC, abstractmethod
from typing import Any

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


NVCC_RELEASE_PATTERN = re.compile(r"release (?P<major>\d+)[.](?P<minor>\d+)")


def detect_computation_backend() -> ComputationBackend:
    fallback = CPUBackend()
    try:
        output = (
            subprocess.check_output(
                "nvcc --version", shell=True, stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
        match = NVCC_RELEASE_PATTERN.findall(output)
        if not match:
            return fallback

        major, minor = match[0]
        return CUDABackend(int(major), int(minor))
    except subprocess.CalledProcessError:
        return fallback
