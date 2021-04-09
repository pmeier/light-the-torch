from typing import Any, Optional

__all__ = ["find_compatible_torch_version"]


class Version:
    @classmethod
    def from_str(cls, version: str) -> "Version":
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else None
        patch = int(parts[2]) if len(parts) > 2 else None
        return cls(major, minor, patch)

    def __init__(self, major: int, minor: Optional[int], patch: Optional[int]) -> None:
        self.major = major
        self.minor = minor
        self.patch = patch
        self.parts = (major, minor, patch)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Version):
            return False

        return all(
            [
                self_part == other_part
                for self_part, other_part in zip(self.parts, other.parts)
                if self_part is not None and other_part is not None
            ]
        )

    def __hash__(self) -> int:
        return hash(self.parts)

    def __repr__(self) -> str:
        return ".".join([str(part) for part in self.parts if part is not None])


COMPATIBILITY = {
    "torchvision": {
        Version(0, 9, 1): Version(1, 8, 1),
        Version(0, 9, 0): Version(1, 8, 0),
        Version(0, 8, 0): Version(1, 7, 0),
        Version(0, 7, 0): Version(1, 6, 0),
        Version(0, 6, 1): Version(1, 5, 1),
        Version(0, 6, 0): Version(1, 5, 0),
        Version(0, 5, 0): Version(1, 4, 0),
        Version(0, 4, 2): Version(1, 3, 1),
        Version(0, 4, 1): Version(1, 3, 0),
        Version(0, 4, 0): Version(1, 2, 0),
        Version(0, 3, 0): Version(1, 1, 0),
        Version(0, 2, 2): Version(1, 0, 1),
    }
}


def find_compatible_torch_version(dist: str, version: str) -> str:
    version = Version.from_str(version)
    dist_compatibility = COMPATIBILITY[dist]
    candidates = [x for x in dist_compatibility.keys() if x == version]
    if not candidates:
        raise RuntimeError(
            f"No compatible torch version was found for {dist}=={version}"
        )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Multiple compatible torch versions were found for {dist}=={version}:\n"
            f"{', '.join([str(candidate) for candidate in candidates])}\n"
        )

    return str(dist_compatibility[candidates[0]])
