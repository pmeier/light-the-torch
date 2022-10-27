import abc
import enum
import itertools
import re

from pip._internal.models.search_scope import SearchScope

from . import _cb as cb


class Channel(enum.Enum):
    STABLE = enum.auto()
    TEST = enum.auto()
    NIGHTLY = enum.auto()
    LTS = enum.auto()

    @classmethod
    def from_str(cls, string):
        return cls[string.upper()]


class PatchedPackages:
    _PATCHED_PACKAGE_CLSS_MAP = {}

    @classmethod
    def _register(cls, name):
        def wrapper(patched_package_cls):
            cls._PATCHED_PACKAGE_CLSS_MAP[name] = patched_package_cls
            return patched_package_cls

        return wrapper

    def __init__(self, options):
        self._options = options
        self._patched_packages_map = {
            name: cls(options) for name, cls in self._PATCHED_PACKAGE_CLSS_MAP.items()
        }

    def get(self, name):
        return self._patched_packages_map.get(name)


class _PatchedPackage(abc.ABC):
    def __init__(self, options):
        self._options = options

    @abc.abstractmethod
    def make_search_scope(self):
        pass

    @abc.abstractmethod
    def filter_candidates(self, candidates):
        pass

    @abc.abstractmethod
    def make_sort_key(self, candidate):
        pass


class _PatchedPyTorchPackage(_PatchedPackage):
    def _get_extra_index_urls(self, computation_backends, channel):
        if channel == Channel.STABLE:
            channel_paths = [""]
        elif channel == Channel.LTS:
            channel_paths = [
                f"lts/{major}.{minor}/"
                for major, minor in [
                    (1, 8),
                ]
            ]
        else:
            channel_paths = [f"{channel.name.lower()}/"]
        return [
            f"https://download.pytorch.org/whl/{channel_path}{backend}"
            for channel_path, backend in itertools.product(
                channel_paths, sorted(computation_backends)
            )
        ]

    def make_search_scope(self):
        return SearchScope(
            find_links=[],
            index_urls=self._get_extra_index_urls(
                self._options.computation_backends, self._options.channel
            ),
            no_index=False,
        )

    _COMPUTATION_BACKEND_PATTERN = re.compile(
        r"/(?P<computation_backend>(cpu|cu\d+|rocm([\d.]+)))/"
    )

    def _extract_local_specifier(self, candidate):
        local = candidate.version.local

        if local is None:
            match = self._COMPUTATION_BACKEND_PATTERN.search(candidate.link.path)
            local = match["computation_backend"] if match else "any"

        # Early PyTorch distributions used the "any" local specifier to indicate a
        # pure Python binary. This was changed to no local specifier later.
        # Setting this to "cpu" is technically not correct as it will exclude this
        # binary if a non-CPU backend is requested. Still, this is probably the
        # right thing to do, since the user requested a specific backend and
        # although this binary will work with it, it was not compiled against it.
        if local == "any":
            local = "cpu"

        return local

    def filter_candidates(self, candidates):
        return [
            candidate
            for candidate in candidates
            if self._extract_local_specifier(candidate)
            in self._options.computation_backends
        ]

    def make_sort_key(self, candidate):
        return (
            cb.ComputationBackend.from_str(self._extract_local_specifier(candidate)),
            candidate.version.base_version,
        )


for name in ["torch", "torchvision", "torchaudio"]:
    PatchedPackages._register(name)(_PatchedPyTorchPackage)


@PatchedPackages._register("torchdata")
class _TorchData(_PatchedPyTorchPackage):
    def make_search_scope(self):
        if self._options.channel == Channel.STABLE:
            return SearchScope(
                find_links=[],
                index_urls=["https://pypi.org/simple"],
                no_index=False,
            )

        return super().make_search_scope()
