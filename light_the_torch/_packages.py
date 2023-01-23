import abc
import dataclasses
import enum
import itertools
import re

from pip._internal.models.search_scope import SearchScope

from . import _cb as cb

__all__ = ["packages"]


@dataclasses.dataclass
class _Package(abc.ABC):
    name: str

    @abc.abstractmethod
    def make_search_scope(self, options):
        pass

    @abc.abstractmethod
    def filter_candidates(self, candidates, options):
        pass

    @abc.abstractmethod
    def make_sort_key(self, candidate, options):
        pass


# FIXME: move this to cli patch
#  create patch.cli and patch.packages
class Channel(enum.Enum):
    STABLE = enum.auto()
    TEST = enum.auto()
    NIGHTLY = enum.auto()
    LTS = enum.auto()

    @classmethod
    def from_str(cls, string):
        return cls[string.upper()]


packages = {}


class _PyTorchDistribution(_Package):
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

    def make_search_scope(self, options):
        return SearchScope(
            find_links=[],
            index_urls=self._get_extra_index_urls(
                options.computation_backends, options.channel
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

    def filter_candidates(self, candidates, options):
        return [
            candidate
            for candidate in candidates
            if self._extract_local_specifier(candidate) in options.computation_backends
        ]

    def make_sort_key(self, candidate, options):
        return (
            cb.ComputationBackend.from_str(self._extract_local_specifier(candidate)),
            candidate.version.base_version,
        )


# FIXME: check whether all of these are hosted on all channels
#  If not, change `_TorchData` below to a more general class
# FIXME: check if they are valid at all
for name in {
    "torch",
    "torch_model_archiver",
    "torch_tb_profiler",
    "torcharrow",
    "torchaudio",
    "torchcsprng",
    "torchdistx",
    "torchserve",
    "torchtext",
    "torchvision",
}:
    packages[name] = _PyTorchDistribution(name)


class _TorchData(_PyTorchDistribution):
    def make_search_scope(self, options):
        if options.channel == Channel.STABLE:
            return SearchScope(
                find_links=[],
                index_urls=["https://pypi.org/simple"],
                no_index=False,
            )

        return super().make_search_scope(options)
