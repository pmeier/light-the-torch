import contextlib
import dataclasses
import enum
import functools
import itertools
import optparse
import os
import re
import sys
import unittest.mock
from typing import List, Set
from unittest import mock

import pip._internal.cli.cmdoptions

from pip._internal.index.collector import CollectedSources
from pip._internal.index.package_finder import CandidateEvaluator
from pip._internal.index.sources import build_source
from pip._internal.models.search_scope import SearchScope

import light_the_torch as ltt

from . import _cb as cb

from ._utils import apply_fn_patch


class Channel(enum.Enum):
    STABLE = enum.auto()
    TEST = enum.auto()
    NIGHTLY = enum.auto()
    LTS = enum.auto()

    @classmethod
    def from_str(cls, string):
        return cls[string.upper()]


PYTORCH_DISTRIBUTIONS = {
    "torch",
    "torch_model_archiver",
    "torch_tb_profiler",
    "torcharrow",
    "torchaudio",
    "torchcsprng",
    "torchdata",
    "torchdistx",
    "torchserve",
    "torchtext",
    "torchvision",
}


def patch(pip_main):
    @functools.wraps(pip_main)
    def wrapper(argv=None):
        if argv is None:
            argv = sys.argv[1:]

        with apply_patches(argv):
            return pip_main(argv)

    return wrapper


# adapted from https://stackoverflow.com/a/9307174
class PassThroughOptionParser(optparse.OptionParser):
    def __init__(self):
        super().__init__(add_help_option=False)

    def _process_args(self, largs, rargs, values):
        while rargs:
            try:
                super()._process_args(largs, rargs, values)
            except (optparse.BadOptionError, optparse.AmbiguousOptionError) as error:
                largs.append(error.opt_str)


@dataclasses.dataclass
class LttOptions:
    computation_backends: Set[cb.ComputationBackend] = dataclasses.field(
        default_factory=lambda: {cb.CPUBackend()}
    )
    channel: Channel = Channel.STABLE

    @staticmethod
    def computation_backend_parser_options():
        return [
            optparse.Option(
                "--pytorch-computation-backend",
                help=(
                    "Computation backend for compiled PyTorch distributions, "
                    "e.g. 'cu102', 'cu115', or 'cpu'. "
                    "Multiple computation backends can be passed as a comma-separated "
                    "list, e.g 'cu102,cu113,cu116'. "
                    "If not specified, the computation backend is detected from the "
                    "available hardware, preferring CUDA over CPU."
                ),
            ),
            optparse.Option(
                "--cpuonly",
                action="store_true",
                help=(
                    "Shortcut for '--pytorch-computation-backend=cpu'. "
                    "If '--computation-backend' is used simultaneously, "
                    "it takes precedence over '--cpuonly'."
                ),
            ),
        ]

    @staticmethod
    def channel_parser_option() -> optparse.Option:
        return optparse.Option(
            "--pytorch-channel",
            help=(
                "Channel to download PyTorch distributions from, e.g. 'stable' , "
                "'test', 'nightly' and 'lts'. "
                "If not specified, defaults to 'stable' unless '--pre' is given in "
                "which case it defaults to 'test'."
            ),
        )

    @staticmethod
    def _parse(argv):
        parser = PassThroughOptionParser()

        for option in LttOptions.computation_backend_parser_options():
            parser.add_option(option)
        parser.add_option(LttOptions.channel_parser_option())
        parser.add_option("--pre", dest="pre", action="store_true")

        opts, _ = parser.parse_args(argv)
        return opts

    @classmethod
    def from_pip_argv(cls, argv: List[str]):
        if not argv or argv[0] != "install":
            return cls()

        opts = cls._parse(argv)

        if opts.pytorch_computation_backend is not None:
            cbs = {
                cb.ComputationBackend.from_str(string.strip())
                for string in opts.pytorch_computation_backend.split(",")
            }
        elif opts.cpuonly:
            cbs = {cb.CPUBackend()}
        elif "LTT_PYTORCH_COMPUTATION_BACKEND" in os.environ:
            cbs = {
                cb.ComputationBackend.from_str(string.strip())
                for string in os.environ["LTT_PYTORCH_COMPUTATION_BACKEND"].split(",")
            }
        else:
            cbs = cb.detect_compatible_computation_backends()

        if opts.pytorch_channel is not None:
            channel = Channel.from_str(opts.pytorch_channel)
        elif opts.pre:
            channel = Channel.TEST
        else:
            channel = Channel.STABLE

        return cls(cbs, channel)


@contextlib.contextmanager
def apply_patches(argv):
    options = LttOptions.from_pip_argv(argv)

    patches = [
        patch_cli_version(),
        patch_cli_options(),
        patch_link_collection(options.computation_backends, options.channel),
        patch_candidate_selection(options.computation_backends),
    ]

    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)

        yield stack


@contextlib.contextmanager
def patch_cli_version():
    with apply_fn_patch(
        "pip",
        "_internal",
        "cli",
        "main_parser",
        "get_pip_version",
        postprocessing=lambda input, output: f"ltt {ltt.__version__} from {ltt.__path__[0]}\n{output}",
    ):
        yield


@contextlib.contextmanager
def patch_cli_options():
    def postprocessing(input, output):
        for option in LttOptions.computation_backend_parser_options():
            input.cmd_opts.add_option(option)

    index_group = pip._internal.cli.cmdoptions.index_group

    with apply_fn_patch(
        "pip",
        "_internal",
        "cli",
        "cmdoptions",
        "add_target_python_options",
        postprocessing=postprocessing,
    ):
        with unittest.mock.patch.dict(index_group):
            options = index_group["options"].copy()
            options.append(LttOptions.channel_parser_option)
            index_group["options"] = options
            yield


def get_extra_index_urls(computation_backends, channel):
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


@contextlib.contextmanager
def patch_link_collection(computation_backends, channel):
    search_scope = SearchScope(
        find_links=[],
        index_urls=get_extra_index_urls(computation_backends, channel),
        no_index=False,
    )

    @contextlib.contextmanager
    def context(input):
        if input.project_name not in PYTORCH_DISTRIBUTIONS:
            yield
            return

        with mock.patch.object(input.self, "search_scope", search_scope):
            yield

    def postprocessing(input, output):
        if input.project_name not in PYTORCH_DISTRIBUTIONS:
            return output

        if channel != Channel.STABLE:
            return output

        # Some stable binaries are not hosted on the PyTorch indices. We check if this
        # is the case for the current distribution.
        for remote_file_source in output.index_urls:
            candidates = list(remote_file_source.page_candidates())

            # Cache the candidates, so `pip` doesn't has to retrieve them again later.
            remote_file_source.page_candidates = lambda: iter(candidates)

            # If there are any candidates on the PyTorch indices, we continue normally.
            if candidates:
                return output

        # In case the distribution is not present on the PyTorch indices, we fall back
        # to PyPI.
        _, pypi_file_source = build_source(
            SearchScope(
                find_links=[],
                index_urls=["https://pypi.org/simple"],
                no_index=False,
            ).get_index_urls_locations(input.project_name)[0],
            candidates_from_page=input.candidates_from_page,
            page_validator=input.self.session.is_secure_origin,
            expand_dir=False,
            cache_link_parsing=False,
        )

        return CollectedSources(find_links=[], index_urls=[pypi_file_source])

    with apply_fn_patch(
        "pip",
        "_internal",
        "index",
        "collector",
        "LinkCollector",
        "collect_sources",
        context=context,
        postprocessing=postprocessing,
    ):
        yield


@contextlib.contextmanager
def patch_candidate_selection(computation_backends):
    computation_backend_pattern = re.compile(
        r"/(?P<computation_backend>(cpu|cu\d+|rocm([\d.]+)))/"
    )

    def extract_local_specifier(candidate):
        local = candidate.version.local

        if local is None:
            match = computation_backend_pattern.search(candidate.link.path)
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

    def preprocessing(input):
        if not input.candidates:
            return

        candidates = iter(input.candidates)
        candidate = next(candidates)

        if candidate.name not in PYTORCH_DISTRIBUTIONS:
            # At this stage all candidates have the same name. Thus, if the first is
            # not a PyTorch distribution, we don't need to check the rest and can
            # return without changes.
            return

        input.candidates = [
            candidate
            for candidate in itertools.chain([candidate], candidates)
            if extract_local_specifier(candidate) in computation_backends
        ]

    vanilla_sort_key = CandidateEvaluator._sort_key

    def patched_sort_key(candidate_evaluator, candidate):
        # At this stage all candidates have the same name. Thus, we don't need to
        # mirror the exact key structure that the vanilla sort keys have.
        return (
            vanilla_sort_key(candidate_evaluator, candidate)
            if candidate.name not in PYTORCH_DISTRIBUTIONS
            else (
                cb.ComputationBackend.from_str(extract_local_specifier(candidate)),
                candidate.version.base_version,
            )
        )

    with apply_fn_patch(
        "pip",
        "_internal",
        "index",
        "package_finder",
        "CandidateEvaluator",
        "get_applicable_candidates",
        preprocessing=preprocessing,
    ):
        with unittest.mock.patch.object(
            CandidateEvaluator, "_sort_key", new=patched_sort_key
        ):
            yield
