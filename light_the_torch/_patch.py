import contextlib
import dataclasses

import enum
import functools

import optparse
import re
import sys
import unittest.mock
from typing import List, Set
from unittest import mock

import pip._internal.cli.cmdoptions
import pip._internal.index.collector
import pip._internal.index.package_finder
from pip._internal.index.package_finder import CandidateEvaluator
from pip._internal.models.search_scope import SearchScope
from pip._vendor.packaging.version import Version

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
                # TODO: describe multiple inputs
                help=(
                    "Computation backend for compiled PyTorch distributions, "
                    "e.g. 'cu102', 'cu115', or 'cpu'. "
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
            # FIXME add help text
            help="",
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
                cb.ComputationBackend.from_str(string)
                for string in opts.pytorch_computation_backend.split(",")
            }
        elif opts.cpuonly:
            cbs = {cb.CPUBackend()}
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
    # TODO: this template is not valid for all backends
    channel_path = f"{channel.name.lower()}/" if channel != Channel.STABLE else ""
    return [
        f"https://download.pytorch.org/whl/{channel_path}{backend}"
        for backend in sorted(computation_backends)
    ]


@contextlib.contextmanager
def patch_link_collection(computation_backends, channel):
    if channel == channel != Channel.LTS:
        find_links = []
        index_urls = get_extra_index_urls(computation_backends, channel)
    else:
        # TODO: expand this when there are more LTS versions
        # TODO: switch this to index_urls when
        #  https://github.com/pytorch/pytorch/pull/74753 is resolved
        find_links = ["https://download.pytorch.org/whl/lts/1.8/torch_lts.html"]
        index_urls = []

    search_scope = SearchScope.create(find_links=find_links, index_urls=index_urls)

    @contextlib.contextmanager
    def context(input):
        if input.project_name not in PYTORCH_DISTRIBUTIONS:
            yield
            return

        with mock.patch.object(input.self, "search_scope", search_scope):
            yield

    with apply_fn_patch(
        "pip",
        "_internal",
        "index",
        "collector",
        "LinkCollector",
        "collect_sources",
        context=context,
    ):
        yield


@contextlib.contextmanager
def patch_candidate_selection(computation_backends):
    computation_backend_pattern = re.compile(
        r"/(?P<computation_backend>(cpu|cu\d+|rocm([\d.]+)))/"
    )

    def preprocessing(input):
        candidates = []
        for candidate in input.candidates:
            if candidate.name not in PYTORCH_DISTRIBUTIONS:
                # At this stage all candidates have the same name. Thus, if the first is
                # not a PyTorch distribution, we don't need to check the rest and can
                # return without changes.
                return

            if candidate.version.local is None:
                match = computation_backend_pattern.search(candidate.link.path)
                if match:
                    local = match["computation_backend"]
                else:
                    local = "any"
                candidate.version = Version(f"{candidate.version.base_version}+{local}")

            # Early PyTorch distributions used the "any" local specifier to indicate a
            # pure Python binary. This was changed to no local specifier later.
            # Setting this to "cpu" is technically not correct as it will exclude this
            # binary if a non-CPU backend is requested. Still, this is probably the
            # right thing to do, since the user requested a specific backend and
            # although this binary ill work with it, it was not compiled against it.
            if candidate.version.local == "any":
                candidate.version = Version(f"{candidate.version.base_version}+cpu")

            if candidate.version.local not in computation_backends:
                continue

            candidates.append(candidate)

        input.candidates = candidates

    sort_key = CandidateEvaluator._sort_key

    def patched_sort_key(candidate_evaluator, candidate):
        return (
            sort_key(candidate_evaluator, candidate)
            if candidate.name not in PYTORCH_DISTRIBUTIONS
            else (
                cb.ComputationBackend.from_str(candidate.version.local),
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
