import contextlib
import dataclasses
import functools
import itertools
import optparse
import os
import sys
import unittest.mock
from typing import List, Set
from unittest import mock

import pip._internal.cli.cmdoptions
from pip._internal.index.package_finder import CandidateEvaluator

import light_the_torch as ltt
from . import _cb as cb
from ._packages import Channel, PatchedPackages
from ._utils import apply_fn_patch


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

    packages = PatchedPackages(options)

    patches = [
        patch_cli_version(),
        patch_cli_options(),
        patch_link_collection(packages),
        patch_candidate_selection(packages),
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
def patch_link_collection(packages):
    @contextlib.contextmanager
    def context(input):
        package = packages.get(input.project_name)
        if not package:
            yield
            return

        with mock.patch.object(input.self, "search_scope", package.make_search_scope()):
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
def patch_candidate_selection(packages):
    def preprocessing(input):
        if not input.candidates:
            return

        # At this stage all candidates have the same name. Thus, if the first is
        # not a PyTorch distribution, we don't need to check the rest and can
        # return without changes.
        package = packages.get(input.candidates[0].name)
        if not package:
            return

        input.candidates = list(package.filter_candidates(input.candidates))

    vanilla_sort_key = CandidateEvaluator._sort_key

    def patched_sort_key(candidate_evaluator, candidate):
        # At this stage all candidates have the same name. Thus, we don't need to
        # mirror the exact key structure that the vanilla sort keys have.
        package = packages.get(candidate.name)
        if not package:
            return vanilla_sort_key(candidate_evaluator, candidate)

        return package.make_sort_key(candidate)

    @contextlib.contextmanager
    def context(input):
        # TODO: refactor this to early return here
        with unittest.mock.patch.object(
            CandidateEvaluator, "_sort_key", new=patched_sort_key
        ):
            yield

    with apply_fn_patch(
        "pip",
        "_internal",
        "index",
        "package_finder",
        "CandidateEvaluator",
        "get_applicable_candidates",
        preprocessing=preprocessing,
        context=context,
    ):
        yield
