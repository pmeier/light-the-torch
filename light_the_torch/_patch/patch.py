import contextlib
import functools
import sys
import unittest.mock
from unittest import mock

import pip._internal.cli.cmdoptions
from pip._internal.index.package_finder import CandidateEvaluator

import light_the_torch as ltt
from .cli import LttOptions
from .packages import packages
from .utils import apply_fn_patch


def patch_pip_main(pip_main):
    @functools.wraps(pip_main)
    def wrapper(argv=None):
        if argv is None:
            argv = sys.argv[1:]

        with apply_patches(argv):
            return pip_main(argv)

    return wrapper


@contextlib.contextmanager
def apply_patches(argv):
    options = LttOptions.from_pip_argv(argv)

    patches = [
        patch_cli_version(),
        patch_cli_options(),
        patch_link_collection(packages, options),
        patch_candidate_selection(packages, options),
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


@contextlib.contextmanager
def patch_link_collection(packages, options):
    @contextlib.contextmanager
    def context(input):
        package = packages.get(input.project_name)
        if not package:
            yield
            return

        with mock.patch.object(
            input.self, "search_scope", package.make_search_scope(options)
        ):
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
def patch_candidate_selection(packages, options):
    def preprocessing(input):
        if not input.candidates:
            return

        # At this stage all candidates have the same name. Thus, if the first is
        # not a PyTorch distribution, we don't need to check the rest and can
        # return without changes.
        package = packages.get(input.candidates[0].name)
        if not package:
            return

        input.candidates = list(package.filter_candidates(input.candidates, options))

    def patched_sort_key(candidate_evaluator, candidate):
        package = packages.get(candidate.name)
        assert package
        return package.make_sort_key(candidate, options)

    @contextlib.contextmanager
    def context(input):
        # At this stage all candidates have the same name. Thus, we don't need to
        # mirror the exact key structure that the vanilla sort keys have.
        if not input.candidates or input.candidates[0].name not in packages:
            yield
            return

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
