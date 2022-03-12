import contextlib
import functools
import optparse
import re
import sys
import unittest.mock
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    cast,
)
from unittest import mock
from urllib.parse import urljoin

import pip._internal.cli.cmdoptions
import pip._internal.index.collector
import pip._internal.index.package_finder
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import CandidateEvaluator
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope

from ._utils import (  # extract_ltt_options,
    Channel,
    LttOptions,
    apply_fn_patch,
    channel_option,
    computation_backend_options,
)
from .computation_backend import ComputationBackend

__all__ = ["patch"]

# FLATTEN EVERYTHING

PATCHED_SUB_CMDS = ("install", "uninstall")
PYTORCH_DISTRIBUTIONS = ("torch", "torchvision", "torchaudio", "torchtext")


def patch(pip_main):
    @functools.wraps(pip_main)
    def shim(argv: Optional[List[str]] = None):
        if argv is None:
            argv = sys.argv[1:]

        with (apply_patches if argv[0] == "install" else contextlib.nullcontext)(argv):
            return pip_main(argv)

    return shim


# TODO: apply install patches


@contextlib.contextmanager
def apply_patches(argv: List[str]) -> Iterator[contextlib.ExitStack]:
    options = LttOptions.from_pip_argv(argv)

    patches = [
        patch_cli_computation_backend_options(),
        patch_cli_channel_option(),
        patch_link_collection(options.computation_backends, options.channel),
        patch_link_evaluation(),
        patch_candidate_selection(options.computation_backends),
    ]

    with contextlib.ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)

        yield stack


@contextlib.contextmanager
def patch_cli_computation_backend_options() -> Iterator[None]:
    def postprocessing(input, output) -> None:
        for option in computation_backend_options():
            input.cmd_opts.add_option(option)

    with apply_fn_patch(
        "pip._internal.cli.cmdoptions.add_target_python_options",
        postprocessing=postprocessing,
    ):
        yield


@contextlib.contextmanager
def patch_cli_channel_option() -> Iterator[None]:
    index_group = pip._internal.cli.cmdoptions.index_group
    with unittest.mock.patch.dict(index_group):
        options = index_group["options"].copy()
        options.append(channel_option)
        index_group["options"] = options
        yield


@contextlib.contextmanager
def patch_link_collection(
    computation_backends: Collection[ComputationBackend], channel: Channel
) -> Iterator[None]:
    if channel == Channel.STABLE:
        urls = ["https://download.pytorch.org/whl/torch_stable.html"]
    elif channel == Channel.LTS:
        urls = ["https://download.pytorch.org/whl/lts/1.8/torch_lts.html"]
    else:
        urls = [
            f"https://download.pytorch.org/whl/"
            f"{channel.name.lower()}/{backend}/torch_{channel.name.lower()}.html"
            for backend in sorted(computation_backends)
        ]
    search_scope = SearchScope.create(find_links=urls, index_urls=[])

    @contextlib.contextmanager
    def context(input):
        if input.project_name not in PYTORCH_DISTRIBUTIONS:
            return

        with mock.patch.object(input.self, "search_scope", search_scope):
            yield

    with apply_fn_patch(
        "pip._internal.index.collector.LinkCollector.collect_sources", context=context
    ):
        yield


@contextlib.contextmanager
def patch_link_evaluation() -> Iterator[None]:
    HAS_LOCAL_PATTERN = re.compile(r"[+](cpu|cu\d+)$")
    COMPUTATION_BACKEND_PATTERN = re.compile(
        r"^/whl/(?P<computation_backend>(cpu|cu\d+))/"
    )

    def postprocessing(input, output: Tuple[bool, Optional[str]]):
        is_candidate, result = output
        if not is_candidate:
            return output

        has_local = HAS_LOCAL_PATTERN.search(result) is not None
        if has_local:
            return output

        computation_backend = COMPUTATION_BACKEND_PATTERN.match(input.link.path)
        if computation_backend:
            local = computation_backend["computation_backend"]
        else:
            local = "any"

        return True, f"{result}+{local}"

    with apply_fn_patch(
        "pip._internal.index.package_finder.LinkEvaluator.evaluate_link",
        postprocessing=postprocessing,
    ):
        yield


@contextlib.contextmanager
def patch_candidate_selection(
    computation_backends: Collection[ComputationBackend],
) -> Iterator[None]:
    allowed_locals = {None, *computation_backends}

    def postprocessing(
        input, output: List[InstallationCandidate]
    ) -> List[InstallationCandidate]:
        return [
            candidate
            for candidate in output
            if candidate.name not in PYTORCH_DISTRIBUTIONS
            or candidate.version.local in allowed_locals
        ]

    foo = CandidateEvaluator._sort_key

    def sort_key(candidate_evaluator, candidate):
        if candidate.name not in PYTORCH_DISTRIBUTIONS:
            return foo(candidate_evaluator, candidate)

        return (
            ComputationBackend.from_str(candidate.version.local.replace("any", "cpu")),
            candidate.version,
        )

    with apply_fn_patch(
        "pip._internal.index.package_finder.CandidateEvaluator.get_applicable_candidates",
        postprocessing=postprocessing,
    ):
        with unittest.mock.patch.object(CandidateEvaluator, "_sort_key", new=sort_key):
            yield
