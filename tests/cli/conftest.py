import io
import sys

import pytest


@pytest.fixture
def patch_argv(mocker):
    def patch_argv_(*args):
        return mocker.patch.object(sys, "argv", ["ltt", *args])

    return patch_argv_


@pytest.fixture
def patch_extract_dists(mocker):
    def patch_extract_dists_(return_value=None):
        if return_value is None:
            return_value = []
        return mocker.patch(
            "light_the_torch.cli.commands.ltt.extract_dists",
            return_value=return_value,
        )

    return patch_extract_dists_


@pytest.fixture
def patch_find_links(mocker):
    def patch_find_links_(return_value=None):
        if return_value is None:
            return_value = []
        return mocker.patch(
            "light_the_torch.cli.commands.ltt.find_links",
            return_value=return_value,
        )

    return patch_find_links_


@pytest.fixture
def patch_stdout(mocker):
    def patch_stdout_():
        return mocker.patch.object(sys, "stdout", io.StringIO())

    return patch_stdout_
