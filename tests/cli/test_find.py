import pytest

from light_the_torch import cli

from .utils import exits


@pytest.fixture
def patch_find_argv(patch_argv):
    def patch_find_argv_(*args):
        return patch_argv("find", *args)

    return patch_find_argv_


def test_ltt_find(subtests, patch_find_argv, patch_find_links, patch_stdout):
    pip_install_args = ["foo"]
    links = ["bar", "baz"]

    patch_find_argv(*pip_install_args)
    find_links = patch_find_links(links)
    stdout = patch_stdout()

    with exits():
        cli.main()

    with subtests.test("find_links"):
        args, _ = find_links.call_args
        assert args[0] == pip_install_args

    with subtests.test("stdout"):
        output = stdout.getvalue().strip()
        assert output == "\n".join(links)


def test_ltt_find_verbose(patch_find_argv, patch_find_links):
    patch_find_argv("--verbose")
    find_links = patch_find_links([])

    with exits():
        cli.main()

    _, kwargs = find_links.call_args
    assert "verbose" in kwargs
    assert kwargs["verbose"]
