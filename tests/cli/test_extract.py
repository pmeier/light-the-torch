import pytest

from light_the_torch import cli

from .utils import exits


@pytest.fixture
def patch_extract_argv(patch_argv):
    def patch_extract_argv_(*args):
        return patch_argv("extract", *args)

    return patch_extract_argv_


def test_ltt_extract(
    subtests, patch_extract_argv, patch_extract_pytorch_dists, patch_stdout
):
    pip_install_args = ["foo"]
    dists = ["bar", "baz"]

    patch_extract_argv(*pip_install_args)
    extract_pytorch_dists = patch_extract_pytorch_dists(dists)
    stdout = patch_stdout()

    with exits():
        cli.main()

    with subtests.test("extract_pytorch_dists"):
        args, _ = extract_pytorch_dists.call_args
        assert args[0] == pip_install_args

    with subtests.test("stdout"):
        output = stdout.getvalue().strip()
        assert output == "\n".join(dists)


def test_ltt_extract_verbose(patch_extract_argv, patch_extract_pytorch_dists):
    patch_extract_argv("--verbose")
    extract_pytorch_dists = patch_extract_pytorch_dists([])

    with exits():
        cli.main()

    _, kwargs = extract_pytorch_dists.call_args
    assert "verbose" in kwargs
    assert kwargs["verbose"]
