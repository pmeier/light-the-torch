import argparse
import contextlib
import optparse
import sys
from io import StringIO

import pytest

import light_the_torch
from light_the_torch import cli
from light_the_torch import computation_backend as cb


@contextlib.contextmanager
def exits(code=None, error=False):
    with pytest.raises(SystemExit) as info:
        yield

    ret = info.value.code

    if code is not None:
        assert ret == code

    if error:
        assert ret >= 1
    else:
        assert ret is None or ret == 0


@pytest.fixture
def patch_argv(mocker):
    def patch(*args):
        return mocker.patch.object(sys, "argv", ["tl", *args])

    return patch


def test_main_help_smoke(subtests, mocker, patch_argv):
    for arg in ("-h", "--help"):
        with subtests.test(arg=arg):
            patch_argv(arg)
            stdout = mocker.patch.object(sys, "stdout", StringIO())

            with exits():
                cli.main()

            assert stdout.getvalue().strip()


def test_main_version(subtests, mocker, patch_argv):
    for arg in ("-V", "--version"):
        with subtests.test(arg=arg):
            patch_argv(arg)
            stdout = mocker.patch.object(sys, "stdout", StringIO())

            with exits():
                cli.main()

            out = stdout.getvalue().strip()
            assert out == f"{light_the_torch.__name__}=={light_the_torch.__version__}"


def test_main_no_distributions(mocker, patch_argv):
    patch_argv()
    mocker.patch("ltt.cli.ltt.resolve_dists", return_value=[])
    mocker.patch("ltt.cli.ltt.find_links", side_effect=RuntimeError)

    with exits():
        cli.main()


def test_main_no_install(mocker, patch_argv):
    mocker.patch("ltt.cli.ltt.resolve_dists", return_value=["generic_pytorch_dist"])
    links = [
        "https://download.pytorch.org/foo.whl",
        "https://download.pytorch.org/bar.whl",
    ]
    mocker.patch("ltt.cli.ltt.find_links", return_value=links)
    patch_argv("--no-install", "baz")
    stdout = mocker.patch.object(sys, "stdout", StringIO())

    with exits():
        cli.main()

    out = stdout.getvalue().strip()
    assert "\n".join(links) == out


def test_main_install(subtests, mocker, patch_argv):
    mocker.patch("ltt.cli.ltt.resolve_dists", return_value=["generic_pytorch_idst"])
    links = [
        "https://download.pytorch.org/foo.whl",
        "https://download.pytorch.org/bar.whl",
    ]
    mocker.patch("ltt.cli.ltt.find_links", return_value=links)
    install_cmd = "pip install {links}"
    arg = "baz"
    patch_argv("--install-cmd", install_cmd, arg)

    check_call_mock = mocker.patch("ltt.cli.subprocess.check_call")

    with exits():
        cli.main()

    check_call_mock.assert_called_once()

    cmd = check_call_mock.call_args[0][0]
    assert cmd == install_cmd.format(links=" ".join(links))


def test_main_full_install(subtests, mocker, patch_argv):
    mocker.patch("ltt.cli.ltt.resolve_dists", return_value=["generic_pytorch_dist"])
    links = [
        "https://download.pytorch.org/foo.whl",
        "https://download.pytorch.org/bar.whl",
    ]
    mocker.patch("ltt.cli.ltt.find_links", return_value=links)
    install_cmd = "pip install {links}"
    arg = "baz"
    patch_argv("--full-install", "--install-cmd", install_cmd, arg)

    check_call_mock = mocker.patch("ltt.cli.subprocess.check_call")

    with exits():
        cli.main()

    assert check_call_mock.call_count == 2

    cmd = check_call_mock.call_args[0][0]
    assert cmd == install_cmd.format(links=arg)


def test_parse_args_smoke(patch_argv):
    args = ["--full-install", "--", "--editable", "."]
    patch_argv(*args)
    cli.parse_args()


def test_add_pip_opts_to_install_cmd():
    install_cmd = "pip install {opts} {links}"
    opts = ["--foo", "bar"]

    tl_options = argparse.Namespace(install_cmd=install_cmd)
    pip_install_args_without_opts = ["baz"]
    pip_install_args = [*opts, *pip_install_args_without_opts]

    cli.add_pip_opts_to_install_cmd(
        tl_options, pip_install_args, pip_install_args_without_opts
    )

    assert tl_options.install_cmd == install_cmd.format(
        opts=" ".join(opts), links="{links}"
    )


def test_add_pip_opts_to_install_cmd_no_opts():
    install_cmd = "pip install {links}"
    opts = ["--foo", "bar"]

    tl_options = argparse.Namespace(install_cmd=install_cmd)
    pip_install_args_without_opts = ["baz"]
    pip_install_args = [*opts, *pip_install_args_without_opts]

    cli.add_pip_opts_to_install_cmd(
        tl_options, pip_install_args, pip_install_args_without_opts
    )

    assert tl_options.install_cmd == install_cmd


def test_add_pip_opts_to_install_cmd_disguised_args(subtests):
    install_cmd = "pip install {opts} {links}"
    disguised_args = (
        ["-e", "foo/"],
        ["--editable", "foo/"],
        ["-r", "foo.txt"],
        ["--requirement", "foo.txt"],
    )
    pip_install_args_without_opts = []

    for opts in disguised_args:
        with subtests.test(opts=opts):
            tl_options = argparse.Namespace(install_cmd=install_cmd)
            pip_install_args = [*opts, *pip_install_args_without_opts]

            cli.add_pip_opts_to_install_cmd(
                tl_options, pip_install_args, pip_install_args_without_opts
            )

        assert tl_options.install_cmd == install_cmd.format(opts="", links="{links}")


def test_set_defaults_for_no_install(subtests, mocker):
    format_control = mocker.MagicMock()
    format_control.only_binary = set()
    pip_install_args = optparse.Values(
        {
            "verbose": 0,
            "quiet": 0,
            "target_dir": None,
            "format_control": format_control,
        }
    )
    cli.set_defaults_for_no_install(pip_install_args)

    with subtests.test("verbosity"):
        verbosity = pip_install_args.verbose - pip_install_args.quiet
        assert verbosity < 0

    with subtests.test("target_dir"):
        target_dir = pip_install_args.target_dir
        assert isinstance(target_dir, str)
        assert target_dir

    with subtests.test("format_control"):
        format_control = pip_install_args.format_control
        assert format_control.only_binary == {":all:"}


def test_parse_ltt_args_smoke(subtests):
    args = []
    tl_options, pip_install_args = cli.parse_ltt_args(args)

    names_and_types = (
        ("computation_backend", (type(None), str)),
        ("full_install", bool),
        ("install_cmd", str),
        ("no_install", bool),
    )
    for name, type_ in names_and_types:
        with subtests.test(name):
            assert isinstance(getattr(tl_options, name), type_)

    assert isinstance(pip_install_args, list)


def test_parse_ltt_args_install_cmd_no_links():
    args = ["--install-cmd", "no proper links substitution"]
    with exits(error=True):
        cli.parse_ltt_args(args)


def test_parse_ltt_args_computation_backend(subtests):
    computation_backends = (cb.CPUBackend(), cb.CUDABackend(4, 2))
    for backend in computation_backends:
        with subtests.test(backend):
            args = ["--computation-backend", str(backend)]
            tl_options, _ = cli.parse_ltt_args(args)

            assert tl_options.computation_backend == backend


def test_parse_ltt_args_no_full_install():
    args = ["--full-install", "--no-install"]
    with exits(error=True):
        cli.parse_ltt_args(args)


def test_parse_ltt_args_unrecognized_arguments():
    args = ["--unrecognized-argument"]
    with exits(error=True):
        cli.parse_ltt_args(args)


def test_parse_pip_install_args_smoke(subtests):
    opt_args = ["-f", "https://private.repo.com"]
    pos_args = ["foo", "bar"]
    args = [*opt_args, *pos_args]

    args, pip_install_options = cli.parse_pip_install_args(args)

    with subtests.test("pip_install_options"):
        assert isinstance(pip_install_options, optparse.Values)

    with subtests.test("requirements"):
        assert args == pos_args
