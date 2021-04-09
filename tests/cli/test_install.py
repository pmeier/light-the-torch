import pytest

from light_the_torch import cli
from light_the_torch.computation_backend import CPUBackend

from .utils import exits


@pytest.fixture
def patch_install_argv(patch_argv):
    def patch_install_argv_(*args):
        return patch_argv("install", *args)

    return patch_install_argv_


@pytest.fixture
def patch_subprocess_call(mocker):
    def patch_subprocess_call_():
        return mocker.patch("light_the_torch.cli.commands.subprocess.check_call")

    return patch_subprocess_call_


@pytest.fixture
def patch_collect_packages(mocker):
    def patch_collect_packages_(return_value=None):
        if return_value is None:
            return_value = []
        return mocker.patch(
            "light_the_torch.cli.commands.InstallCommand.collect_packages",
            return_value=return_value,
        )

    return patch_collect_packages_


def test_ltt_install(
    subtests, patch_install_argv, patch_find_links, patch_subprocess_call
):
    install_cmd = "python -m pip install {packages}"
    pip_install_args = ["foo", "bar"]
    links = ["https://foo.org"]

    patch_install_argv(*pip_install_args)
    patch_find_links(links)
    subprocess_call = patch_subprocess_call()

    with exits():
        cli.main()

    assert subprocess_call.call_count == 2

    with subtests.test("install PyTorch"):
        call_args = subprocess_call.call_args_list[0]
        args, _ = call_args
        assert args[0] == install_cmd.format(packages=" ".join(links))

    with subtests.test("install remainder"):
        call_args = subprocess_call.call_args_list[1]
        args, _ = call_args
        assert args[0] == install_cmd.format(packages=" ".join(pip_install_args))


def test_ltt_install_force_cpu(
    patch_install_argv, patch_find_links, patch_subprocess_call, patch_collect_packages,
):
    patch_install_argv("--force-cpu")
    find_links = patch_find_links()
    patch_subprocess_call()
    patch_collect_packages()

    with exits():
        cli.main()

    _, kwargs = find_links.call_args
    assert "computation_backend" in kwargs
    assert kwargs["computation_backend"] == CPUBackend()


def test_ltt_install_pytorch_only(
    patch_install_argv, patch_find_links, patch_subprocess_call, patch_collect_packages,
):
    patch_install_argv("--pytorch-only")
    patch_find_links()
    patch_subprocess_call()
    collect_packages = patch_collect_packages()

    with exits():
        cli.main()

    collect_packages.assert_not_called()


def test_ltt_install_channel(
    patch_install_argv, patch_find_links, patch_subprocess_call, patch_collect_packages,
):
    channel = "channel"

    patch_install_argv(f"--channel={channel}")
    find_links = patch_find_links()
    patch_subprocess_call()
    patch_collect_packages()

    with exits():
        cli.main()

    _, kwargs = find_links.call_args
    assert "channel" in kwargs
    assert kwargs["channel"] == channel


def test_ltt_install_install_cmd(
    patch_install_argv, patch_find_links, patch_subprocess_call,
):
    install_cmd = "custom install {packages}"
    packages = ["foo", "bar"]

    patch_install_argv("--pytorch-only", "--install-cmd", install_cmd)
    patch_find_links(packages)
    subprocess_call = patch_subprocess_call()

    with exits():
        cli.main()

    args, _ = subprocess_call.call_args
    assert args[0] == install_cmd.format(packages=" ".join(packages))


def test_ltt_install_install_cmd_no_subs(patch_install_argv):
    patch_install_argv("--install-cmd", "no proper packages substitution")

    with exits(error=True):
        cli.main()


def test_ltt_install_editables(
    patch_install_argv, patch_find_links, patch_subprocess_call,
):
    install_cmd = "custom install {packages}"
    editables = [".", "foo"]
    args = ["bar", "baz"]
    cmd = install_cmd.format(packages=" ".join([f"-e {e}" for e in editables] + args))

    patch_install_argv(
        "--install-cmd",
        install_cmd,
        "--",
        "-e",
        editables[0],
        "--editable",
        editables[1],
        *args,
    )
    patch_find_links()
    subprocess_call = patch_subprocess_call()

    with exits():
        cli.main()

    args, _ = subprocess_call.call_args
    assert args[0] == cmd


def test_ltt_install_requirements(
    patch_install_argv, patch_find_links, patch_subprocess_call,
):
    install_cmd = "custom install {packages}"
    requirements = ["requirements.txt", "requirements-dev.txt"]
    args = ["foo", "bar"]
    cmd = install_cmd.format(
        packages=" ".join([f"-r {r}" for r in requirements] + args)
    )

    patch_install_argv(
        "--install-cmd",
        install_cmd,
        "--",
        "-r",
        requirements[0],
        "--requirement",
        requirements[1],
        *args,
    )
    patch_find_links()
    subprocess_call = patch_subprocess_call()

    with exits():
        cli.main()

    args, _ = subprocess_call.call_args
    assert args[0] == cmd


def test_ltt_install_verbose(
    patch_install_argv, patch_find_links, patch_subprocess_call, patch_collect_packages,
):
    patch_install_argv("--verbose")
    find_links = patch_find_links()
    patch_subprocess_call()
    patch_collect_packages()

    with exits():
        cli.main()

    _, kwargs = find_links.call_args
    assert "verbose" in kwargs
    assert kwargs["verbose"]


def test_install_unrecognized_argument(patch_install_argv):
    patch_install_argv("--unrecognized-argument")

    with exits(error=True):
        cli.main()
