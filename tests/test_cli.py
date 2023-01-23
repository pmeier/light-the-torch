import contextlib
import io
import shlex
import subprocess
import sys

import light_the_torch as ltt
import pytest

from light_the_torch._cli import main
from light_the_torch._patch import Channel


@pytest.mark.parametrize("cmd", ["ltt", "python -m light_the_torch"])
def test_entry_point_smoke(cmd):
    subprocess.run(shlex.split(cmd), shell=False)


@contextlib.contextmanager
def exits(*, should_succeed=True, expected_code=None, check_err=None, check_out=None):
    def parse_checker(checker):
        if checker is None or callable(checker):
            return checker

        if isinstance(checker, str):
            checker = (checker,)

        def check_fn(text):
            for phrase in checker:
                assert phrase in text

        return check_fn

    check_err = parse_checker(check_err)
    check_out = parse_checker(check_out)

    with pytest.raises(SystemExit) as info:
        with contextlib.redirect_stderr(io.StringIO()) as raw_err:
            with contextlib.redirect_stdout(io.StringIO()) as raw_out:
                yield

    returned_code = info.value.code or 0
    succeeded = returned_code == 0
    err = raw_err.getvalue().strip()
    out = raw_out.getvalue().strip()

    if expected_code is not None:
        if returned_code == expected_code:
            return

        raise AssertionError(
            f"Returned and expected return code mismatch: "
            f"{returned_code} != {expected_code}."
        )

    if should_succeed:
        if succeeded:
            if check_out:
                check_out(out)

            return

        raise AssertionError(
            f"Program should have succeeded, but returned code {returned_code} "
            f"and printed the following to STDERR: '{err}'."
        )
    else:
        if not succeeded:
            if check_err:
                check_err(err)

            return

        raise AssertionError("Program shouldn't have succeeded, but did.")


@pytest.fixture
def set_argv(mocker):
    def patch(*options):
        return mocker.patch_pip_main.object(sys, "argv", ["ltt", *options])

    return patch


@pytest.mark.parametrize("option", ["-h", "--help"])
def test_help_smoke(set_argv, option):
    set_argv(option)

    def check_out(out):
        assert out

    with exits(check_out=check_out):
        main()


@pytest.mark.parametrize("option", ["-V", "--version"])
def test_version(set_argv, option):
    set_argv(option)

    with exits(check_out=f"ltt {ltt.__version__} from {ltt.__path__[0]}"):
        main()


@pytest.mark.parametrize(
    "option",
    [
        "--pytorch-computation-backend",
        "--cpuonly",
        "--pytorch-channel",
    ],
)
def test_ltt_options_smoke(set_argv, option):
    set_argv("install", "--help")

    with exits(check_out=option):
        main()


def test_pytorch_channel_values(set_argv):
    set_argv("install", "--help")

    with exits(check_out=[f"'{channel.name.lower()}'" for channel in Channel]):
        main()
