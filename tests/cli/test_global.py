import subprocess

import light_the_torch as ltt
from light_the_torch import cli

from .utils import exits


def test_ltt_main_smoke(subtests):
    for arg in ("-h", "-V"):
        cmd = f"python -m light_the_torch {arg}"
        with subtests.test(cmd=cmd):
            subprocess.check_call(cmd, shell=True)


def test_ltt_help_smoke(subtests, patch_argv, patch_stdout):
    for arg in ("-h", "--help"):
        with subtests.test(arg=arg):
            patch_argv(arg)
            stdout = patch_stdout()

            with exits():
                cli.main()

            assert stdout.getvalue().strip()


def test_ltt_version(subtests, patch_argv, patch_stdout):
    for arg in ("-V", "--version"):
        with subtests.test(arg=arg):
            patch_argv(arg)
            stdout = patch_stdout()

            with exits():
                cli.main()

            output = stdout.getvalue().strip()
            assert output.startswith(f"{ltt.__name__}=={ltt.__version__}")


def test_ltt_unknown_subcommand(patch_argv):
    subcommand = "unkown"
    patch_argv(subcommand)

    with exits(error=True):
        cli.main()
