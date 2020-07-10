import argparse
import optparse
import subprocess
import sys
from copy import copy
from typing import Iterable, List, NoReturn, Optional, Sequence, Tuple

import light_the_torch as ltt

from ._core.common import PatchedInstallCommand
from .computation_backend import ComputationBackend

__all__ = ["main"]


def main(args: Optional[List[str]] = None) -> None:
    args, ltt_options, pip_install_options = parse_args(args=args)

    dists = ltt.resolve_dists(args, options=pip_install_options)

    if not dists:
        exit_ok()

    links = ltt.find_links(
        dists,
        options=pip_install_options,
        computation_backend=ltt_options.computation_backend,
    )

    if ltt_options.no_install:
        print("\n".join(links))
        exit_ok()

    cmd = ltt_options.install_cmd.format(links=" ".join(links))
    subprocess.check_call(cmd, shell=True)

    if not ltt_options.full_install:
        exit_ok()

    cmd = ltt_options.install_cmd.format(links=" ".join(args))
    subprocess.check_call(cmd, shell=True)

    exit_ok()


def exit_ok(code: int = 0) -> NoReturn:
    sys.exit(code)


def exit_error(code: int = 1) -> NoReturn:
    sys.exit(code)


def parse_args(
    args: Optional[List[str]] = None,
) -> Tuple[List[str], argparse.Namespace, optparse.Values]:
    if args is None:
        args = sys.argv[1:]

    ltt_options, pip_install_args = parse_ltt_args(args)
    args, pip_install_options = parse_pip_install_args(pip_install_args)

    add_pip_opts_to_install_cmd(ltt_options, pip_install_args, args)

    if ltt_options.no_install:
        set_defaults_for_no_install(pip_install_options)

    return args, ltt_options, pip_install_options


def parse_ltt_args(args: List[str]) -> Tuple[argparse.Namespace, List[str]]:
    parser = make_ltt_parser()
    options = parser.parse_args(args)

    if options.version:
        print(f"{ltt.__name__}=={ltt.__version__}")
        exit_ok()
    else:
        delattr(options, "version")

    if "{links}" not in options.install_cmd:
        exit_error()

    pip_install_args = options.args
    delattr(options, "args")

    if options.computation_backend is not None:
        options.computation_backend = ComputationBackend.from_str(
            options.computation_backend
        )

    return options, pip_install_args


def parse_pip_install_args(args: List[str]) -> Tuple[List[str], optparse.Values]:
    parser = PatchedInstallCommand().parser
    options, args = parser.parse_args(args)
    return args, options


def _remove_disguised_args(
    opts: List[str], names: Iterable[str], include_next: bool = False
) -> List[str]:
    for name in names:
        try:
            idx = opts.index(name)
            del opts[idx]
            if include_next:
                # After the first del statement the next item in opts is now at idx
                del opts[idx]
            break
        except ValueError:
            continue

    return opts


def add_pip_opts_to_install_cmd(
    ltt_options: argparse.Namespace,
    pip_install_args: List[str],
    pip_install_args_without_opts: List[str],
) -> None:
    install_cmd = ltt_options.install_cmd
    if "{opts}" not in install_cmd:
        return

    # Can't use set subtraction here, since we need to preserve ordering
    opts = copy(pip_install_args)
    for arg in pip_install_args_without_opts:
        opts.remove(arg)
    opts = _remove_disguised_args(opts, ("-e", "--editable"), include_next=True)
    opts = _remove_disguised_args(opts, ("-r", "--requirement"), include_next=True)

    ltt_options.install_cmd = install_cmd.format(opts=" ".join(opts), links="{links}")


def set_defaults_for_no_install(pip_install_args: optparse.Values) -> None:
    # disable pip output for safe piping
    pip_install_args.verbose = 0
    pip_install_args.quiet = 1

    # set required values if --python-version or --platform is used
    if pip_install_args.target_dir is None:
        pip_install_args.target_dir = "."

    if not pip_install_args.format_control.only_binary:
        pip_install_args.format_control.only_binary = {":all:"}


class LTTParser(argparse.ArgumentParser):
    def parse_known_args(
        self,
        args: Optional[Sequence[str]] = None,
        namespace: Optional[argparse.Namespace] = None,
    ) -> Tuple[argparse.Namespace, List[str]]:
        args, argv = super().parse_known_args(args=args, namespace=namespace)
        if not argv:
            return args, argv

        message = (
            f"Unrecognized arguments: {', '.join(argv)}. If they were meant as pip "
            "install arguments, they have to be passed after the '--' seperator."
        )
        self.error(message)


def make_ltt_parser() -> LTTParser:
    parser = LTTParser(
        prog="ltt",
        description=(
            "Install PyTorch distributions from the stable releases. The computation "
            "backend  is autodetected from the available hardware preferring CUDA over "
            "CPU."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-V", "--version", action="store_true", help="show version and exit",
    )
    parser.add_argument(
        "--computation-backend",
        type=str,
        help="pin computation backend, e.g. 'cpu' or 'cu102'",
    )
    install_group = parser.add_mutually_exclusive_group()
    install_group.add_argument(
        "--full-install",
        action="store_true",
        help="install remaining requirements after PyTorch distributions are installed",
    )
    parser.add_argument(
        "--install-cmd",
        type=str,
        default="pip install {opts} {links}",
        help=(
            "installation command. '{links}' is substituted for the links. If present, "
            "'{opts}' is substituted for most additional pip install options. "
            "Exceptions are -e / --editable <path/url> and -r / --requirement <file>"
        ),
    )
    install_group.add_argument(
        "--no-install",
        action="store_true",
        help="print wheel links instead of installing",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help=(
            "arguments passed to pip install. Required PyTorch distributions are "
            "extracted and installed. Optional arguments for pip install have to be "
            "seperated by '--'"
        ),
    )
    return parser
