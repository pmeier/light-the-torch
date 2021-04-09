import argparse
import sys
from typing import List, Optional, Sequence, Tuple

from .._pip.common import make_pip_install_parser
from .commands import make_command

__all__ = ["main"]


def main(args: Optional[List[str]] = None) -> None:
    args = parse_args(args)
    cmd = make_command(args)

    try:
        pip_install_args = args.args
    except AttributeError:
        pip_install_args = []
    cmd.run(pip_install_args)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    if args is None:
        args = sys.argv[1:]

    parser = make_ltt_parser()
    return parser.parse_args(args)


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
            f"Unrecognized arguments: {', '.join(argv)}. If they were meant as "
            "optional 'pip install' arguments, they have to be passed after a '--' "
            "seperator."
        )
        self.error(message)

    @staticmethod
    def add_verbose(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--verbose",
            action="store_true",
            help=(
                "print more output to STDOUT. For fine control use -v / --verbose and "
                "-q / --quiet of the 'pip install' options"
            ),
        )

    @staticmethod
    def add_pip_install_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "args",
            nargs="*",
            help=(
                "arguments of 'pip install'. Optional arguments have to be seperated "
                "by '--'"
            ),
        )

    @staticmethod
    def add_common_arguments(parser: argparse.ArgumentParser) -> None:
        LTTParser.add_verbose(parser)
        LTTParser.add_pip_install_args(parser)


def make_ltt_parser() -> LTTParser:
    parser = LTTParser(prog="ltt")
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="show light-the-torch version and path and exit",
    )

    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")
    add_ltt_install_parser(subparsers)
    add_ltt_extract_parser(subparsers)
    add_ltt_find_parser(subparsers)

    return parser


SubParsers = argparse._SubParsersAction


def add_ltt_install_parser(subparsers: SubParsers) -> None:
    parser = subparsers.add_parser(
        "install",
        description=(
            "Install PyTorch distributions from the stable releases. The computation "
            "backend  is auto-detected from the available hardware preferring CUDA "
            "over CPU."
        ),
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="disable computation backend auto-detection and use CPU instead",
    )
    parser.add_argument(
        "--pytorch-only",
        action="store_true",
        help="install only PyTorch distributions",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="stable",
        help=(
            "Channel of the PyTorch wheels. "
            "Can be one of 'stable' (default), 'test', or 'nightly'"
        ),
    )
    parser.add_argument(
        "--install-cmd",
        type=str,
        default="python -m pip install {packages}",
        help=(
            "installation command for the PyTorch distributions and additional "
            "packages. Defaults to 'python -m pip install {packages}'"
        ),
    )
    LTTParser.add_common_arguments(parser)


def add_ltt_extract_parser(subparsers: SubParsers) -> None:
    parser = subparsers.add_parser(
        "extract", description="Extract required PyTorch distributions"
    )
    LTTParser.add_common_arguments(parser)


def add_ltt_find_parser(subparsers: SubParsers) -> None:
    parser = subparsers.add_parser(
        "find", description="Find wheel links for the required PyTorch distributions"
    )
    parser.add_argument(
        "--computation-backend",
        type=str,
        help=(
            "Only use wheels compatible with COMPUTATION_BACKEND, for example 'cu102' "
            "or 'cpu'. Defaults to the computation backend of the running system, "
            "preferring CUDA over CPU."
        ),
    )
    parser.add_argument(
        "--channel",
        type=str,
        default="stable",
        help=(
            "Channel of the PyTorch wheels. "
            "Can be one of 'stable' (default), 'test', or 'nightly'"
        ),
    )
    add_pip_install_arguments(parser, "platform", "python_version")
    LTTParser.add_common_arguments(parser)


def add_pip_install_arguments(parser: argparse.ArgumentParser, *dests: str) -> None:
    pip_install_parser = make_pip_install_parser()
    option_group = pip_install_parser.option_groups[0]
    for dest in dests:
        options = [option for option in option_group.option_list if option.dest == dest]
        assert len(options) == 1
        option = options[0]

        parser.add_argument(*option._short_opts, *option._long_opts, help=option.help)
