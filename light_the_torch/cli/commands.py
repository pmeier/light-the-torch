import argparse
import subprocess
import sys
import warnings
from abc import ABC, abstractmethod
from os import path
from typing import Dict, List, NoReturn, Optional, Type

import light_the_torch as ltt

from .._pip.common import make_pip_install_parser
from ..computation_backend import CPUBackend

__all__ = ["make_command"]


class Command(ABC):
    @abstractmethod
    def __init__(self, args: argparse.Namespace) -> None:
        pass

    def run(self, pip_install_args: List[str]) -> None:
        self._run(pip_install_args)
        self.exit()

    @abstractmethod
    def _run(self, pip_install_args: List[str]) -> None:
        pass

    def exit(self, code: Optional[int] = None, error: bool = False) -> NoReturn:
        if code is None:
            code = 1 if error else 0
        sys.exit(code)


class GlobalCommand(Command):
    def __init__(self, args: argparse.Namespace) -> None:
        self.version = args.version

    def _run(self, pip_install_args: List[str]) -> None:
        if self.version:
            root = path.abspath(path.join(path.dirname(__file__), ".."))
            print(f"{ltt.__name__}=={ltt.__version__} from {root}")


class ExtractCommand(Command):
    def __init__(self, args: argparse.Namespace) -> None:
        self.verbose = args.verbose

    def _run(self, pip_install_args: List[str]) -> None:
        dists = ltt.extract_dists(pip_install_args, verbose=self.verbose)
        print("\n".join(dists))


class FindCommand(Command):
    def __init__(self, args: argparse.Namespace) -> None:
        # TODO split by comma
        self.computation_backends = args.computation_backend
        self.channel = args.channel
        self.platform = args.platform
        self.python_version = args.python_version
        self.verbose = args.verbose

    def _run(self, pip_install_args: List[str]) -> None:
        links = ltt.find_links(
            pip_install_args,
            computation_backends=self.computation_backends,
            channel=self.channel,
            platform=self.platform,
            python_version=self.python_version,
            verbose=self.verbose,
        )
        print("\n".join(links))


class InstallCommand(Command):
    def __init__(self, args: argparse.Namespace) -> None:
        self.force_cpu = args.force_cpu
        self.pytorch_only = args.pytorch_only
        self.channel = args.channel

        install_cmd = args.install_cmd
        if "{packages}" not in install_cmd:
            self.exit(error=True)
        self.install_cmd = install_cmd

        self.verbose = args.verbose

    def _run(self, pip_install_args: List[str]) -> None:
        links = ltt.find_links(
            pip_install_args,
            computation_backends={CPUBackend()} if self.force_cpu else None,
            channel=self.channel,
            verbose=self.verbose,
        )
        if links:
            cmd = self.install_cmd.format(packages=" ".join(links))
            subprocess.check_call(cmd, shell=True)
        else:
            warnings.warn(
                f"Didn't find any PyTorch distribution in "
                f"'{' '.join(pip_install_args)}'",
                RuntimeWarning,
            )

        if self.pytorch_only:
            self.exit()

        cmd = self.install_cmd.format(packages=self.collect_packages(pip_install_args))
        subprocess.check_call(cmd, shell=True)

    def collect_packages(self, pip_install_args: List[str]) -> str:
        parser = make_pip_install_parser()
        options, args = parser.parse_args(pip_install_args)
        editables = [f"-e {e}" for e in options.editables]
        requirements = [f"-r {r}" for r in options.requirements]
        return " ".join(editables + requirements + args)


COMMAD_CLASSES: Dict[Optional[str], Type[Command]] = {
    None: GlobalCommand,
    "extract": ExtractCommand,
    "find": FindCommand,
    "install": InstallCommand,
}


def make_command(args: argparse.Namespace) -> Command:
    cls = COMMAD_CLASSES[args.subcommand]
    return cls(args)
