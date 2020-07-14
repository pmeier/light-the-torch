import optparse
from typing import Any, Iterable, Type, TypeVar, cast

from pip._internal.commands.install import InstallCommand
from pip._internal.resolution.base import BaseResolver
from pip._internal.resolution.legacy.resolver import Resolver
from pip._internal.utils.logging import setup_logging
from pip._internal.utils.temp_dir import global_tempdir_manager, tempdir_registry

__all__ = [
    "InternalLTTError",
    "PatchedInstallCommand",
    "make_pip_install_parser",
    "run",
    "new_from_similar",
    "PatchedResolverBase",
]


class InternalLTTError(RuntimeError):
    def __init__(self) -> None:
        msg = (
            "Unexpected internal ltt error. If you ever encounter this "
            "message during normal operation, please submit a bug report at "
            "https://github.com/pmeier/light-the-torch/issues"
        )
        super().__init__(msg)


class PatchedInstallCommand(InstallCommand):
    def __init__(
        self, name: str = "name", summary: str = "summary", **kwargs: Any
    ) -> None:
        super().__init__(name, summary, **kwargs)


def make_pip_install_parser() -> optparse.OptionParser:
    return cast(optparse.OptionParser, PatchedInstallCommand().parser)


def get_verbosity(options: optparse.Values, verbose: bool) -> int:
    if not verbose:
        return -1

    return cast(int, options.verbose) - cast(int, options.quiet)


def run(
    cmd: InstallCommand, args: Iterable[str], options: optparse.Values, verbose: bool
) -> int:
    with cmd.main_context():
        cmd.tempdir_registry = cmd.enter_context(tempdir_registry())
        cmd.enter_context(global_tempdir_manager())

        setup_logging(
            verbosity=get_verbosity(options, verbose),
            no_color=options.no_color,
            user_log_file=options.log,
        )

        return cast(int, cmd.run(options, list(args)))


def get_public_or_private_attr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except AttributeError:
        try:
            return getattr(obj, f"_{name}")
        except AttributeError:
            msg = f"'{type(obj)}' has no attribute '{name}' or '_{name}'"
            raise AttributeError(msg)


T = TypeVar("T")


def new_from_similar(cls: Type[T], obj: Any, names: Iterable[str], **kwargs: Any) -> T:
    attrs = {name: get_public_or_private_attr(obj, name) for name in names}
    attrs.update(kwargs)
    return cls(**attrs)  # type: ignore[call-arg]


class PatchedResolverBase(Resolver):
    @classmethod
    def from_resolver(cls, resolver: BaseResolver) -> "PatchedResolverBase":
        return new_from_similar(
            cls,
            resolver,
            (
                "preparer",
                "finder",
                "wheel_cache",
                "upgrade_strategy",
                "force_reinstall",
                "ignore_dependencies",
                "ignore_installed",
                "ignore_requires_python",
                "use_user_site",
                "make_install_req",
                "py_version_info",
            ),
        )
