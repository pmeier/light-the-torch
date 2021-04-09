import contextlib
import re
from typing import Any, List, NoReturn, cast

from pip._internal.req.req_install import InstallRequirement
from pip._internal.req.req_set import RequirementSet

from ..compatibility import find_compatible_torch_version
from .common import InternalLTTError, PatchedInstallCommand, PatchedResolverBase, run

__all__ = ["extract_dists"]


def extract_dists(pip_install_args: List[str], verbose: bool = False) -> List[str]:
    """Extract direct or indirect required PyTorch distributions.

    Args:
        pip_install_args: Arguments passed to ``pip install`` that will be searched for
            required PyTorch distributions
        verbose: If ``True``, print additional information to STDOUT.

    Returns:
        Resolved required PyTorch distributions.
    """
    cmd = StopAfterPytorchDistsFoundInstallCommand()
    options, args = cmd.parser.parse_args(pip_install_args)
    try:
        run(cmd, args, options, verbose)
    except PytorchDistsFound as resolution:
        return resolution.dists
    else:
        raise InternalLTTError


class PytorchDistsFound(RuntimeError):
    def __init__(self, dists: List[str]) -> None:
        self.dists = dists


class StopAfterPytorchDistsFoundResolver(PatchedResolverBase):
    PYTORCH_CORE = "torch"
    PYTORCH_SUBS = ("vision", "text", "audio")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pytorch_dists = (
            self.PYTORCH_CORE,
            *[f"{self.PYTORCH_CORE}{sub}" for sub in self.PYTORCH_SUBS],
        )
        self._pytorch_core_pattern = re.compile(
            f"^{self.PYTORCH_CORE}(?!({'|'.join(self.PYTORCH_SUBS)}))"
        )
        self._required_pytorch_dists: List[str] = []

    def _resolve_one(
        self, requirement_set: RequirementSet, req_to_install: InstallRequirement
    ) -> List[InstallRequirement]:
        if req_to_install.name not in self._pytorch_dists:
            return cast(
                List[InstallRequirement],
                super()._resolve_one(requirement_set, req_to_install),
            )

        self._required_pytorch_dists.append(str(req_to_install.req))
        return []

    def resolve(
        self, root_reqs: List[InstallRequirement], check_supported_wheels: bool
    ) -> NoReturn:
        super().resolve(root_reqs, check_supported_wheels)
        raise PytorchDistsFound(self.required_pytorch_dists)

    @property
    def required_pytorch_dists(self) -> List[str]:
        dists = self._required_pytorch_dists
        if not dists:
            return []

        # If the distribution was found in an extra requirement, pip appends this as
        # additional information. We remove that here.
        dists = [dist.split(";")[0] for dist in dists]

        if not any(self._pytorch_core_pattern.match(dist) for dist in dists):
            torch = self.PYTORCH_CORE

            with contextlib.suppress(RuntimeError):
                torch_versions = {
                    find_compatible_torch_version(*dist.split("=="))
                    for dist in dists
                    if "==" in dist
                }
                if len(torch_versions) == 1:
                    torch = f"{torch}=={torch_versions.pop()}"

            dists.insert(0, torch)

        return dists


class StopAfterPytorchDistsFoundInstallCommand(PatchedInstallCommand):
    def make_resolver(
        self, *args: Any, **kwargs: Any
    ) -> StopAfterPytorchDistsFoundResolver:
        resolver = super().make_resolver(*args, **kwargs)
        return StopAfterPytorchDistsFoundResolver.from_resolver(resolver)
