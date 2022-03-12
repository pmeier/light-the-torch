from __future__ import annotations

import contextlib
import dataclasses
import enum
import functools
import importlib
import inspect
import itertools
import optparse
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)
from unittest import mock

from . import computation_backend as cb

__all__ = [
    "InternalError",
    "canocialize_name",
    "apply_fn_patch",
    "computation_backend_options",
]


class InternalError(RuntimeError):
    def __init__(self) -> None:
        # TODO: check against pip version
        msg = (
            "Unexpected internal pytorch-pip-shim error. If you ever encounter this "
            "message during normal operation, please submit a bug report at "
            "https://github.com/pmeier/pytorch-pip-shim/issues"
        )
        super().__init__(msg)


def canocialize_name(name: str) -> str:
    return name.lower().replace("_", "-")


# TODO: make backend also an enum


class Channel(enum.Enum):
    STABLE = enum.auto()
    TEST = enum.auto()
    NIGHTLY = enum.auto()
    LTS = enum.auto()

    @classmethod
    def from_str(cls, string: str) -> Channel:
        return cls[string.upper()]


# adapted from https://stackoverflow.com/a/9307174
class PassThroughOptionParser(optparse.OptionParser):
    def __init__(self, add_help_option: bool = False, **kwargs: Any) -> None:
        super().__init__(add_help_option=add_help_option, **kwargs)

    def _process_args(
        self, largs: List[str], rargs: List[str], values: optparse.Values
    ) -> None:
        while rargs:
            try:
                super()._process_args(largs, rargs, values)
            except (optparse.BadOptionError, optparse.AmbiguousOptionError) as error:
                largs.append(error.opt_str)


@dataclasses.dataclass
class LttOptions:
    computation_backends: List[cb.ComputationBackend] = dataclasses.field(
        default_factory=lambda: [cb.CPUBackend()]
    )
    channel: Channel = Channel.STABLE

    @classmethod
    def from_pip_argv(cls, argv: List[str]) -> LttOptions:
        if argv[0] != "install":
            return cls()

        parser = PassThroughOptionParser()

        for option in computation_backend_options():
            parser.add_option(option)

        parser.add_option(channel_option())
        # regular pip option
        parser.add_option("--pre", dest="pre", action="store_true")

        opts, _ = parser.parse_args(argv)

        if opts.pytorch_computation_backend is not None:
            cbs = [
                cb.ComputationBackend.from_str(string)
                for string in opts.pytorch_computation_backend.split(",")
            ]
        elif opts.cpuonly:
            cbs = [cb.CPUBackend()]
        else:
            cbs = cb.detect_compatible_computation_backends()

        if opts.pytorch_channel is not None:
            channel = Channel.from_str(opts.pytorch_channel)
        elif opts.pre:
            channel = Channel.TEST
        else:
            channel = Channel.STABLE

        return cls(cbs, channel)


# def extract_ltt_options(args: List[str]) -> SimpleNamespace:
#     if args[0] != "install":
#         args = []
#
#     parser = make_pip_args_parser()
#     opts, _ = parser.parse_args(args)
#
#     return SimpleNamespace(
#         computation_backend=process_computation_backend(opts), nightly=opts.nightly
#     )


def make_pip_args_parser() -> PassThroughOptionParser:
    parser = PassThroughOptionParser()

    for option in computation_backend_options():
        parser.add_option(option)

    parser.add_option(channel_option())
    # regular pip option
    parser.add_option("--pre", dest="pre", action="store_true")

    return parser


def computation_backend_options() -> List[optparse.Option]:
    return [
        optparse.Option(
            "--pytorch-computation-backend",
            "--pcb",
            help=(
                "Computation backend for compiled PyTorch distributions, "
                "e.g. 'cu92', 'cu101', or 'cpu'. "
                "If not specified, the computation backend is detected from the "
                "available hardware, preferring CUDA over CPU."
            ),
        ),
        optparse.Option(
            "--cpuonly",
            action="store_true",
            help=(
                "Shortcut for '--pytorch-computation-backend=cpu'. "
                "If '--computation-backend' is used simultaneously, "
                "it takes precedence over '--cpu'."
            ),
        ),
    ]


def channel_option() -> optparse.Option:
    return optparse.Option("--pytorch-channel", "--pc", help="FOO",)


def process_computation_backend(opts: optparse.Values) -> cb.ComputationBackend:
    return cb.CPUBackend()
    if opts.pytorch_computation_backend is not None:
        return cb.ComputationBackend.from_str(opts.pytorch_computation_backend)

    if opts.cpu:
        return cb.CPUBackend()

    return cb.detect()


Args = Union[Tuple[()], Tuple[Any], Tuple[Any, ...]]
Kwargs = Dict[str, Any]


def preprocessing_noop(args: Args, kwargs: Kwargs) -> Tuple[Args, Kwargs]:
    return args, kwargs


def postprocessing_noop(args: Args, kwargs: Kwargs, output: Any) -> Any:
    return output


@contextlib.contextmanager
def context_noop(args: Args, kwargs: Kwargs) -> Iterator[None]:
    yield


T = TypeVar("T")


class Input(Dict[str, T]):
    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__fn__ = fn

    def __getattr__(self, key: str) -> T:
        return self[key]

    def __setattr__(self, key: str, value: T) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        del self[key]

    @classmethod
    def from_call_args(cls, fn: Callable, *args: Any, **kwargs: Any) -> Input:
        params = iter(inspect.signature(fn).parameters.values())
        for arg, param in zip(args, params):
            kwargs[param.name] = arg
        for param in params:
            if (
                param.name not in kwargs
                and param.default is not inspect.Parameter.empty
            ):
                kwargs[param.name] = param.default
        return cls(fn, kwargs)

    def to_call_args(self):
        params = iter(inspect.signature(self.__fn__).parameters.values())

        args = []
        for param in params:
            if param.kind != inspect.Parameter.POSITIONAL_ONLY:
                break

            args.append(self[param.name])
        else:
            return (), dict()
        args = tuple(args)

        kwargs = dict()
        sentinel = object()
        for param in itertools.chain([param], params):
            kwarg = self.get(param.name, sentinel)
            if kwarg is not sentinel:
                kwargs[param.name] = kwarg

        return args, kwargs


@contextlib.contextmanager
def apply_fn_patch(
    target: str,
    preprocessing=lambda input: input,
    context=contextlib.nullcontext,
    postprocessing=lambda input, output: output,
) -> Iterator[None]:
    fn = import_fn(target)

    @functools.wraps(fn)
    def new(*args: Any, **kwargs: Any) -> Any:
        input = Input.from_call_args(fn, *args, **kwargs)

        input = preprocessing(input)
        with context(input):
            args, kwargs = input.to_call_args()
            output = fn(*args, **kwargs)
        return postprocessing(input, output)

    with mock.patch(target, new=new):
        yield


def import_fn(target: str):
    attrs = []
    name = target
    while name:
        try:
            module = importlib.import_module(name)
            break
        except ImportError:
            name, attr = name.rsplit(".", 1)
            attrs.append(attr)
    else:
        raise InternalError

    obj = module
    for attr in attrs[::-1]:
        obj = getattr(obj, attr)

    return obj
