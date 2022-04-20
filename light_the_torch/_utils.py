import contextlib
import functools
import importlib
import inspect
import itertools

from unittest import mock


class InternalError(RuntimeError):
    def __init__(self) -> None:
        # TODO: check against pip version
        # TODO: fix wording
        msg = (
            "Unexpected internal pytorch-pip-shim error. If you ever encounter this "
            "message during normal operation, please submit a bug report at "
            "https://github.com/pmeier/pytorch-pip-shim/issues"
        )
        super().__init__(msg)


class Input(dict):
    def __init__(self, fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__fn__ = fn

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value) -> None:
        self[key] = value

    def __delattr__(self, key) -> None:
        del self[key]

    @classmethod
    def from_call_args(cls, fn, *args, **kwargs):
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
    *parts,
    preprocessing=lambda input: input,
    context=contextlib.nullcontext,
    postprocessing=lambda input, output: output,
):
    target = ".".join(parts)
    fn = import_fn(target)

    @functools.wraps(fn)
    def new(*args, **kwargs):
        input = Input.from_call_args(fn, *args, **kwargs)

        preprocessing(input)
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
