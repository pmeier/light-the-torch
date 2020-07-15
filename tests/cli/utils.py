import contextlib

import pytest

__all__ = ["exits"]


@contextlib.contextmanager
def exits(code=None, error=False):
    with pytest.raises(SystemExit) as info:
        yield

    ret = info.value.code

    if code is not None:
        assert ret == code

    if error:
        assert ret >= 1
    else:
        assert ret is None or ret == 0
