import pytest

from light_the_torch._core import common


def test_get_public_or_private_attr_public_and_private():
    class ObjWithPublicAndPrivateAttribute:
        attr = "public"
        _attr = "private"

    obj = ObjWithPublicAndPrivateAttribute()
    assert common.get_public_or_private_attr(obj, "attr") == "public"


def test_get_public_or_private_attr_public_only():
    class ObjWithPublicAttribute:
        attr = "public"

    obj = ObjWithPublicAttribute()
    assert common.get_public_or_private_attr(obj, "attr") == "public"


def test_get_public_or_private_attr_private_only():
    class ObjWithPrivateAttribute:
        _attr = "private"

    obj = ObjWithPrivateAttribute()
    assert common.get_public_or_private_attr(obj, "attr") == "private"


def test_get_public_or_private_attr_no_attribute():
    class ObjWithoutAttribute:
        pass

    obj = ObjWithoutAttribute()

    with pytest.raises(AttributeError):
        common.get_public_or_private_attr(obj, "attr")


def test_new_from_similar():
    class Object:
        def __init__(self, foo, bar="bar"):
            self.foo = foo
            self._bar = bar

        def __eq__(self, other):
            foo = self.foo == other.foo
            bar = self._bar == other._bar
            return foo and bar

    class PatchedObj(Object):
        def __init__(self, foo, bar="patched_default_bar", baz=None):
            super().__init__(foo, bar=bar)
            self._baz = baz

    obj = Object("foo")
    new_obj = common.new_from_similar(PatchedObj, obj, ("foo", "bar"), baz="baz")

    assert new_obj is not obj
    assert isinstance(new_obj, PatchedObj)
    assert new_obj == obj
    assert common.get_public_or_private_attr(new_obj, "baz") == "baz"
