from os import path

import toml
from setuptools_scm import get_version


def get_version_scheme(root):
    try:
        content = toml.load(path.join(root, "pyproject.toml"))
        return content["tool"]["setuptools_scm"]["version_scheme"]
    except (FileNotFoundError, KeyError):
        return None


def main(root="."):
    version_scheme = get_version_scheme(root)
    version = get_version(
        root=root, version_scheme=version_scheme, local_scheme="no-local-version"
    )
    print(version)


if __name__ == "__main__":
    root = path.abspath(path.join(path.dirname(__file__), ".."))
    main(root)
