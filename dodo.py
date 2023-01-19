import itertools
import os
import pathlib
import shlex

from doit.action import CmdAction


HERE = pathlib.Path(__file__).parent
PACKAGE_NAME = "light_the_torch"

CI = os.environ.get("CI") == "true"

DOIT_CONFIG = dict(
    verbosity=2,
    backend="json",
    default_tasks=[
        "lint",
        "test",
        "publishable",
    ],
)


def do(*cmd, cwd=HERE):
    if len(cmd) == 1 and callable(cmd[0]):
        cmd = cmd[0]
    else:
        cmd = list(itertools.chain.from_iterable(shlex.split(part) for part in cmd))
    return CmdAction(cmd, shell=False, cwd=cwd)


def task_install():
    """Installs all development requirements and light-the-torch in development mode"""
    yield dict(
        name="dev",
        actions=[do("python -m pip install --upgrade -r requirements-dev.txt")],
    )
    yield dict(
        name="project",
        actions=[do("python -m pip install --upgrade -e .")],
    )


def task_format():
    """Auto-formats all project files"""
    return dict(
        actions=[
            do("pre-commit run --all-files"),
        ]
    )


def task_lint():
    """Lints all project files"""
    return dict(
        actions=[
            do("flake8 --config=.flake8"),
        ],
    )


def task_test():
    """Runs the test suite"""
    return dict(
        actions=[
            do(
                "pytest -c pytest.ini",
                f"--cov-report={'xml' if CI else 'term'}",
            )
        ]
    )


def task_build():
    """Builds the source distribution and wheel"""
    return dict(
        actions=[
            do("python -m build ."),
        ],
        clean=[
            do(f"rm -rf build dist {PACKAGE_NAME}.egg-info"),
        ],
    )


def task_publishable():
    """Checks if metadata is correct"""
    yield dict(
        name="twine",
        actions=[
            # We need the lambda here to lazily glob the files in dist/*, since they
            # are only created by the build task rather than when this task is
            # created.
            do(lambda: ["twine", "check", *list((HERE / "dist").glob("*"))]),
        ],
        task_dep=["build"],
    )
    yield dict(
        name="check-wheel-contents",
        actions=[
            do("check-wheel-contents dist"),
        ],
        task_dep=["build"],
    )


def task_publish():
    """Publishes to PyPI"""
    # TODO: check if env vars are set
    return dict(
        # We need the lambda here to lazily glob the files in dist/*, since they are
        # only created by the build task rather than when this task is created.
        actions=[
            do(lambda: ["twine", "upload", *list((HERE / "dist").glob("*"))]),
        ],
        task_dep=[
            "lint",
            "test",
            "publishable",
        ],
    )
