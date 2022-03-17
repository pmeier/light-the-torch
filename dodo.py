import itertools
import os
import pathlib
import shlex

from doit.action import CmdAction


HERE = pathlib.Path(__file__).parent
PACKAGE_NAME = "light_the_torch"

CI = os.environ.get("CI") == "1"

DOIT_CONFIG = dict(
    verbosity=2,
    backend="json",
)


def do(*cmd):
    if len(cmd) == 1:
        cmd = cmd[0]
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    return CmdAction(cmd, shell=False, cwd=HERE)


def task_install(pip="python -m pip"):
    """Installs all development requirements and light-the-torch in development mode"""
    yield dict(
        name="requirements",
        actions=[do(f"{pip} install -r requirements-dev.txt")],
    )
    yield dict(
        name="install",
        actions=[
            do(f"{pip} install -e ."),
        ],
    )


def task_setup():
    """Sets up a development environment for light-the-torch"""
    dev_env = HERE / ".venv"
    return dict(
        actions=[
            do(f"virtualenv {dev_env} --prompt='(light-the-torch-dev) '"),
            *itertools.chain.from_iterable(
                sub_task["actions"]
                for sub_task in task_install(dev_env / "bin" / "pip")
            ),
            lambda: print(
                f"run `source {dev_env / 'bin' / 'activate'}` the virtual environment"
            ),
        ],
        clean=[do(f"rm -rf {dev_env}")],
        uptodate=[lambda: dev_env.exists()],
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
        ]
    )


def task_test():
    """Runs the test suite"""

    def run(passthrough):
        return [
            "pytest",
            "-c",
            HERE / "pytest.ini",
            f"--cov-report={'xml' if CI else 'term'}",
            *passthrough,
        ]

    return dict(
        actions=[do(run)],
        pos_arg="passthrough",
    )


def task_build():
    """Builds the source distribution and wheel of light-the-torch"""
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
    """Publishes light-the-torch to PyPI"""
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
