import pathlib

from doit.action import CmdAction

HERE = pathlib.Path(__file__).parent


def task_dev_env():
    dev_env = HERE / ".venv"
    pip = dev_env / "bin" / "pip"
    return dict(
        actions=[
            ["rm", "-rf", dev_env],
            ["virtualenv", dev_env, "--prompt=(light-the-torch-dev) "],
            [pip, "install", "-r", HERE / "requirements-dev.txt"],
            [pip, "install", "-e", HERE],
        ],
    )


def task_format():
    return dict(actions=[["pre-commit", "run", "--all-files"]])


def task_lint():
    yield dict(
        name="flake8", actions=[["flake8", "--config=.flake8"]],
    )
    yield dict(
        name="mypy", actions=[["mypy", "--config-file=mypy.ini"]],
    )


def task_test():
    def run(coverage, passthrough):
        cmd = [
            "pytest",
            "-c",
            HERE / "pytest.ini",
        ]
        if coverage:
            cmd.extend(
                [
                    f"--cov={HERE / 'light_the_torch'}",
                    "--cov-report=xml",
                    f"--cov-config={HERE / '.coveragerc'}",
                ]
            )
        cmd.extend(passthrough)
        return cmd

    return dict(
        actions=[CmdAction(run, shell=False)],
        verbosity=2,
        params=[dict(name="coverage", long="coverage", type=bool, default=False,)],
        pos_arg="passthrough",
    )


def task_publishable():
    return dict(
        actions=[
            [
                "rm",
                "-rf",
                HERE / "build",
                HERE / "dist",
                HERE / "light_the_torch.egg-info",
            ],
            ["python", "-m", "build", "--sdist", "--wheel", HERE],
            ["twine", "check", *list((HERE / "dist").glob("*"))],
            ["check-wheel-contents", HERE / "dist"],
        ],
    )


def task_publish():
    task = task_publishable()
    task["actions"].append(["twine", "upload", *list((HERE / "dist").glob("*"))])
    return task
