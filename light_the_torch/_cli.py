from pip._internal.cli.main import main as pip_main

from ._patch import patch

main = patch(pip_main)
