from pip._internal.cli.main import main

from ._patch import patch_pip_main

main = patch_pip_main(main)
