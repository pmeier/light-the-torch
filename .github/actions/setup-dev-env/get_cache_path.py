import os
import pathlib


def main():
    rel_path = {
        "Linux": pathlib.PurePosixPath(".cache") / "pip",
        "Windows": pathlib.PureWindowsPath("AppData") / "Local" / "pip" / "Cache",
        "macOS": pathlib.PurePosixPath("Library") / "Caches" / "pip",
    }[os.environ["OS"]]
    print(pathlib.Path.home() / rel_path)


if __name__ == "__main__":
    main()
