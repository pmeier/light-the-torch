import configparser
from os import path


def extract_requirements(root, file="tox.ini"):
    config = configparser.ConfigParser()
    config.read(path.join(root, "..", file))
    return config["testenv:docs"]["deps"].strip().split("\n")


def write_requirements_file(root, requirements, file="requirements-rtd.txt"):
    with open(path.join(root, file), "w") as fh:
        fh.write("\n".join(requirements) + "\n")


def main(root):
    requirements = extract_requirements(root)
    write_requirements_file(root, requirements)


if __name__ == "__main__":
    root = path.dirname(__file__)
    main(root)
