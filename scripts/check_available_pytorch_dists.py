import itertools

import requests
from bs4 import BeautifulSoup

from light_the_torch._cb import _MINIMUM_DRIVER_VERSIONS, CPUBackend, CUDABackend

from light_the_torch._patch import Channel, get_extra_index_urls, PYTORCH_DISTRIBUTIONS


COMPUTATION_BACKENDS = {
    CUDABackend(cuda_version.major, cuda_version.minor)
    for minimum_driver_versions in _MINIMUM_DRIVER_VERSIONS.values()
    for cuda_version, minimum_driver_version in minimum_driver_versions.items()
}
COMPUTATION_BACKENDS.add(CPUBackend())

EXTRA_INDEX_URLS = set(
    itertools.chain.from_iterable(
        get_extra_index_urls(COMPUTATION_BACKENDS, channel) for channel in iter(Channel)
    )
)


def main():
    available = set()
    for url in EXTRA_INDEX_URLS:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, features="html.parser")

        available.update(tag.string for tag in soup.find_all(name="a"))

    missing = available - PYTORCH_DISTRIBUTIONS
    extra = PYTORCH_DISTRIBUTIONS - available
    if not (missing or extra):
        return

    print("PYTORCH_DISTRIBUTIONS = {")
    for dist in sorted(available):
        print(f'    "{dist}",')
    print("}")


if __name__ == "__main__":
    main()
