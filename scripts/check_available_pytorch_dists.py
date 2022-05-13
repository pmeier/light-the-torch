#!/usr/bin/env python

import itertools
import json

import requests
from bs4 import BeautifulSoup

from light_the_torch._cb import _MINIMUM_DRIVER_VERSIONS, CPUBackend, CUDABackend
from light_the_torch._patch import Channel, get_extra_index_urls, PYTORCH_DISTRIBUTIONS

EXCLUDED_PYTORCH_DIST = {
    "nestedtensor",
    "pytorch_csprng",
    "torch_cuda80",
    "torch_nightly",
    "torchaudio_nightly",
    # "torchrec_nightly",
    # "torchrec_nightly_3.7_cu11.whl",
    # "torchrec_nightly_3.8_cu11.whl",
    # "torchrec_nightly_3.9_cu11.whl",
}
PATCHED_PYTORCH_DISTS = set(PYTORCH_DISTRIBUTIONS)

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
        if not response.ok:
            continue

        soup = BeautifulSoup(response.text, features="html.parser")

        available.update(tag.string for tag in soup.find_all(name="a"))
    available = available - EXCLUDED_PYTORCH_DIST

    missing = available - PATCHED_PYTORCH_DISTS
    extra = PATCHED_PYTORCH_DISTS - available

    if missing or extra:
        print(",".join(sorted(available)))


if __name__ == "__main__":
    main()
