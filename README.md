# `light-the-torch`

[![BSD-3-Clause License](https://img.shields.io/github/license/pmeier/light-the-torch)](https://opensource.org/licenses/BSD-3-Clause)
[![Project Status: WIP](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)
[![Code coverage via codecov.io](https://codecov.io/gh/pmeier/light-the-torch/branch/main/graph/badge.svg)](https://codecov.io/gh/pmeier/light-the-torch)

`light-the-torch` is a small utility that wraps `pip` to ease the installation process
for PyTorch distributions and third-party packages that depend on them. It auto-detects
compatible CUDA versions from the local setup and installs the correct PyTorch binaries
without user interference.

- [Why do I need it?](#why-do-i-need-it)
- [How do I install it?](#how-do-i-install-it)
- [How do I use it?](#how-do-i-use-it)
- [How does it work?](#how-does-it-work)

## Why do I need it?

---

TL;DR: TODO

---

PyTorch distributions are fully `pip install`'able, but PyPI, the default `pip` search
index, has some limitations:

1. PyPI regularly only allows binaries up to a size of
   [approximately 60 MB](https://github.com/pypa/packaging-problems/issues/86). One can
   [request a file size limit increase](https://pypi.org/help/#file-size-limit) (and the
   PyTorch team probably does that for every release), but it is still not enough:
   although PyTorch has pre-built binaries for Windows with CUDA, they cannot be
   installed through PyPI due to their size.
2. PyTorch uses local version specifiers to indicate for which computation backend the
   binary was compiled, for example `torch==1.11.0+cpu`. Unfortunately, local specifiers
   are not allowed on PyPI. Thus, only the binaries compiled with one CUDA version are
   uploaded without an indication of the CUDA version. If you do not have a CUDA capable
   GPU, downloading this is only a waste of bandwidth and disk capacity. If on the other
   hand your NVIDIA driver version simply doesn't support the CUDA version the binary
   was compiled with, you can't use any of the GPU features.

To overcome this, PyTorch also hosts _all_ binaries
[themselves](https://download.pytorch.org/whl/torch_stable.html). To access them, you
still can use `pip install`, but have to use some
[additional options](https://pytorch.org/get-started/locally/):

```shell
pip install torch==1.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

While this is certainly an improvement, it also has it downside: in addition to the
computation backend, the version has to be specified exactly. Without knowing what the
latest release is, it is impossible to install it as simple as `pip install torch`
normally would.

At this point you might justifiably as: why don't you just use `conda` as PyTorch
recommends?

```shell
conda install pytorch cpuonly -c pytorch
```

This should cover all cases, right? Well, no. The above command is only enough if you
just need PyTorch.

---

TODO

Otherwise

if you are a library author

pip is the most prominent package manager it is thus almost set to also publish your
package to PyPI Now, how do you tell your users to install your lbrary?

Imagine the case of a package that depends on PyTorch, but cannot be installed with
`conda` since it is hosted on PyPI? You can't use the `-f` option since the package in
question is not hosted by PyTorch. Thus, you now have to manually track down (and
resolve in the case of multiple packages) the PyTorch distributions, install them in a
first step and only install the actual package (and all other dependencies) afterwards.

2. The most prominent public conda channel is conda-forge.

If just want to use `pip install` like you always did before without worrying about any
of the stuff above, `light-the-torch` was made for you.

---

## How do I install it?

Installing `pytorch-pip-shim` is as easy as

```shell
pip install light-the-torch
```

Since it depends on `pip` and it might be upgraded during installation,
[Windows users](https://pip.pypa.io/en/stable/installing/#upgrading-pip) should install
it with

```shell
python -m pip install light-the-torch
```

## How do I use it?

After `light-the-torch` is installed you can use its CLI interface `ltt` as drop-in
replacement for `pip`:

```shell
ltt install torch
```

In fact, `ltt` is `pip` with a few added options:

- By default, `ltt` uses the local NVIDIA driver version to select the correct binary
  for you. You can pass the `--pytorch-computation-backend` / `--pcb` option to manually
  specify the computation backend you want to use:

  ```shell
  ltt install --pytorch-computation-backend=cu102 torch
  ```

- By default, `ltt` installs stable PyTorch binaries. In addition, PyTorch provides
  nightly, test, and LTS binaries. You can switch the channel you want to install from
  with the `--pytorch-channel` / `--pch` option:

  ```shell
  ltt install --pytorch-channel=nightly torch
  ```

  If the channel option is not passed, using `pip`'s builtin `--pre` option will install
  PyTorch test binaries.

Of course you are not limited to install only PyTorch distributions. Everything detailed
above also works if you install packages that depend on PyTorch.

## How does it work?

The authors of `pip` **do not condone** the use of `pip` internals as they might break
without warning. As a results of this, `pip` has no capability for plugins to hook into
specific tasks.

`light-the-torch` works by monkey-patching `pip` internals during the installation
process:

- While searching for a download link for a PyTorch distribution, `light-the-torch`
  replaces the default search index with an official PyTorch download link. This is
  equivalent to calling `pip install` with the `-f` option only for PyTorch
  distributions.
- While evaluating possible PyTorch installation candidates, `light-the-torch` culls
  binaries not compatible with the available hardware.
