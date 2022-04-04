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
can still use `pip install` them, but some
[additional options](https://pytorch.org/get-started/locally/) are needed:

```shell
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
```

While this is certainly an improvement, it still has a few downsides:

1. You need to know what computation backend, e.g. CUDA 11.3 (`cu113`), is supported on
   your local machine. This can be quite challenging for new users and at least tedious
   for more experienced ones.
2. Besides the stable binaries, PyTorch also offers nightly, test, and long-time support
   (LTS) ones. To install them, you need a different `--extra-index-url` value for each.
   For the nightly and test channel you also need to supply the `--pre` option.
3. If you want to install any package hosted on PyPI that depends on PyTorch, you always
   also specify all PyTorch distributions to install. Otherwise, the `--extra-index-url`
   flag is ignored and the PyTorch distributions hosted on PyPI will be installed.

If any of these points don't sound appealing to you, and you just want to have the same
user experience as `pip install` for PyTorch distributions, `light-the-torch` was made
for you.

## How do I install it?

Installing `light-the-torch` is as easy as

```shell
pip install --pre light-the-torch
```

Since it depends on `pip` and it might be upgraded during installation,
[Windows users](https://pip.pypa.io/en/stable/installing/#upgrading-pip) should install
it with

```shell
python -m pip install --pre light-the-torch
```

## How do I use it?

After `light-the-torch` is installed you can use its CLI interface `ltt` as drop-in
replacement for `pip`:

```shell
ltt install torch
```

In fact, `ltt` is `pip` with a few added options:

- By default, `ltt` uses the local NVIDIA driver version to select the correct binary
  for you. You can pass the `--pytorch-computation-backend` option to manually specify
  the computation backend you want to use:

  ```shell
  ltt install --pytorch-computation-backend=cu102 torch
  ```

- By default, `ltt` installs stable PyTorch binaries. To install binaries from nightly,
  test, or LTS channels pass the `--pytorch-channel` option:

  ```shell
  ltt install --pytorch-channel=nightly torch
  ```

  If `--pytorch-channel` is not passed, using `pip`'s builtin `--pre` option will
  install PyTorch test binaries.

Of course you are not limited to install only PyTorch distributions. Everything shown
above also works if you install packages that depend on PyTorch:

```shell
ltt install --pytorch-computation-backend=cpu --pytorch-channel=nightly pystiche
```

## How does it work?

The authors of `pip` **do not condone** the use of `pip` internals as they might break
without warning. As a results of this, `pip` has no capability for plugins to hook into
specific tasks.

`light-the-torch` works by monkey-patching `pip` internals at runtime:

- While searching for a download link for a PyTorch distribution, `light-the-torch`
  replaces the default search index with an official PyTorch download link. This is
  equivalent to calling `pip install` with the `--extra-index-url` option only for
  PyTorch distributions.
- While evaluating possible PyTorch installation candidates, `light-the-torch` culls
  binaries incompatible with the hardware.
