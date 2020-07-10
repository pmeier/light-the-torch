light-the-torch
===============

.. start-badges

.. list-table::
    :stub-columns: 1

    * - package
      - |license| |status|
    * - code
      - |black| |mypy| |lint|
    * - tests
      - |tests| |coverage|

.. end-badges

With each release of a PyTorch distribution (``torch``, ``torchvision``,
``torchaudio``, ``torchtext``) the wheels are published for combinations of different
Python versions, platforms, and computation backends (CPU, CUDA). Unfortunately, a
differentation based on the computation backend is not supported by
`PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_ . As a workaround the
computation backend is added as a local specifier. For example

.. code-block:: sh

  torch==1.5.1+<computation backend>

Due to this restriction only the wheels of the latest CUDA release are uploaded to
`PyPI <https://pypi.org/search/?q=torch>`_ and thus easily ``pip install`` able. For
other CUDA versions or the installation without CUDA support, one has to resort to
manual version specification:

.. code-block:: sh

  pip install -f https://download.pytorch.org/whl/torch_stable.html torch==1.5.1+<computation backend>

This is especially frustrating if one wants to install packages that depend on one or
several PyTorch distributions: for each package the required PyTorch distributions have
to be manually tracked down, resolved, and installed before the other requirements can
be installed.

``light-the-torch`` offers a small CLI based on ``pip`` to install the PyTorch
distributions from the stable releases. Similar to the Python version and platform, the
computation backend is auto-detected from the available hardware preferring CUDA over
CPU.

Installation
============

The latest **published** version can be installed with

.. code-block:: sh

  pip install light-the-torch


The latest, potentially unstable **development** version can be installed with

.. code-block::

  pip install git+https://github.com/pmeier/light-the-torch

Usage
=====

``light-the-torch`` is invoked with its shorthand ``ltt``

.. code-block:: sh

  $ ltt --help

  usage: ltt [-h] [-V] [--computation-backend COMPUTATION_BACKEND]
             [--full-install] [--install-cmd INSTALL_CMD] [--no-install]
             [args [args ...]]

  Install PyTorch distributions from the stable releases. The computation
  backend is autodetected from the available hardware preferring CUDA over CPU.

  positional arguments:
    args                  arguments passed to pip install. Required PyTorch
                          distributions are extracted and installed. Optional
                          arguments for pip install have to be seperated by '--'
                          (default: None)

  optional arguments:
    -h, --help            show this help message and exit
    -V, --version         show version and exit (default: False)
    --computation-backend COMPUTATION_BACKEND
                          pin computation backend, e.g. 'cpu' or 'cu102'
                          (default: None)
    --full-install        install remaining requirements after PyTorch
                          distributions are installed (default: False)
    --install-cmd INSTALL_CMD
                          installation command. '{links}' is substituted for the
                          links. If present, '{opts}' is substituted for most
                          additional pip install options. Exceptions are -e /
                          --editable <path/url> and -r / --requirement <file>
                          (default: pip install {opts} {links})
    --no-install          print wheel links instead of installing (default:
                          False)

.. note::

  The following examples were run on a linux machine with Python 3.6 and CUDA 10.1. The
  distributions hosted on PyPI were built with CUDA 10.2.

Example 1
---------

``ltt`` can be used to install PyTorch distributions without worrying about the
computation backend:

.. code-block:: sh

  $ ltt torch torchvision
  [...]
  Successfully installed future-0.18.2 numpy-1.19.0 pillow-7.2.0 torch-1.5.1+cu101 torchvision-0.6.1+cu101

Example 2
---------

``ltt`` extracts the required PyTorch distributions from the positional arguments:

.. code-block:: sh

  $ ltt kornia
  [...]
  Successfully installed torch-1.5.0+cu101

Example 3
---------

The ``--full-install`` option can be used as a replacement for ``pip install``:

.. code-block::

  $ ltt --full-install kornia
  [...]
  Successfully installed future-0.18.2 numpy-1.19.0 torch-1.5.0+cu101
  [...]
  Successfully installed kornia-0.3.1

Example 4
---------

The ``--no-install`` option can be used to pipe or redirect the PyTorch wheel links.
For example, generating a ``requirements.txt`` file:

.. code-block:: sh

  $ ltt --no-install torchaudio > requirements.txt
  $ cat requirements.txt
  https://download.pytorch.org/whl/cu101/torch-1.5.1%2Bcu101-cp36-cp36m-linux_x86_64.whl
  https://download.pytorch.org/whl/torchaudio-0.5.1-cp36-cp36m-linux_x86_64.whl

Example 5
---------

The ``--computation-backend`` option as well as the ``--platform`` and
``--python-version`` options from ``pip install`` can be used to disable the
autodetection:

.. code-block::

  $ ltt \
    --no-install \
    --computation-backend cu92 \
    -- \
    --python-version 37 \
    --platform win_amd64 \
    torchtext
  https://download.pytorch.org/whl/cu92/torch-1.5.1%2Bcu92-cp37-cp37m-win_amd64.whl
  https://download.pytorch.org/whl/torchtext-0.6.0-py3-none-any.whl

.. note::

  Optional arguments for ``pip install`` have to be passed after a ``--`` seperator.

.. |license|
  image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause
    :alt: License

.. |status|
  image:: https://www.repostatus.org/badges/latest/wip.svg
    :alt: Project Status: WIP
    :target: https://www.repostatus.org/#wip

.. |black|
  image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black
   
.. |mypy|
  image:: http://www.mypy-lang.org/static/mypy_badge.svg
    :target: http://mypy-lang.org/
    :alt: mypy

.. |lint|
  image:: https://github.com/pmeier/light-the-torch/workflows/lint/badge.svg
    :target: https://github.com/pmeier/light-the-torch/actions?query=workflow%3Alint+branch%3Amaster
    :alt: Lint status via GitHub Actions

.. |tests|
  image:: https://github.com/pmeier/light-the-torch/workflows/tests/badge.svg
    :target: https://github.com/pmeier/light-the-torch/actions?query=workflow%3Atests+branch%3Amaster
    :alt: Test status via GitHub Actions

.. |coverage|
  image:: https://codecov.io/gh/pmeier/light-the-torch/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pmeier/light-the-torch
    :alt: Test coverage via codecov.io
