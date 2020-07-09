light-the-torch
===============

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

The latest **stable** version can be installed with

.. code-block:: sh

  pip install lighter


The **latest** potentially unstable version can be installed with

.. code-block::

  pip install git+https://github.com/pmeier/lighter

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
  image:: https://codecov.io/gh/pmeier/ltt/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/pmeier/ltt
    :alt: Test coverage via codecov.io
