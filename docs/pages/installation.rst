Installation
============

It is recommended to use Unix machine with ``python >= 3.7``

Basic
--------

.. code-block:: bash

    pip install replay-rec


Development
---------------

You can also clone repository and install with poetry

.. code-block:: bash

    git clone git@github.com:sberbank-ai-lab/RePlay.git
    cd RePlay
    pip install -U pip wheel
    pip install -U requests pypandoc cython optuna poetry
    poetry install

Poetry resolves dependencies from ``pyproject.toml`` and fixes versions into ``poetry.lock`` file.
New packages can be added into configuration file with ``poetry add package``.
