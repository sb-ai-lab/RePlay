Installation
============

It is recommended to use Unix machine with ``python >= 3.7``

Basic
--------

.. code-block:: bash

    pip install replay-rec


Troubleshooting
------------------

**General**

If you have an installation trouble, update the core packages:

.. code-block:: bash

    pip install --upgrade pip wheel


**Implicit**

RePlay depends on `implicit <https://github.com/benfred/implicit>`_, which requires C++ compiler
and may require installation of additional packages to build *implicit* from source on Unix machines.

If you are facing an error during *implicit* installation, try the following:

.. code-block:: bash

    sudo apt-get install python3-dev
    sudo apt-get install build-essential


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
