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


**RePlay dependencies compilation**

RePlay depends on packages (e.g. LightFM, Implicit) that perform  C/C++ extension compilation on installation.
This requires C++ compiler, header files and other necessary components to be installed.

An example of error indicating header files absence is: ``Python.h: No such file or directory``

To install the necessary packages run the following for Ubuntu:

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
