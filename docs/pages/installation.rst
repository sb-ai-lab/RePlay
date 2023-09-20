Installation
============

It is recommended to use Unix machine with ``python >= 3.8``

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

    git clone git@github.com:sb-ai-lab/RePlay.git
    cd RePlay
    pip install -U pip wheel
    pip install -U requests pypandoc cython optuna poetry lightfm
    poetry install

Poetry resolves dependencies from ``pyproject.toml`` and fixes versions into ``poetry.lock`` file.
New packages can be added into configuration file with ``poetry add package``.

Adding new model
-------------------

Usually for new model we need updates in ``fit``, ``predict`` and ``predict_pairs``.

Typical function chains for ``fit`` and ``predict[_pairs]``::

    fit->_fit_wrap->_fit
    predict[_pairs]->_predict[_pairs]_wrap->_predict[_pairs]

For *LightFMWrap*::

    predict[_pairs]->_predict[_pairs]_wrap->_predict[_pairs]->_predict_selected_pairs

For *Word2VecRec* and models inherited from *NeighbourRec*::

    predict[_pairs]->_predict[_pairs]_wrap->_predict[_pairs]->_predict_pairs_inner

For models inherited from *BaseTorchRec*::

    fit->_fit_wrap->_fit->train
    predict[_pairs]->_predict[_pairs]_wrap->_predict[_pairs]->_predict_by_user[_pairs]->_predict_pairs_inner

Currently *AssociationRulesItemRec* doesn't have ``predict[_pairs]`` interface.
*ClusterRec*, *ImplicitWrap* and *RandomRec* currently doesn't have ``predict_pairs`` interface.

Current model inheritance in RePlay:

.. image:: /images/model_inheritance.jpg

`Current framework architecture <https://miro.com/app/board/uXjVOhTSHK0=/?share_link_id=748466292621>`_
