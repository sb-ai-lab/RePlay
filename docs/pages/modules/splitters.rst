.. _splitters:

Splitters
==========

.. automodule:: replay.splitters

Below is the documentation of the core splitter 
classes implemented in RePlay. For practical use, 
splitters can be composed and combined with auxiliary 
utilities from RePlay (see :doc:`/pages/modules/preprocessing`) to obtain
different data partitioning schemes.

As proposed in the paper `Time to Split: Exploring Data Splitting Strategies for Offline
Evaluation of Sequential Recommenders (RecSys'25) <https://arxiv.org/pdf/2507.16289>`_, 
advanced data‑splitting schemes — Global Temporal Split (GTS) 
with last interaction as the target and GTS with a random interaction as the target,
can be implemented in RePlay by composing :class:`~replay.splitters.time_splitter.TimeSplitter` 
with either :class:`~replay.splitters.last_n_splitter.LastNSplitter` (e.g., ``N=1``) 
or :class:`~replay.splitters.random_next_n_splitter.RandomNextNSplitter` (e.g., ``N=1``).
These pipelines can be complemented with auxiliary utilities, such as cold‑start
filtering via :func:`~replay.preprocessing.filters.filter_cold` (see
:mod:`~replay.preprocessing.filters` and :doc:`/pages/modules/preprocessing`) and
dataset merging via :func:`~replay.preprocessing.utils.merge_subsets`.
For an end‑to‑end illustration, see ``examples/04_splitters.ipynb``.

Splits are returned with ``split`` method.

.. autofunction:: replay.splitters.base_splitter.Splitter.split


TwoStageSplitter
----------------

.. autoclass:: replay.splitters.two_stage_splitter.TwoStageSplitter
   :special-members: __init__

KFolds
---------

.. autofunction:: replay.splitters.k_folds.KFolds


TimeSplitter
-------------
.. autoclass:: replay.splitters.time_splitter.TimeSplitter
   :special-members: __init__

LastNSplitter
-------------
.. autoclass:: replay.splitters.last_n_splitter.LastNSplitter
   :special-members: __init__

RatioSplitter
-------------
.. autoclass:: replay.splitters.ratio_splitter.RatioSplitter
   :special-members: __init__

RandomSplitter
----------------
.. autoclass:: replay.splitters.random_splitter.RandomSplitter
   :special-members: __init__

NewUsersSplitter
-----------------
.. autoclass:: replay.splitters.new_users_splitter.NewUsersSplitter
   :special-members: __init__

ColdUserRandomSplitter
------------------------
.. autoclass:: replay.splitters.cold_user_random_splitter.ColdUserRandomSplitter
   :special-members: __init__

RandomNextNSplitter
--------------------
.. autoclass:: replay.splitters.random_next_n_splitter.RandomNextNSplitter
   :special-members: __init__