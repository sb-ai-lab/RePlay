.. _splitters:

Splitters
==========

.. automodule:: replay.splitters

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
