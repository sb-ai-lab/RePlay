Preprocessing
===================

.. automodule:: replay.preprocessing

.. _data-preparator:

Data Preparation
___________________

Replay has a number of requirements for input data.
We await that input columns are in the form ``[user_id, item_id, timestamp, relevance]``.
And internal format is a spark DataFrame with indexed integer values for ``[user_idx, item_idx]``.
You can convert indexes of your Spark DataFrame with ``Indexer`` class.

.. autoclass:: replay.preprocessing.data_preparator.Indexer
   :members:

If your DataFrame is in the form of Pandas DataFrame and has different column names, you can either
preprocess it yourself with ``convert2spark`` function or use ``DataPreparator`` class


.. autoclass:: replay.preprocessing.data_preparator.DataPreparator
   :members:


Filters
___________________

.. automodule:: replay.preprocessing.filters
    :members:
