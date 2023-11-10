Preprocessing
===================

.. automodule:: replay.preprocessing


Filters
___________________

.. automodule:: replay.preprocessing.filters
    :members:


CSRConverter
___________________
Convert input data to csr sparse matrix.

.. autoclass:: replay.preprocessing.converter.CSRConverter
    :members:


Sessionizer
___________________
Create and filter sessions from given interactions.

.. autoclass:: replay.preprocessing.sessionizer.Sessionizer
    :members:


Padder (Experimental)
__________________________
Pad array columns in dataframe.

.. autoclass:: replay.experimental.preprocessing.padder.Padder
    :members:


SequenceGenerator (Experimental)
_____________________________________
Creating sequences for sequential models.

.. autoclass:: replay.experimental.preprocessing.sequence_generator.SequenceGenerator
    :members:


Data Preparation (Experimental)
_______________________________

Replay has a number of requirements for input data.
We await that input columns are in the form ``[user_id, item_id, timestamp, relevance]``.
And internal format is a spark DataFrame with indexed integer values for ``[user_idx, item_idx]``.
You can convert indexes of your Spark DataFrame with ``Indexer`` class.

.. autoclass:: replay.experimental.preprocessing.data_preparator.Indexer
   :members:

If your DataFrame is in the form of Pandas DataFrame and has different column names, you can either
preprocess it yourself with ``convert2spark`` function or use ``DataPreparator`` class


.. autoclass:: replay.experimental.preprocessing.data_preparator.DataPreparator
   :members: