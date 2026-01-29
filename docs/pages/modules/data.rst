Data
==================================

.. automodule:: replay.data

Dataset
----------------------
.. autoclass:: replay.data.Dataset
    :members:

DatasetLabelEncoder
----------------------
.. autoclass:: replay.data.dataset_utils.DatasetLabelEncoder
    :members:

FeatureType
----------------------
.. auto_autoenum:: replay.data.FeatureType
    :members:

FeatureSource
----------------------
.. auto_autoenum:: replay.data.FeatureSource
    :members:

FeatureHint
----------------------
.. auto_autoenum:: replay.data.FeatureHint
    :members:

FeatureInfo
----------------------
.. autoclass:: replay.data.FeatureInfo
    :members:

FeatureSchema
----------------------
.. autoclass:: replay.data.FeatureSchema
    :members:

GetSchema
----------------------
.. autofunction:: replay.data.get_schema


Neural Networks
----------------------
This submodule is only available when the `PyTorch` is installed.

.. automodule:: replay.data.nn

TensorFeatureInfo
______________________
.. autoclass:: replay.data.nn.TensorFeatureInfo
    :members:

TensorFeatureSource
______________________
.. autoclass:: replay.data.nn.TensorFeatureSource
    :members:

TensorSchema
______________________
.. autoclass:: replay.data.nn.TensorSchema
    :members:

SequenceTokenizer
______________________
.. autoclass:: replay.data.nn.SequenceTokenizer
    :members:

PandasSequentialDataset
_______________________
.. autoclass:: replay.data.nn.PandasSequentialDataset
    :members:
    :inherited-members:

TorchSequentialBatch
______________________
.. autoclass:: replay.data.nn.TorchSequentialBatch
    :members:

TorchSequentialDataset
______________________
.. autoclass:: replay.data.nn.TorchSequentialDataset
    :members: __init__

TorchSequentialValidationBatch
______________________________
.. autoclass:: replay.data.nn.TorchSequentialValidationBatch
    :members:

TorchSequentialValidationDataset
________________________________
.. autoclass:: replay.data.nn.TorchSequentialValidationDataset
    :members: __init__


.. _parquet-processing:

Parquet processing
__________________

This module contains the implementation of ``ParquetDataset`` - a combination of PyTorch-compatible dataset and sampler designed for working with the Parquet file format.
The main advantages offered by this dataset are:

1. Batch-wise reading and processing of data, allowing it to work with large datasets in memory-constrained settings.
2. Full built-in support for Torch's Distributed Data Parallel mode.
3. Automatic padding of data according to the provided schema.

``ParquetDataset`` is primarily configured using column schemas - dictionaries containing target columns as keys and their shape/padding specifiers as values.
An example column schema:

.. code-block:: python

    schema = {
        "user_id": {} # Empty metadata represents a non-array column.
        "seq_1": {"shape": 5} # 1-D sequences of length 5 using default padding value as -1.
        "seq_2": {"shape": [5, 6], "padding": -2} # 2-D sequences with custom padding values
    }

ParquetDataset
```````````````
.. autoclass:: replay.data.nn.parquet.ParquetDataset
    :members: __init__

.. _Parquet-Module:

ParquetModule (Lightning DataModule)
____________________________________

.. autoclass:: replay.data.nn.ParquetModule
    :members: __init__

**Example**

This is a minimal usage example of ``ParquetModule``. It uses train data only, and the Transforms are defined to support further training of the SasRec model.

..

See the full example in `examples/09_sasrec_example.ipynb`. 

    .. code-block:: python

        from replay.data.nn import ParquetModule
        from replay.nn.transform.template import make_default_sasrec_transforms

        metadata = {
            "user_id": {},
            "item_id": {"shape": 50, "padding": 51},
        }
        transforms = make_default_sasrec_transforms(tensor_schema, query_column="user_id")
        parquet_datamodule = ParquetModule(
            batch_size=64,
            metadata=metadata,
            transforms=transforms,
            train_path="data/train.parquet",
        )
