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


ParquetDatamodule
_________________

.. autoclass:: replay.data.nn.ParquetModule
    :members: __init__


Transforms
__________

This submodule contains a set of standard torch transformations necessary for training recommendation neural network models. 
These Transforms are intended for use with the ParquetModule. For applying specify a sequence of transformations for every data split as ParquetModule's ``transforms`` parameter. 
Specified transformations will be applyed per batch on device, then the resulting batch will be used as model input. 


BatchingTransform
`````````````````
.. autoclass:: replay.nn.transforms.BatchingTransform
    :members: __init__

CopyTransform
`````````````
.. autoclass:: replay.nn.transforms.CopyTransform
    :members: __init__

GroupTransform
``````````````
.. autoclass:: replay.nn.transforms.GroupTransform
    :members: __init__

RenameTransform
```````````````
.. autoclass:: replay.nn.transforms.RenameTransform
    :members: __init__

UnsqueezeTransform
``````````````````
.. autoclass:: replay.nn.transforms.UnsqueezeTransform
    :members: __init__

NextTokenTransform
``````````````````
.. autoclass:: replay.nn.transforms.NextTokenTransform
    :members: __init__

TokenMaskTransform
``````````````````
.. autoclass:: replay.nn.transforms.TokenMaskTransform
    :members: __init__

SequenceRollTransform
`````````````````````
.. autoclass:: replay.nn.transforms.SequenceRollTransform
    :members: __init__

UniformNegativeSamplingTransform
````````````````````````````````
.. autoclass:: replay.nn.transforms.UniformNegativeSamplingTransform


Parquet processing
______________

This module contains the implementation of ``ParquetDataset`` - a combination of PyTorch-compatible dataset and sampler designed for working with the Parquet file format.
The main advantages offered by this dataset are:

1. Batch-wise reading and processing of data, allowing it to work with large datasets in memory-constrained settings.
2. Full built-in support for Torch's Distributed Data Parallel mode.
3. Automatic padding of data according to the provided schema.

``ParquetDataset`` is primarily configured using column schemas - dictionaries containing target columns as keys and their shape/padding specifiers as values.
An example column schema:
.. code-block:: python
    schema = {
        "user_id": {} # Empty metadata represents a categorical column.
        "seq_1": {"shape": 5} # 1-D sequences of length 5
        "seq_1": {"shape": (5, 6), "padding_value": -1} # 2-D sequences with custom padding values
    }

Of note: ``ParquetDataset`` only supports numerical values - ensure that all of your data is boolean/integer/float to properly use this class.

ParquetDataset
`````````````````````
.. autoclass:: replay.data.nn.parquet.ParquetDataset
    :members: __init__
