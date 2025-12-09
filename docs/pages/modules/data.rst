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

.. autoclass:: replay.data.nn.module.datamodule.ParquetModule
    :members: __init__


Transforms
__________

This submodule contains a set of standard torch transformations necessary for training recommendation neural network models. 
These Transforms are intended for use with the ParquetModule. For applying specify a sequence of transformations for every data split as ParquetModule's ``transforms`` parameter. 
Specified transformations will be applyed per batch on device, then the resulting batch will be used as model input. 


BatchingTransform
`````````````````
.. autoclass:: replay.data.nn.transforms.BatchingTransform
    :members: __init__

CopyTransform
`````````````
.. autoclass:: replay.data.nn.transforms.CopyTransform
    :members: __init__

GroupTransform
``````````````
.. autoclass:: replay.data.nn.transforms.GroupTransform
    :members: __init__

RenameTransform
```````````````
.. autoclass:: replay.data.nn.transforms.RenameTransform
    :members: __init__

UnsqueezeTransform
``````````````````
.. autoclass:: replay.data.nn.transforms.UnsqueezeTransform
    :members: __init__

NextTokenTransform
``````````````````
.. autoclass:: replay.data.nn.transforms.NextTokenTransform
    :members: __init__

TokenMaskTransform
``````````````````
.. autoclass:: replay.data.nn.transforms.TokenMaskTransform
    :members: __init__

SequenceRollTransform
`````````````````````
.. autoclass:: replay.data.nn.transforms.SequenceRollTransform
    :members: __init__

UniformNegativeSamplingTransform
````````````````````````````````
.. autoclass:: replay.data.nn.transforms.UniformNegativeSamplingTransform
    :members: __init__
