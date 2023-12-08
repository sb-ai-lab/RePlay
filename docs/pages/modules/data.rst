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
