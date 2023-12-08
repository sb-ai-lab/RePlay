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
        __init__,
        keep_common_query_ids,
        __len__,
        cardinality_callback,
        get_query_id,
        get_all_query_ids,
        get_sequence_length,
        get_max_sequence_length,
        get_sequence,
        get_sequence_by_query_id,
        filter_by_query_id,
        schema

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
