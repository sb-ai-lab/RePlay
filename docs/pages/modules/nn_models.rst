.. _Transforms:

Transforms for ParquetModule
====================================================

This submodule contains a set of standard torch transformations necessary for training recommendation neural network models. 
These Transforms are intended for use with the :ref:`Parquet-Module`. For applying specify a sequence of transformations for every data split as ParquetModule's ``transforms`` parameter. 
Specified transformations will be applyed per batch on device, then the resulting batch will be used as model input. 


BatchingTransform
__________________
.. autoclass:: replay.nn.transforms.BatchingTransform
    :members: __init__

CopyTransform
__________________
.. autoclass:: replay.nn.transforms.CopyTransform
    :members: __init__

GroupTransform
__________________
.. autoclass:: replay.nn.transforms.GroupTransform
    :members: __init__

RenameTransform
__________________
.. autoclass:: replay.nn.transforms.RenameTransform
    :members: __init__

UnsqueezeTransform
__________________
.. autoclass:: replay.nn.transforms.UnsqueezeTransform
    :members: __init__

NextTokenTransform
__________________
.. autoclass:: replay.nn.transforms.NextTokenTransform
    :members: __init__

TokenMaskTransform
__________________
.. autoclass:: replay.nn.transforms.TokenMaskTransform
    :members: __init__

SequenceRollTransform
______________________
.. autoclass:: replay.nn.transforms.SequenceRollTransform
    :members: __init__

UniformNegativeSamplingTransform
_________________________________
.. autoclass:: replay.nn.transforms.UniformNegativeSamplingTransform