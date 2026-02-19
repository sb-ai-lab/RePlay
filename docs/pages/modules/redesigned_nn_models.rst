SasRec
======

SasRec
______
.. autoclass:: replay.nn.sequential.SasRec
   :members: __init__, forward, from_params

SasRec Building Blocks
______________________

SasRecBody
``````````````````````````````

.. autoclass:: replay.nn.sequential.SasRecBody
   :members: __init__, forward

SasRecTransformerLayer
``````````````````````````````
.. autoclass:: replay.nn.sequential.SasRecTransformerLayer
   :members: __init__, forward

PositionAwareAggregator
``````````````````````````````
.. autoclass:: replay.nn.sequential.PositionAwareAggregator
   :members: __init__, forward

MultiHead Differential Attention
````````````````````````````````

MultiHeadDifferentialAttention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: replay.nn.attention.MultiHeadDifferentialAttention
   :members: __init__, forward

DiffTransformerBlock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: replay.nn.sequential.DiffTransformerBlock
   :members: __init__, forward

DiffTransformerLayer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: replay.nn.sequential.DiffTransformerLayer
   :members: __init__, forward


SasRec Transforms
_________________
.. autofunction:: replay.nn.transform.template.make_default_sasrec_transforms

TwoTower
=========

.. _TwoTower:

TwoTower
_________
.. autoclass:: replay.nn.sequential.TwoTower
   :members: __init__, forward, from_params

TwoTower Building Blocks
_________________________

TwoTowerBody
``````````````````````````````

.. autoclass:: replay.nn.sequential.TwoTowerBody

QueryTower
``````````````````````````````

.. autoclass:: replay.nn.sequential.QueryTower
   :members: __init__, forward

ItemTower
``````````````````````````````

.. autoclass:: replay.nn.sequential.ItemTower
   :members: __init__, forward

FeaturesReader
``````````````````````````````

.. autoclass:: replay.nn.sequential.twotower.FeaturesReader
   :members: __init__

TwoTower Transforms
___________________
.. autofunction:: replay.nn.transform.template.make_default_twotower_transforms


.. _Losses:

Losses
======

`BCE`_, `BCESampled`_, `CESampled`_, `LogInCE`_, `LogInCESampled`_, `LogOutCE`_ support the calculation of logits for the case of multi-positive labels (there are several labels for each position in the sequence).
Source of multi-positive labels: https://arxiv.org/abs/2205.04507

BCE
___
.. autoclass:: replay.nn.loss.BCE
   :members: forward

BCESampled
__________
.. autoclass:: replay.nn.loss.BCESampled
   :members: __init__, forward

CE
___
.. autoclass:: replay.nn.loss.CE
   :members: __init__, forward

CESampled
__________
.. autoclass:: replay.nn.loss.CESampled
   :members: __init__, forward

LogInCE
_______
.. autoclass:: replay.nn.loss.LogInCE
   :members: __init__, forward

LogInCESampled
______________
.. autoclass:: replay.nn.loss.LogInCESampled
   :members: __init__, forward

LogOutCE
________
.. autoclass:: replay.nn.loss.LogOutCE
   :members: __init__, forward

Scalable Cross Entropy
______________________

SCEParams
`````````
.. autoclass:: replay.models.nn.loss.SCEParams

ScalableCrossEntropyLoss
````````````````````````
.. autoclass:: replay.models.nn.loss.ScalableCrossEntropyLoss
   :members: __init__, __call__

Model Building Blocks
======================
Building blocks for neural network models.

Embeddings
__________

SequenceEmbedding
`````````````````
.. autoclass:: replay.nn.embedding.SequenceEmbedding
   :members: __init__, forward, embeddings_dim, get_item_weights

CategoricalEmbedding
````````````````````
.. autoclass:: replay.nn.embedding.CategoricalEmbedding
   :members: __init__, forward, embedding_dim, weight

NumericalEmbedding
``````````````````
.. autoclass:: replay.nn.embedding.NumericalEmbedding
   :members: __init__, forward, embedding_dim, weight

IdentityEmbedding
``````````````````
.. autoclass:: replay.nn.embedding.IdentityEmbedding
   :members: __init__, forward, embedding_dim, weight


Aggregators
___________
The main purpose of these modules is to aggregate embeddings.
But in general, you can use them to aggregate any type of tensors.

SumAggregator
`````````````
.. autoclass:: replay.nn.agg.SumAggregator
   :members: __init__, forward, embedding_dim

ConcatAggregator
````````````````
.. autoclass:: replay.nn.agg.ConcatAggregator
   :members: __init__, forward, embedding_dim


Feed Forward Networks
_____________________

PointWiseFeedForward
````````````````````
.. autoclass:: replay.nn.ffn.PointWiseFeedForward
   :members: __init__, forward

SwiGLU
`````````````
.. autoclass:: replay.nn.ffn.SwiGLU
   :members: __init__, forward

SwiGLUEncoder
`````````````
.. autoclass:: replay.nn.ffn.SwiGLUEncoder
   :members: __init__, forward


Attention Masks
_______________

DefaultAttentionMask
````````````````````
.. autoclass:: replay.nn.mask.DefaultAttentionMask
   :members: __init__, __call__


Transformer Heads
_________________

EmbeddingTyingHead
``````````````````
.. autoclass:: replay.nn.head.EmbeddingTyingHead
   :members: forward

Universal Lighting module
=========================

LightningModule
_______________
.. autoclass:: replay.nn.lightning.LightningModule
   :members: __init__, forward, candidates_to_score

TrainOutput
___________
.. autoclass:: replay.nn.output.TrainOutput
   :members:

InferenceOutput
_______________
.. autoclass:: replay.nn.output.InferenceOutput
   :members:


.. _Transforms:

Transforms for ParquetModule
====================================================

This submodule contains a set of standard PyTorch tensor transformations necessary for neural network models. 
Every Transform (transformation) is a child class of ``torch.nn.Module`` which forward pass takes as input a batch (python dictionary) 
and returns a copy of input batch with some applyed transformation. 

These Transforms are intended for use with the :ref:`Parquet-Module`. `ParquetModule` object gets transformations via `transforms` parameter.

For passing `transforms` parameter correctly, specify a sequence (a list) of transformations for every used data split, for example:

.. code-block:: python

   {
      "train": [NextTokenTransform(label_field="item_id", shift=1), ...],
      "validate": [...]
   }


``ParquetModule`` converts every specified list of transformations into ``torch.nn.Sequential``, which will be applied per batch on device, 
then the resulting batch after all transformations will be used as model input. 

RePlay provides functions that create a standard set of transformations for models that can also be used as the basis 
for custom, more complicated sets of transformations. See :ref:`Standard set of transforms for models <transforms-for-models>`.


CopyTransform
__________________
.. autoclass:: replay.nn.transform.CopyTransform
    :members: __init__

EqualityMaskTransform
__________________________
.. autoclass:: replay.nn.transform.EqualityMaskTransform
    :members: __init__

GroupTransform
__________________
.. autoclass:: replay.nn.transform.GroupTransform
    :members: __init__

RenameTransform
__________________
.. autoclass:: replay.nn.transform.RenameTransform
    :members: __init__

UnsqueezeTransform
__________________
.. autoclass:: replay.nn.transform.UnsqueezeTransform
    :members: __init__

NextTokenTransform
__________________
.. autoclass:: replay.nn.transform.NextTokenTransform
    :members: __init__

TokenMaskTransform
__________________
.. autoclass:: replay.nn.transform.TokenMaskTransform
    :members: __init__

TrimTransform
__________________
.. autoclass:: replay.nn.transform.TrimTransform
    :members: __init__

SelectTransform
______________________
.. autoclass:: replay.nn.transform.SelectTransform
    :members: __init__

SequenceRollTransform
______________________
.. autoclass:: replay.nn.transform.SequenceRollTransform
    :members: __init__

UniformNegativeSamplingTransform
_________________________________
.. autoclass:: replay.nn.transform.UniformNegativeSamplingTransform
    :members: __init__

MultiClassNegativeSamplingTransform
____________________________________
.. autoclass:: replay.nn.transform.MultiClassNegativeSamplingTransform
    :members: __init__


.. _transforms-for-models:

Standard set of transforms for models
_____________________________________

SasRec Transforms
````````````````````
.. autofunction:: replay.nn.transform.template.make_default_sasrec_transforms
    :noindex:

TwoTower Transforms
````````````````````
.. autofunction:: replay.nn.transform.template.make_default_twotower_transforms
    :noindex:

Easy training, validation and inference with Lightning
========================================================
Replay provides Callbacks and Postprocessors to make the model training, validation and inference process as convenient as possible.

During training/validation:

   You can define the list of validation metrics and the model is determined to be the best and is saved if the metric updates its value during validation.

During inference:

   You can get the recommendations in the following formats: ``PySpark DataFrame``, ``Pandas DataFrame``, ``Polars DataFrame``, ``PyTorch tensors``.
   Each of the types corresponds a callback. You can filter the results using postprocessors strategy.
   In addition to outputting logits (scores) from the model, you can output any hidden states using ``HiddenStateCallback``.

For a better understanding, you should look at examples of using neural network models.


Callbacks
_________

ComputeMetricsCallback
`````````````````````````
.. autoclass:: replay.nn.lightning.callback.ComputeMetricsCallback
   :members: __init__

PandasTopItemsCallback
````````````````````````
.. autoclass:: replay.nn.lightning.callback.PandasTopItemsCallback
   :members: __init__, get_result

PolarsTopItemsCallback
````````````````````````
.. autoclass:: replay.nn.lightning.callback.PolarsTopItemsCallback
   :members: __init__, get_result

SparkTopItemsCallback
```````````````````````
.. autoclass:: replay.nn.lightning.callback.SparkTopItemsCallback
   :members: __init__, get_result

TorchTopItemsCallback
```````````````````````
.. autoclass:: replay.nn.lightning.callback.TorchTopItemsCallback
   :members: __init__, get_result

HiddenStatesCallback
`````````````````````
.. autoclass:: replay.nn.lightning.callback.HiddenStatesCallback
   :members: __init__, get_result

Postprocessors
______________

PostprocessorBase
`````````````````
.. autoclass:: replay.nn.lightning.postprocessor.PostprocessorBase
   :members: __init__, on_validation, on_prediction

SeenItemsFilter
```````````````
.. autoclass:: replay.nn.lightning.postprocessor.SeenItemsFilter
   :members: __init__, on_validation, on_prediction
