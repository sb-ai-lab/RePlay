.. _Transforms:

Transforms for ParquetModule
====================================================

This submodule contains a set of standard PyTorch tensor transformations necessary for neural network models. 
These Transforms are intended for use with the :ref:`Parquet-Module`. For applying specify a sequence of transformations for every data split as ParquetModule's ``transforms`` parameter. 
Specified transformations will be applied per batch on device, then the resulting batch will be used as model input. 


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

SasRecAggregator
``````````````````````````````
.. autoclass:: replay.nn.sequential.SasRecAggregator
   :members: __init__, forward

MultiHead Differential Attention
````````````````````````````````

MultiHeadDifferentialAttention
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: replay.nn.sequential.MultiHeadDifferentialAttention
   :members: __init__, forward

DiffTransformerBlock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: replay.nn.sequential.DiffTransformerBlock
   :members: __init__, forward

DiffTransformerLayer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: replay.nn.sequential.DiffTransformerLayer
   :members: __init__, forward

Bert4Rec
========

Bert4Rec
________
.. autoclass:: replay.models.nn.sequential.Bert4Rec
   :members: __init__, forward, predict

Bert4RecModel
_____________
.. autoclass:: replay.models.nn.sequential.bert4rec.Bert4RecModel
   :members: __init__, forward, predict, forward_step, get_logits, get_query_embeddings

Bert4RecTrainingDataset
_______________________
.. autoclass:: replay.models.nn.sequential.bert4rec.Bert4RecTrainingDataset
   :members: __init__

Bert4RecValidationDataset
_________________________
.. autoclass:: replay.models.nn.sequential.bert4rec.Bert4RecValidationDataset
   :members: __init__

Bert4RecPredictionDataset
_________________________
.. autoclass:: replay.models.nn.sequential.bert4rec.Bert4RecPredictionDataset
   :members: __init__

Bert4RecTrainingBatch
_____________________
.. autoclass:: replay.models.nn.sequential.bert4rec.Bert4RecTrainingBatch
   :members:

Bert4RecValidationBatch
_______________________
.. autoclass:: replay.models.nn.sequential.bert4rec.Bert4RecValidationBatch
   :members:

Bert4RecPredictionBatch
_______________________
.. autoclass:: replay.models.nn.sequential.bert4rec.Bert4RecPredictionBatch
   :members:

SasRec (legacy)
===============

SasRec
________
.. autoclass:: replay.models.nn.sequential.SasRec
   :members: __init__, forward, predict

SasRecModel
_____________
.. autoclass:: replay.models.nn.sequential.sasrec.SasRecModel
   :members: __init__, forward, predict, forward_step, get_logits, get_query_embeddings

SasRecTrainingDataset
_______________________
.. autoclass:: replay.models.nn.sequential.sasrec.SasRecTrainingDataset
   :members: __init__

SasRecValidationDataset
_________________________
.. autoclass:: replay.models.nn.sequential.sasrec.SasRecValidationDataset
   :members: __init__

SasRecPredictionDataset
_________________________
.. autoclass:: replay.models.nn.sequential.sasrec.SasRecPredictionDataset
   :members: __init__

SasRecTrainingBatch
_____________________
.. autoclass:: replay.models.nn.sequential.sasrec.SasRecTrainingBatch
   :members:

SasRecValidationBatch
_______________________
.. autoclass:: replay.models.nn.sequential.sasrec.SasRecValidationBatch
   :members:

SasRecPredictionBatch
_______________________
.. autoclass:: replay.models.nn.sequential.sasrec.SasRecPredictionBatch
   :members:

Compiled sequential models
==========================
Sequential models like SasRec and Bert4Rec can be converted to ONNX format for fast inference on CPU.

SasRecCompiled
______________
.. autoclass:: replay.models.nn.sequential.compiled.SasRecCompiled
   :members: compile, predict

Bert4RecCompiled
________________
.. autoclass:: replay.models.nn.sequential.compiled.Bert4RecCompiled
   :members: compile, predict


Losses
======

**Multi-positive labels support**

   `BCE`_, `BCESampled`_, `CESampled`_, `LogInCE`_, `LogInCESampled`_, `LogOutCE`_ support the calculation of logits for the case of multi-positive labels (there are several labels for each position in the sequence).

   Source: https://arxiv.org/abs/2205.04507

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
.. autoclass:: replay.nn.SequenceEmbedding
   :members: __init__, forward, embeddings_dim, get_item_weights

CategoricalEmbedding
````````````````````
.. autoclass:: replay.nn.CategoricalEmbedding
   :members: __init__, forward, embedding_dim, weight

NumericalEmbedding
``````````````````
.. autoclass:: replay.nn.NumericalEmbedding
   :members: __init__, forward, embedding_dim, weight


Aggregators
___________
The main purpose of these modules is to aggregate embeddings.
But in general, you can use them to aggregate any type of tensors.

SumAggregator
`````````````
.. autoclass:: replay.nn.SumAggregator
   :members: __init__, forward, embedding_dim

ConcatAggregator
````````````````
.. autoclass:: replay.nn.ConcatAggregator
   :members: __init__, forward, embedding_dim


Feed Forward Networks
_____________________

PointWiseFeedForward
````````````````````
.. autoclass:: replay.nn.PointWiseFeedForward
   :members: __init__, forward

SwiGLU
`````````````
.. autoclass:: replay.nn.SwiGLU
   :members: __init__, forward

SwiGLUEncoder
`````````````
.. autoclass:: replay.nn.SwiGLUEncoder
   :members: __init__, forward


Attention Masks
_______________

DefaultAttentionMask
````````````````````
.. autoclass:: replay.nn.DefaultAttentionMask
   :members: __init__, __call__


Transformer Heads
_________________

EmbeddingTyingHead
``````````````````
.. autoclass:: replay.nn.EmbeddingTyingHead
   :members: forward

Universal Lighting module
=========================
LightningModule
_______________
.. autoclass:: replay.nn.lightning.LightningModule
   :members: __init__, forward, candidates_to_score

TrainOutput
___________
.. autoclass:: replay.nn.TrainOutput
   :members:

InferenceOutput
_______________
.. autoclass:: replay.nn.InferenceOutput
   :members:

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
.. autoclass:: replay.nn.lightning.callbacks.ComputeMetricsCallback
   :members: __init__

PandasTopItemsCallback
````````````````````````
.. autoclass:: replay.nn.lightning.callbacks.PandasTopItemsCallback
   :members: __init__, get_result

PolarsTopItemsCallback
````````````````````````
.. autoclass:: replay.nn.lightning.callbacks.PolarsTopItemsCallback
   :members: __init__, get_result

SparkTopItemsCallback
```````````````````````
.. autoclass:: replay.nn.lightning.callbacks.SparkTopItemsCallback
   :members: __init__, get_result

TorchTopItemsCallback
```````````````````````
.. autoclass:: replay.nn.lightning.callbacks.TorchTopItemsCallback
   :members: __init__, get_result

HiddenStatesCallback
`````````````````````
.. autoclass:: replay.nn.lightning.callbacks.HiddenStatesCallback
   :members: __init__, get_result

Postprocessors
______________

PostprocessorBase
`````````````````
.. autoclass:: replay.nn.lightning.postprocessors.PostprocessorBase
   :members: __init__, on_validation, on_prediction

SeenItemsFilter
```````````````
.. autoclass:: replay.nn.lightning.postprocessors.SeenItemsFilter
   :members: __init__, on_validation, on_prediction
