Universal Lighting module
=================================================================
LightingModule
______________
.. autoclass:: replay.models.nn.LightingModule
   :members: __init__, forward, candidates_to_score

TrainOutput
___________
.. autoclass:: replay.models.nn.TrainOutput
   :members:

InferenceOutput
_______________
.. autoclass:: replay.models.nn.InferenceOutput
   :members:


SasRec
========

SasRec
__________
.. autoclass:: replay.models.nn.sequential.sasrec_v2.SasRec
   :members: __init__, forward

SasRecBuilder
_____________
.. autoclass:: replay.models.nn.sequential.sasrec_v2.SasRecBuilder
   :members: embedder, attn_mask_builder, embedding_aggregator, encoder, output_normalization, loss, default, build

SasRecBody
__________
.. autoclass:: replay.models.nn.sequential.sasrec_v2.SasRecBody
   :members: __init__, forward

SasRecEmbeddingAggregator
_________________________
.. autoclass:: replay.models.nn.sequential.sasrec_v2.SasRecEmbeddingAggregator
   :members: __init__, forward

Losses
======
BCE
___
.. autoclass:: replay.models.nn.loss.BCE
   :members: __init__, forward

BCESampled
__________
.. autoclass:: replay.models.nn.loss.BCESampled
   :members: __init__, forward

CE
___
.. autoclass:: replay.models.nn.loss.CE
   :members: __init__, forward

CESampled
__________
.. autoclass:: replay.models.nn.loss.CESampled
   :members: __init__, forward

LogInCE
_______
.. autoclass:: replay.models.nn.loss.LogInCE
   :members: __init__, forward

LogInCESampled
______________
.. autoclass:: replay.models.nn.loss.LogInCESampled
   :members: __init__, forward

LogOutCE
________
.. autoclass:: replay.models.nn.loss.LogOutCE
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


Common building blocks
======================
Building blocks for neural network models.

Embedders
_________

CategoricalEmbedding
````````````````````
.. autoclass:: replay.models.nn.sequential.common.embedding.CategoricalEmbedding
   :members: __init__, forward, embedding_dim, weight

NumericalEmbedding
``````````````````
.. autoclass:: replay.models.nn.sequential.common.embedding.NumericalEmbedding
   :members: __init__, forward, embedding_dim, weight

SequentialEmbedder
``````````````````
.. autoclass:: replay.models.nn.sequential.common.embedding.SequentialEmbedder
   :members: __init__, forward, embeddings_dim, get_item_weights


Embedding Aggregators
_____________________

SumAggregator
`````````````
.. autoclass:: replay.models.nn.sequential.common.agg.SumAggregator
   :members: __init__, forward, embedding_dim

ConcatAggregator
````````````````
.. autoclass:: replay.models.nn.sequential.common.agg.ConcatAggregator
   :members: __init__, forward, embedding_dim


Feed Forward Networks
_____________________

PointWiseFeedForward
````````````````````
.. autoclass:: replay.models.nn.sequential.common.ffn.PointWiseFeedForward
   :members: __init__, forward

SwiGLUEncoder
`````````````
.. autoclass:: replay.models.nn.sequential.common.ffn.SwiGLUEncoder
   :members: __init__, forward


Attention Masks
_____________________

DefaultAttentionMaskBuilder
```````````````````````````
.. autoclass:: replay.models.nn.sequential.common.mask.DefaultAttentionMaskBuilder
   :members: __init__, __call__


Transformer Layers
__________________

TransformerLayer
````````````````
.. autoclass:: replay.models.nn.sequential.common.transformer.TransformerLayer
   :members: __init__, forward

DiffTransformerLayer
````````````````````
.. autoclass:: replay.models.nn.sequential.common.diff_transformer.DiffTransformerLayer
   :members: __init__, forward


Transformer Heads
_________________

EmbeddingTyingHead
``````````````````
.. autoclass:: replay.models.nn.sequential.common.head.EmbeddingTyingHead
   :members: __init__, forward


Easy training and validation with Lightning
========================================================
Replay provides Callbacks and Postprocessors to make the model training and validation process as convenient as possible.

During training:

You can define the list of validation metrics and the model is determined to be the best and is saved if the metric
updates its value during validation.

During inference:

You can get the recommendations in four formats: PySpark DataFrame, Pandas DataFrame, Polars DataFrame, PyTorch tensors. Each of the types corresponds a callback.
You can filter the results using postprocessors strategy.

For a better understanding, you should look at examples of using neural network models.

Callbacks
_________

ValidationMetricsCallback
`````````````````````````
.. autoclass:: replay.models.nn.sequential.callbacks.ValidationMetricsCallback
   :members: __init__

SparkPredictionCallback
```````````````````````
.. autoclass:: replay.models.nn.sequential.callbacks.SparkPredictionCallback
   :members: __init__, get_result

PandasPredictionCallback
````````````````````````
.. autoclass:: replay.models.nn.sequential.callbacks.PandasPredictionCallback
   :members: __init__, get_result

TorchPredictionCallback
```````````````````````
.. autoclass:: replay.models.nn.sequential.callbacks.TorchPredictionCallback
   :members: __init__, get_result

QueryEmbeddingsPredictionCallback
`````````````````````````````````
.. autoclass:: replay.models.nn.sequential.callbacks.QueryEmbeddingsPredictionCallback
   :members: __init__, get_result

Postprocessors
______________

RemoveSeenItems
```````````````
.. autoclass:: replay.models.nn.sequential.postprocessors.postprocessors.RemoveSeenItems
   :members: __init__

SampleItems
```````````
.. autoclass:: replay.models.nn.sequential.postprocessors.postprocessors.SampleItems
   :members: __init__
