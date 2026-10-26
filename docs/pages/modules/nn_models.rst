BERT4Rec
========

BERT4Rec
________
.. autoclass:: replay.models.nn.sequential.BERT4Rec
   :members: __init__, forward, predict

BERT4RecModel
_____________
.. autoclass:: replay.models.nn.sequential.bert4rec.BERT4RecModel
   :members: __init__, forward, predict, forward_step, get_logits, get_query_embeddings

BERT4RecTrainingDataset
_______________________
.. autoclass:: replay.models.nn.sequential.bert4rec.BERT4RecTrainingDataset
   :members: __init__

BERT4RecValidationDataset
_________________________
.. autoclass:: replay.models.nn.sequential.bert4rec.BERT4RecValidationDataset
   :members: __init__

BERT4RecPredictionDataset
_________________________
.. autoclass:: replay.models.nn.sequential.bert4rec.BERT4RecPredictionDataset
   :members: __init__

BERT4RecTrainingBatch
_____________________
.. autoclass:: replay.models.nn.sequential.bert4rec.BERT4RecTrainingBatch
   :members:

BERT4RecValidationBatch
_______________________
.. autoclass:: replay.models.nn.sequential.bert4rec.BERT4RecValidationBatch
   :members:

BERT4RecPredictionBatch
_______________________
.. autoclass:: replay.models.nn.sequential.bert4rec.BERT4RecPredictionBatch
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
Sequential models like SasRec and BERT4Rec can be converted to ONNX format for fast inference on CPU.

SasRecCompiled
______________
.. autoclass:: replay.models.nn.sequential.compiled.SasRecCompiled
   :members: compile, predict

BERT4RecCompiled
________________
.. autoclass:: replay.models.nn.sequential.compiled.BERT4RecCompiled
   :members: compile, predict
