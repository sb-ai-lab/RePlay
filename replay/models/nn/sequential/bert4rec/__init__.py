from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from .dataset import (
        BertPredictionBatch,
        BertPredictionDataset,
        BertTrainingBatch,
        BertTrainingDataset,
        BertValidationBatch,
        BertValidationDataset,
        UniformBertMasker,
    )
    from .lightning import Bert4Rec
    from .model import BertModel
