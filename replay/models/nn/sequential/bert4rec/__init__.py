from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from .dataset import (
        BERT4RecPredictionBatch,
        BERT4RecPredictionDataset,
        BERT4RecTrainingBatch,
        BERT4RecTrainingDataset,
        BERT4RecUniformMasker,
        BERT4RecValidationBatch,
        BERT4RecValidationDataset,
    )
    from .lightning import BERT4Rec
    from .model import BERT4RecModel
