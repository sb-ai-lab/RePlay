from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from .dataset import (
        Bert4RecPredictionBatch,
        Bert4RecPredictionDataset,
        Bert4RecTrainingBatch,
        Bert4RecTrainingDataset,
        Bert4RecValidationBatch,
        Bert4RecValidationDataset,
        Bert4RecUniformMasker,
    )
    from .lightning import Bert4Rec
    from .model import Bert4RecModel
