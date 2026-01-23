import pytest
import torch

from replay.nn.lightning import LightningModule


def test_module_set_invalid_candidates_to_score(sasrec_model):
    sasrec = LightningModule(sasrec_model)
    with pytest.raises(ValueError):
        sasrec.candidates_to_score = torch.LongTensor([0, 0])
