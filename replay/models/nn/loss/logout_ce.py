from typing import Callable, Optional

import torch
from amazme.replay.data.nn import TensorMap

from .base import SampledLossBase, mask_negative_logits, weight_loss_with_sample_weight


class LogOutCE(SampledLossBase):
    def __init__(
        self,
        vocab_size: int,
        padding_idx: int,
        explicit_negatives_padding_value: Optional[int] = None,
        sample_weight_feature_name: Optional[str] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.explicit_negatives_padding_value = explicit_negatives_padding_value
        self.sample_weight_feature_name = sample_weight_feature_name

        reduction = "none" if self.sample_weight_feature_name else "mean"
        self._loss = torch.nn.CrossEntropyLoss(
            ignore_index=self.padding_idx,
            reduction=reduction,
        )

        self._logits_callback = None

    @property
    def logits_callback(self) -> Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        if self._logits_callback is None:
            msg = "The callback for getting logits is not defined"
            raise AttributeError(msg)
        return self._logits_callback

    @logits_callback.setter
    def logits_callback(self, func: Optional[Callable]) -> None:
        self._logits_callback = func

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        initial_target_padding_mask = target_padding_mask
        num_positives = target_padding_mask.size(2)
        # [batch_size, seq_len, num_positives] -> [batch_size * seq_len]
        if num_positives == 1:
            target_padding_mask = target_padding_mask.squeeze(-1)
        else:
            target_padding_mask = target_padding_mask.sum(-1).bool()
        masked_batch_size = target_padding_mask.sum().item()

        logits: torch.Tensor = self.logits_callback(model_embeddings)  # [batch_size, seq_len, vocab_size]
        negative_labels = torch.arange(self.vocab_size, dtype=torch.long, device=padding_mask.device)

        # [batch_size, seq_len, vocab_size] -> [masked_batch_size, vocab_size]
        logits = logits[target_padding_mask]

        # [batch_size, seq_len, num_positives] -> [masked_batch_size, num_positives]
        positive_labels = positive_labels[target_padding_mask]

        # [batch_size, seq_len, num_positives] -> [masked_batch_size, num_positives]
        target_padding_mask = initial_target_padding_mask[target_padding_mask]

        positive_ids = torch.arange(masked_batch_size, dtype=torch.long, device=padding_mask.device)
        # [masked_batch_size, vocab_size] -> [masked_batch_size, num_positives]
        positive_logits = logits[positive_ids.unsqueeze(-1), positive_labels]

        # [masked_batch_size, vocab_size] - assign low values to some negative logits
        negative_logits = mask_negative_logits(
            logits,
            negative_labels,
            positive_labels,
            self.explicit_negatives_padding_value,
        )

        # [masked_batch_size, num_negatives] -> [masked_batch_size, 1, num_negatives]
        negative_logits = negative_logits.unsqueeze(-2)
        # [masked_batch_size, 1, num_negatives] -> [masked_batch_size, num_positives, num_negatives]
        negative_logits = negative_logits.repeat(1, target_padding_mask.size(-1), 1)
        # [masked_batch_size, num_positives, num_negatives] -> [masked_batch_size, num_negatives]
        negative_logits = negative_logits[target_padding_mask]
        # [masked_batch_size, num_positives] -> [masked_batch_size]
        positive_logits = positive_logits[target_padding_mask]
        # [masked_batch_size] -> [masked_batch_size, 1]
        positive_logits = positive_logits.unsqueeze(-1)

        # [masked_batch_size, 1 + num_negatives] - all logits
        logits = torch.cat((positive_logits, negative_logits), dim=-1)
        # [masked_batch_size] - positives are always at 0 position for all recommendation points
        target = torch.zeros(logits.size(0), dtype=torch.long, device=padding_mask.device)
        # [masked_batch_size] - loss for all recommendation points
        loss = self._loss(logits, target)
        loss = weight_loss_with_sample_weight(
            feature_tensors,
            initial_target_padding_mask,
            loss,
            self.sample_weight_feature_name,
        )
        return loss
