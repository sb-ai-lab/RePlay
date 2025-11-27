from typing import Callable, Optional

import torch

from replay.data.nn import TensorMap

from .base import SampledLossBase, mask_negative_logits


class BCE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._loss = torch.nn.BCEWithLogitsLoss(reduction="sum")
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
        feature_tensors: TensorMap,  # noqa: ARG002
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,  # noqa: ARG002
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        logits = self.logits_callback(model_embeddings)

        # [batch_size, seq_len, num_positives] -> [batch_size, seq_len]
        if target_padding_mask.size(-1) == 1:
            target_padding_mask.squeeze_(-1)
        else:
            target_padding_mask = target_padding_mask.sum(-1).bool()

        # Take only logits which correspond to non-padded tokens
        # [batch_size, seq_len, vocab_size] -> [masked_batch_size, vocab_size]
        logits = logits[target_padding_mask]

        # [batch_size, seq_len, num_positives] -> [masked_batch_size, num_positives]
        labels = positive_labels[target_padding_mask]

        bce_labels = torch.zeros_like(logits)

        # Fill positives with ones, all negatives are zeros
        bce_labels.scatter_(
            dim=-1,
            index=labels,
            value=1,
        )

        loss = self._loss(logits, bce_labels) / logits.size(0)
        return loss


class BCESampled(SampledLossBase):
    def __init__(self, log_epsilon: float = 1e-6, clamp_border: float = 100.0):
        super().__init__()
        self.log_epsilon = log_epsilon
        self.clamp_border = clamp_border
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
        feature_tensors: TensorMap,  # noqa: ARG002
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        sampled = self.get_sampled_logits(
            model_embeddings,
            positive_labels,
            negative_labels,
            target_padding_mask,
        )
        positive_logits = sampled["positive_logits"]  # [masked_batch_size, 1]
        negative_logits = sampled["negative_logits"]  # [masked_batch_size, num_negatives]
        positive_labels = sampled["positive_labels"]  # [masked_batch_size, num_positives]
        negative_labels = sampled["negative_labels"]  # [masked_batch_size, num_negatives] or [num_negatives]

        # Reject negative samples matching target label & correct for remaining samples
        negative_logits = mask_negative_logits(
            negative_logits,
            negative_labels,
            positive_labels,
        )

        positive_prob = torch.sigmoid(positive_logits)
        negative_prob = torch.sigmoid(negative_logits)

        positive_loss = torch.clamp(
            torch.log((positive_prob) + self.log_epsilon), -self.clamp_border, self.clamp_border
        ).sum()
        negative_loss = torch.clamp(
            torch.log((1 - negative_prob) + self.log_epsilon), -self.clamp_border, self.clamp_border
        ).sum()

        loss = -(positive_loss + negative_loss)
        loss /= positive_logits.size(0)

        return loss
