from typing import Callable, Optional

import torch

from replay.data.nn import TensorMap

from .base import SampledLossBase, mask_negative_logits


class CE(torch.nn.Module):
    def __init__(self, padding_idx: int):
        super().__init__()
        self.padding_idx = padding_idx
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
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
        if positive_labels.size(-1) != 1:
            msg = "The case of multi-positive labels is not supported in the CE loss"
            raise NotImplementedError(msg)
        logits: torch.Tensor = self.logits_callback(model_embeddings)  # [batch_size, seq_len, vocab_size]
        labels = positive_labels.masked_fill(
            mask=(~target_padding_mask), value=self.padding_idx
        )  # [batch_size, seq_len, 1]

        # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        logits_flat = logits.view(-1, logits.size(-1))
        # [batch_size, seq_len, 1] -> [batch_size * seq_len]
        labels_flat: torch.LongTensor = labels.view(-1)
        loss = self._loss(logits_flat, labels_flat)
        return loss


class CESampled(SampledLossBase):
    def __init__(self, padding_idx: int):
        super().__init__()
        self.padding_idx = padding_idx
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=self.padding_idx)
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
        padding_mask: torch.BoolTensor,
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

        # [masked_batch_size, num_negatives] - assign low values to some negative logits
        negative_logits = mask_negative_logits(
            negative_logits,
            negative_labels,
            positive_labels,
        )
        # [masked_batch_size, 1 + num_negatives] - all logits
        logits = torch.cat((positive_logits, negative_logits), dim=-1)
        # [masked_batch_size] - positives are always at 0 position for all recommendation points
        target = torch.zeros(positive_logits.size(0), dtype=torch.long, device=padding_mask.device)
        # [masked_batch_size] - loss for all recommendation points
        loss = self._loss(logits, target)
        return loss
