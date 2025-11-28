from typing import Callable, Optional

import torch

from replay.data.nn import TensorMap

from .base import SampledLossBase, mask_negative_logits


class CE(torch.nn.Module):
    """
    Full Cross-Entropy loss, calculated over all items catalog.
    """

    def __init__(self, padding_idx: int):
        """
        :param padding_idx: padding id for label to ignore during loss calculation.
        """
        super().__init__()
        self.padding_idx = padding_idx
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=padding_idx)
        self._logits_callback = None

    @property
    def logits_callback(self) -> Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
        Getter method for the logits computation function.
        Method for logits computation from the model head should be setted as this function
                                                                after the loss initialization.
        The function is expected to receive a tensor of output model embeddings
                    and a tensor of item embeddings optionally, and to return a logits tensor.
        :return: callable function
        """
        if self._logits_callback is None:
            msg = "The callback for getting logits is not defined"
            raise AttributeError(msg)
        return self._logits_callback

    @logits_callback.setter
    def logits_callback(self, func: Optional[Callable]) -> None:
        """
        Setter method for the logits computation function.
        Method for logits computation from the model head should be setted as this function
                                                                after the loss initialization.
        The function is expected to receive a tensor of output model embeddings
                    and a tensor of item embeddings optionally, and to return a logits tensor.
        :param func: callable function.
        """
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
        """
        :param model_embeddings: model output of shape (batch_size, sequence_length, embedding_dim).
        :param positive_labels: ground truth labels of shape (batch_size, sequence_length, 1).
        :param target_padding_mask: padding mask for `positive_labels` of shape (batch_size, sequence_length, 1).
        :return: loss value.
        """
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
    """
    Sampled Cross-Entropy loss (Cross-Entropy with negative sampling), 
    calculated between one positive item and K negatively sampled items.

    The loss supports the calculation of logits for the case of multi-positive labels
        (there are several labels for each position in the sequence).
    """

    def __init__(self, padding_idx: int):
        """
        :param padding_idx: :param padding_idx: padding id for label to be ignored during loss calculation.
        """
        super().__init__()
        self.padding_idx = padding_idx
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self._logits_callback = None

    @property
    def logits_callback(self) -> Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
        Getter method for the logits computation function.
        Method for logits computation from the model head should be setted as this function
                                                                after the loss initialization.
        The function is expected to receive a tensor of output model embeddings
                    and a tensor of item embeddings optionally, and to return a logits tensor.
        :return: callable function
        """
        if self._logits_callback is None:
            msg = "The callback for getting logits is not defined"
            raise AttributeError(msg)
        return self._logits_callback

    @logits_callback.setter
    def logits_callback(self, func: Optional[Callable]) -> None:
        """
        Setter method for the logits computation function.
        Method for logits computation from the model head should be setted as this function
                                                                after the loss initialization.
        The function is expected to receive a tensor of output model embeddings
                    and a tensor of item embeddings optionally, and to return a logits tensor.
        :param func: callable function.
        """
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
        """
        :param model_embeddings: model output of shape (batch_size, sequence_length, embedding_dim).
        :param positive_labels: ground truth labels of positive events of shape (batch_size, sequence_length, num_positives).
        :param negative_labels: labels of sampled negative events of shape (num_negatives).
        :param target_padding_mask: padding mask corresponding for `positive_labels` of shape (batch_size, sequence_length, num_positives).
        :return: computed loss value.
        """
        sampled = self.get_sampled_logits(
            model_embeddings,
            positive_labels,
            negative_labels,
            target_padding_mask,
        )
        positive_logits = sampled["positive_logits"]  # [masked_batch_size, num_positives]
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
        target = torch.zeros(positive_logits.size(0), dtype=torch.long, device=logits.device)
        # [masked_batch_size] - loss for all recommendation points
        loss = self._loss(logits, target)
        return loss
