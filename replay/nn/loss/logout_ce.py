from typing import Callable, Optional

import torch

from replay.data.nn import TensorMap

from .base import mask_negative_logits


class LogOutCE(torch.nn.Module):
    """
    LogOutCE loss (InfoNCE, Information Noise-Contrastive Estimation loss).

        .. math::

            L_{\\text{InfoNCE}} = - \\sum_{p \\in P} \\log \\frac{ \\exp(\\mathrm{sim}(q, p))}
            {\\exp(\\mathrm{sim}(q, p))
            + \\sum_{n \\in N} \\exp(\\mathrm{sim}(q, n))}.

    where q -- query embedding, P -- set of positive logits, N -- set of negative logits,
    :math:`sim(\\cdot, \\cdot)` -- similaruty function.\n

    The loss supports the calculation of logits for the case of multi-positive labels
    (there are several labels for each position in the sequence).
    """

    def __init__(self, vocab_size: int, padding_idx: int):
        """
        :param vocab_size: number of unique items in vocabulary (catalog).
        :param padding_idx: padding id for label to be ignored during loss calculation.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self._loss = torch.nn.CrossEntropyLoss(ignore_index=self.padding_idx)
        self._logits_callback = None

    @property
    def logits_callback(
        self,
    ) -> Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
        Property for calling a function for the logits computation.\n

        This function is expected to receive model's last hidden state
                    and optionally item IDs, and return a logits tensor.

        It is expected that the corresponding head model method will be used as this function,
        for example, the ``get_logits`` method of the ``SasRec`` class.

        :return: callable function.
        """
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
        """
        forward(model_embeddings, positive_labels, target_padding_mask)
        **Note**: At forward pass, the whole catalog of items is used as negatives.
        Next, negative logits, corresponding to positions where negative labels
        coincide with positive ones, are masked.

        :param model_embeddings: model output of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param positive_labels: ground truth labels of positive events
            of shape (batch_size, sequence_length, num_positives).
        :param target_padding_mask: padding mask corresponding for ``positive_labels``
            of shape (batch_size, sequence_length, num_positives).
        :return: computed loss value.
        """
        initial_target_padding_mask = target_padding_mask
        num_positives = target_padding_mask.size(2)
        # [batch_size, seq_len, num_positives] -> [batch_size * seq_len]
        if num_positives == 1:
            target_padding_mask = target_padding_mask.squeeze(-1)
        else:
            target_padding_mask = target_padding_mask.sum(-1).bool()
        masked_batch_size = target_padding_mask.sum().item()

        logits: torch.Tensor = self.logits_callback(
            model_embeddings
        )  # [batch_size, seq_len, vocab_size]
        all_negative_labels = torch.arange(
            self.vocab_size, dtype=torch.long, device=positive_labels.device
        )

        # [batch_size, seq_len, vocab_size] -> [masked_batch_size, vocab_size]
        logits = logits[target_padding_mask]

        # [batch_size, seq_len, num_positives] -> [masked_batch_size, num_positives]
        positive_labels = positive_labels[target_padding_mask]

        # [batch_size, seq_len, num_positives] -> [masked_batch_size, num_positives]
        target_padding_mask = initial_target_padding_mask[target_padding_mask]

        positive_ids = torch.arange(
            masked_batch_size, dtype=torch.long, device=positive_labels.device
        )
        # [masked_batch_size, vocab_size] -> [masked_batch_size, num_positives]
        positive_logits = logits[positive_ids.unsqueeze(-1), positive_labels]

        # [masked_batch_size, vocab_size] - assign low values to some negative logits
        negative_logits = mask_negative_logits(
            logits,
            all_negative_labels,
            positive_labels,
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
        target = torch.zeros(
            logits.size(0), dtype=torch.long, device=positive_labels.device
        )
        # [masked_batch_size] - loss for all recommendation points
        loss = self._loss(logits, target)
        return loss
