from typing import Callable, Optional, TypedDict

import torch

from replay.data.nn import TensorMap

from .base import SampledLossBase, mask_negative_logits


class LogInCESampledOutput(TypedDict):
    positive_logits: torch.Tensor
    negative_logits: torch.Tensor
    positive_labels: torch.LongTensor
    negative_labels: torch.LongTensor
    target_padding_mask: torch.BoolTensor


class LogInCEBase(SampledLossBase):
    def get_sampled_logits(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,  # [batch_size, seq_len, num_positives]
        negative_labels: torch.LongTensor,  # [num_negatives] or [batch_size, seq_len, num_negatives]
        target_padding_mask: torch.BoolTensor,  # [batch_size, seq_len, num_positives]
    ) -> LogInCESampledOutput:
        """
        The function of calculating positive and negative logits in LogInCE losses.
        Based on the embeddings from the model, positive and negative labels.

        The function supports the calculation of logits for the case of multi-positive labels
        (there are several labels for each position in the sequence).

        :param model_embeddings: Embeddings from the model. This is usually the last hidden state.
            Expected shape: ``(batch_size, sequence_length, embedding_dim)``
        :param positive_labels: a tensor containing labels with positive events.
            Expected shape: ``(batch_size, sequence_length, num_positives)``
        :param negative_labels: a tensor containing labels with negative events.
            Expected shape:
                - ``(batch_size, sequence_length, num_negatives)``.
                - ``(num_negatives)`` - a case where the same negative events are used for the entire batch.
        :param target_padding_mask: Padding mask for ``positive_labels`` (targets).
            ``False`` value indicates that the corresponding ``key`` value will be ignored.
            Expected shape: ``(batch_size, sequence_length, num_positives)``

        :returns: LogInCESampledOutput. A dictionary containing positive and negative logits with labels.
        """
        ################## SHAPE CHECKING STAGE START ##################
        batch_size, seq_len, num_positives = positive_labels.size()
        assert target_padding_mask.size() == (batch_size, seq_len, num_positives)
        num_negatives = negative_labels.size(-1)
        assert negative_labels.size() == (batch_size, seq_len, num_negatives) or negative_labels.dim() == 1
        ################## SHAPE CHECKING STAGE END ##################

        # Get output embedding for every user event
        embedding_dim = model_embeddings.size(-1)
        assert model_embeddings.size() == (batch_size, seq_len, embedding_dim)

        # [batch_size, seq_len, num_positives] -> [batch_size, seq_len]
        masked_target_padding_mask: torch.BoolTensor = target_padding_mask.sum(-1).bool()
        masked_batch_size = masked_target_padding_mask.sum().item()

        # Apply target mask
        # [batch_size, seq_len, emb_dim] -> [masked_batch_size, emb_dim]
        model_embeddings = model_embeddings[masked_target_padding_mask]
        assert model_embeddings.size() == (masked_batch_size, embedding_dim)

        # [batch_size, seq_len, num_positives] -> [masked_batch_size, num_positives]
        positive_labels = positive_labels[masked_target_padding_mask]
        assert positive_labels.size() == (masked_batch_size, num_positives)

        if negative_labels.dim() > 1: # pragma: no cover
            # [batch_size, seq_len, num_negatives] -> [masked_batch_size, num_negatives]
            negative_labels = negative_labels[masked_target_padding_mask]
            assert negative_labels.size() == (masked_batch_size, num_negatives)

        positive_logits = self.logits_callback(model_embeddings, positive_labels)
        assert positive_logits.size() == (masked_batch_size, num_positives)

        negative_logits = self.logits_callback(model_embeddings, negative_labels)
        assert negative_logits.size() == (masked_batch_size, num_negatives)

        # [batch_size, seq_len, num_positives] -> [masked_batch_size, num_positives]
        target_padding_mask = target_padding_mask[masked_target_padding_mask]
        assert target_padding_mask.size() == (masked_batch_size, num_positives)

        return {
            "positive_logits": positive_logits,
            "negative_logits": negative_logits,
            "positive_labels": positive_labels,
            "negative_labels": negative_labels,
            "target_padding_mask": target_padding_mask,
        }


class LogInCE(LogInCEBase):
    """
    LogInCE loss (Log InfoNCE, modification of  Information Noise-Contrastive Estimation loss).

    .. math::

        L_{\\text{InfoNCE}} = -\\log \\frac{\\sum_{p \\in P}
        \\exp(\\mathrm{sim}(q, p))}{\\sum_{p \\in P}
        \\exp(\\mathrm{sim}(q, p)) + \\sum_{n \\in N} \\exp(\\mathrm{sim}(q, n))},

    where q -- query embedding, P -- set of positive logits, N -- set of negative logits,
    :math:`sim(\\cdot, \\cdot)` -- similaruty function.

    The loss supports the calculation of logits for the case of multi-positive labels
    (there are several labels for each position in the sequence).
    """

    def __init__(
        self,
        vocab_size: int,
        log_epsilon: float = 1e-6,
        clamp_border: float = 100.0,
    ):
        """
        :param vocab_size: number of unique items in vocabulary (catalog).
        :param log_epsilon: correction to avoid zero in the logarithm during loss calculating.
            Default: ``1e-6``.
        :param clamp_border: upper bound for clamping loss tensor, lower bound will be setted to ``-clamp_border``.
            Default: ``100.0``.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.log_epsilon = log_epsilon
        self.clamp_border = clamp_border
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
        all_negative_labels = torch.arange(
            self.vocab_size,
            dtype=torch.long,
            device=positive_labels.device,
        )
        sampled = self.get_sampled_logits(
            model_embeddings,
            positive_labels,
            all_negative_labels,
            target_padding_mask,
        )
        positive_logits = sampled["positive_logits"]  # [masked_batch_size, num_positives]
        negative_logits = sampled["negative_logits"]  # [masked_batch_size, num_negatives]
        positive_labels = sampled["positive_labels"]  # [masked_batch_size, num_positives]
        all_negative_labels = sampled["negative_labels"]  # [masked_batch_size, num_negatives] or [num_negatives]
        target_padding_mask = sampled["target_padding_mask"]  # [masked_batch_size, num_positives]

        # [masked_batch_size, num_negatives] - assign low values to some negative logits
        negative_logits = mask_negative_logits(
            negative_logits,
            all_negative_labels,
            positive_labels,
        )

        max_values = torch.max(
            positive_logits.max(-1, keepdim=True).values,
            negative_logits.max(-1, keepdim=True).values,
        )  # [masked_batch_size, 1]
        positive_logits = positive_logits - max_values
        negative_logits = negative_logits - max_values

        positive_logits = torch.exp(positive_logits)
        positive_logits = positive_logits * target_padding_mask
        # [masked_batch_size, num_positives] -> [masked_batch_size]
        positive_logits = positive_logits.sum(-1)

        negative_logits = torch.exp(negative_logits)
        # [masked_batch_size, num_negatives] -> [masked_batch_size]
        negative_logits = negative_logits.sum(-1)

        probabilities = positive_logits / (positive_logits + negative_logits)
        loss = -torch.clamp(
            torch.log(probabilities + self.log_epsilon),
            -self.clamp_border,
            self.clamp_border,
        )
        return loss.mean()


class LogInCESampled(LogInCEBase):
    """
    Sampled version of LogInCE (Log InfoNCE) loss (with negative sampling items).

    .. math::

        L_{\\text{InfoNCE}} = -\\log \\frac{\\sum_{p \\in P} \\exp(\\mathrm{sim}(q, p))}{\\sum_{p \\in P}
        \\exp(\\mathrm{sim}(q, p)) + \\sum_{n \\in N_{\\text{sampled}}} \\exp(\\mathrm{sim}(q, n))},

    where q -- query embedding, P -- set of positive logits, :math:`N_sampled` -- set of negative logits,
    :math:`sim(\\cdot, \\cdot)` -- similaruty function.\n
    Same as ``LogInCE``, the difference in the set of negatives.

    The loss supports the calculation of logits for the case of multi-positive labels
    (there are several labels for each position in the sequence).
    """

    def __init__(self, log_epsilon: float = 1e-6, clamp_border: float = 100.0):
        """
        :param log_epsilon: correction to avoid zero in the logarithm during loss calculating.
            Default: 1e-6.
        :param clamp_border: upper bound for clamping loss tensor, lower bound will be setted to -`clamp_border`.
            Default: 100.0.
        """
        super().__init__()
        self.log_epsilon = log_epsilon
        self.clamp_border = clamp_border
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
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        forward(model_embeddings, positive_labels, negative_labels, target_padding_mask)
        :param model_embeddings: model output of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param positive_labels: labels of positive events
            of shape ``(batch_size, sequence_length, num_positives)``.
        :param negative_labels: labels of sampled negative events of shape (num_negatives).
        :param target_padding_mask: padding mask corresponding for ``positive_labels``
            of shape ``(batch_size, sequence_length, num_positives)``
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
        target_padding_mask = sampled["target_padding_mask"]  # [masked_batch_size, num_positives]

        # [masked_batch_size, num_negatives] - assign low values to some negative logits
        negative_logits = mask_negative_logits(
            negative_logits,
            negative_labels,
            positive_labels,
        )

        max_values = torch.max(
            positive_logits.max(-1, keepdim=True).values,
            negative_logits.max(-1, keepdim=True).values,
        )  # [masked_batch_size, 1]
        positive_logits = positive_logits - max_values
        negative_logits = negative_logits - max_values

        positive_logits = torch.exp(positive_logits)
        positive_logits = positive_logits * target_padding_mask
        # [masked_batch_size, num_positives] -> [masked_batch_size]
        positive_logits = positive_logits.sum(-1)

        negative_logits = torch.exp(negative_logits)
        # [masked_batch_size, num_negatives] -> [masked_batch_size]
        negative_logits = negative_logits.sum(-1)

        probabilities = positive_logits / (positive_logits + negative_logits)
        loss = -torch.clamp(
            torch.log(probabilities + self.log_epsilon),
            -self.clamp_border,
            self.clamp_border,
        )
        return loss.mean()
