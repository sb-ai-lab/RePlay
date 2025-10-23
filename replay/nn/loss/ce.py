from typing import Callable, Optional

import torch

from replay.data.nn import TensorMap

from .base import SampledLossBase, mask_negative_logits


class CE(torch.nn.Module):
    """
    Full Cross-Entropy loss
    Calculates loss over all items catalog.
    """

    def __init__(self, **kwargs):
        """
        To calculate the loss, ``torch.nn.CrossEntropyLoss`` is used.
        You can pass all parameters for initializing the object via kwargs.
        """
        super().__init__()
        self._loss = torch.nn.CrossEntropyLoss(**kwargs)
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
        :param model_embeddings: model output of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param positive_labels: labels of positive events
            of shape ``(batch_size, sequence_length, num_positives)``.
        :param target_padding_mask: padding mask corresponding for `positive_labels`
            of shape ``(batch_size, sequence_length, num_positives)``.
        :return: computed loss value.
        """
        if positive_labels.size(-1) != 1:
            msg = "The case of multi-positive labels is not supported in the CE loss"
            raise NotImplementedError(msg)
        logits: torch.Tensor = self.logits_callback(model_embeddings)  # [batch_size, seq_len, vocab_size]
        labels = positive_labels.masked_fill(
            mask=(~target_padding_mask),
            value=self._loss.ignore_index,
        )  # [batch_size, seq_len, 1]

        # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        logits_flat = logits.view(-1, logits.size(-1))
        # [batch_size, seq_len, 1] -> [batch_size * seq_len]
        labels_flat: torch.LongTensor = labels.view(-1)
        loss = self._loss(logits_flat, labels_flat)
        return loss


class CEWeighted(CE):
    """
    Full Cross-Entropy loss
    Calculates loss over all items catalog.

    In addition to calculating the standard loss,
    weights are applied for each sample.
    Therefore, it is expected that the sample weights will be in the generated batch,
    which is fed into the model.
    """

    def __init__(
        self,
        feature_name: str,
        **kwargs,
    ):
        """
        To calculate the loss, ``torch.nn.CrossEntropyLoss`` is used with the parameter ``reduction="none"``.
        You can pass all other parameters for initializing the object via kwargs.

        :param feature_name: the name of the key in the batch.
            The tensor is expected to contain sample weights.
        """
        super().__init__()
        self.feature_name = feature_name
        self._loss = torch.nn.CrossEntropyLoss(reduction="none", **kwargs)

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,  # noqa: ARG002
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        forward(model_embeddings, feature_tensors, positive_labels, target_padding_mask)
        :param feature_tensors: a dictionary of tensors from dataloader.
            This dictionary is expected to contain a key with the name ``feature_name``,
            which is specified in the constructor.
            Expected shape of tensor ``(batch_size, sequence_length, num_positives)``.
        :param model_embeddings: model output of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param positive_labels: labels of positive events
            of shape ``(batch_size, sequence_length, num_positives)``.
        :param target_padding_mask: padding mask corresponding for `positive_labels`
            of shape ``(batch_size, sequence_length, num_positives)``.
        :return: computed loss value.
        """
        loss: torch.Tensor = super().forward(
            model_embeddings,
            None,
            positive_labels,
            None,
            None,
            target_padding_mask,
        )
        sample_weight = feature_tensors[self.feature_name]
        loss = (loss * sample_weight).mean()
        return loss


class CESampled(SampledLossBase):
    """
    Sampled Cross-Entropy loss (Cross-Entropy with negative sampling).
    Calculates loss between one positive item and K negatively sampled items.

    The loss supports the calculation of logits for the case of multi-positive labels
    (there are several labels for each position in the sequence).
    """

    def __init__(
        self,
        negative_labels_ignore_index: int = -100,
        **kwargs,
    ):
        """
        To calculate the loss, ``torch.nn.CrossEntropyLoss`` is used.
        You can pass all parameters for initializing the object via kwargs.

        :param negative_labels_ignore_index: padding value for negative labels.
            This may be the case when negative labels
            are formed at the preprocessing level, rather than the negative sampler.
            The index is ignored and does not contribute to the loss.
            Default: ``-100``.
        """
        super().__init__()
        self.negative_labels_ignore_index = negative_labels_ignore_index
        self._loss = torch.nn.CrossEntropyLoss(**kwargs)
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

        # [masked_batch_size, num_negatives] - assign low values to some negative logits
        negative_logits = mask_negative_logits(
            negative_logits,
            negative_labels,
            positive_labels,
            self.negative_labels_ignore_index,
        )
        # [masked_batch_size, 1 + num_negatives] - all logits
        logits = torch.cat((positive_logits, negative_logits), dim=-1)
        # [masked_batch_size] - positives are always at 0 position for all recommendation points
        target = torch.zeros(positive_logits.size(0), dtype=torch.long, device=logits.device)
        # [masked_batch_size] - loss for all recommendation points
        loss = self._loss(logits, target)
        return loss


class CESampledWeighted(CESampled):
    """
    Sampled Cross-Entropy loss (Cross-Entropy with negative sampling).
    Calculates loss between one positive item and K negatively sampled items.

    In addition to calculating the standard loss,
    weights are applied for each sample.
    Therefore, it is expected that the sample weights will be in the generated batch,
    which is fed into the model.

    The loss supports the calculation of logits for the case of multi-positive labels
    (there are several labels for each position in the sequence).
    """

    def __init__(
        self,
        feature_name: str,
        negative_labels_ignore_index: int = -100,
        **kwargs,
    ):
        """
        To calculate the loss, ``torch.nn.CrossEntropyLoss`` is used with the parameter ``reduction="none"``.
        You can pass all other parameters for initializing the object via kwargs.

        :param feature_name: the name of the key in the batch.
            The tensor is expected to contain sample weights.
        :param negative_labels_ignore_index: padding value for negative labels.
            This may be the case when negative labels
            are formed at the preprocessing level, rather than the negative sampler.
            The index is ignored and does not contribute to the loss.
            Default: ``-100``.
        """
        super().__init__(negative_labels_ignore_index=negative_labels_ignore_index)
        self.feature_name = feature_name
        self._loss = torch.nn.CrossEntropyLoss(reduction="none", **kwargs)

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,  # noqa: ARG002
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        forward(model_embeddings, feature_tensors, positive_labels, negative_labels, target_padding_mask)
        :param model_embeddings: model output of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param feature_tensors: a dictionary of tensors from dataloader.
            This dictionary is expected to contain a key with the name ``feature_name``,
            which is specified in the constructor.
            Expected shape of tensor ``(batch_size, sequence_length, num_positives)``.
        :param positive_labels: labels of positive events
            of shape ``(batch_size, sequence_length, num_positives)``.
        :param negative_labels: labels of sampled negative events of shape (num_negatives).
        :param target_padding_mask: padding mask corresponding for ``positive_labels``
            of shape ``(batch_size, sequence_length, num_positives)``
        :return: computed loss value.
        """
        loss: torch.Tensor = super().forward(
            model_embeddings, None, positive_labels, negative_labels, None, target_padding_mask
        )
        sample_weight = feature_tensors[self.feature_name]
        sample_weight = sample_weight[target_padding_mask]
        loss = (loss * sample_weight).mean()
        return loss
