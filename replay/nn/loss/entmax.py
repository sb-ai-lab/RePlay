from collections.abc import Callable

import torch
from torch.autograd.function import once_differentiable

from replay.data.nn import TensorMap

from .base import SampledLossBase, mask_negative_logits


class _Entmax15LossFunction(torch.autograd.Function):
    """Entmax-1.5 loss computation with an explicit backward."""

    @staticmethod
    def forward(ctx, logits, target):
        z = logits - logits.amax(dim=-1, keepdim=True)
        x = z / 2
        dim = -1

        x_sorted, _ = torch.sort(x, dim=dim, descending=True)

        rho = torch.arange(1, x_sorted.shape[dim] + 1, device=x.device, dtype=x.dtype)
        rho = rho.unsqueeze(0)
        mean = x_sorted.cumsum(dim) / rho
        mean_sq = (x_sorted**2).cumsum(dim) / rho
        ss = rho * (mean_sq - mean**2)
        delta = (1 - ss) / rho

        delta_nz = torch.clamp(delta, 0)
        tau = mean - torch.sqrt(delta_nz)

        support_size = (tau <= x_sorted).sum(dim).unsqueeze(dim)
        tau_star = tau.gather(dim, support_size - 1)
        sqrt_p = torch.clamp(x - tau_star, min=0)
        p = sqrt_p.square()
        entropy = (1 - sqrt_p.pow(3).sum(-1)) / 0.75
        safe_z = z.masked_fill(p == 0, 0)
        loss = (p * safe_z).sum(-1) - z.gather(-1, target[:, None]).squeeze(-1) + entropy

        ctx.save_for_backward(
            p.scatter_add_(
                -1,
                target[:, None],
                -torch.ones_like(target[:, None], dtype=p.dtype),
            )
        )

        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (grad_logits,) = ctx.saved_tensors
        return grad_output.unsqueeze(-1) * grad_logits, None


class Entmax15Loss(torch.nn.Module):
    """Entmax-1.5 loss with ignore_index and CE-style reductions."""

    def __init__(self, ignore_index=-100, reduction="mean"):
        super().__init__()
        if reduction not in ("none", "mean", "sum"):
            msg = "reduction must be 'none', 'mean' or 'sum'"
            raise ValueError(msg)
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if input.ndim < 1 or not input.is_floating_point():
            msg = "input must be floating-point logits with a class axis"
            raise ValueError(msg)
        if target.dtype != torch.long:
            msg = "target must have dtype torch.long"
            raise TypeError(msg)
        expected = () if input.ndim == 1 else input.shape[:1] + input.shape[2:]
        if target.shape != expected:
            msg = f"expected target shape {expected}, got {target.shape}"
            raise ValueError(msg)
        classes = input.shape[0] if input.ndim == 1 else input.shape[1]
        if classes == 0:
            msg = "the class axis must be nonempty"
            raise ValueError(msg)
        logits = input.unsqueeze(0) if input.ndim == 1 else input.movedim(1, -1).reshape(-1, classes)
        labels = target.reshape(-1)
        valid = labels != self.ignore_index
        logits = logits[valid]
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        losses = _Entmax15LossFunction.apply(logits, labels[valid])
        if self.reduction == "sum":
            return losses.sum()
        if self.reduction == "mean":
            return losses.mean()
        return losses.new_zeros(labels.shape).masked_scatter(valid, losses).reshape(target.shape)


class PREntmax(torch.nn.Module):
    """
    Full PR-Entmax loss
    Calculates loss over all items catalog.
    """

    def __init__(self, feature_name: str, gamma: float, eps: float = 1e-6, **kwargs):
        """
        To calculate the loss, ``Entmax15Loss`` is used.
        You can pass all parameters for initializing the object via kwargs.

        :param feature_name: Name of the feature containing item popularity.
        :param gamma: Weight of the popularity correction.
        :param eps: Small value added before logarithm to avoid log of zero.
        """
        super().__init__()
        self.feature_name = feature_name
        self.gamma = gamma
        self.eps = eps
        self._loss = Entmax15Loss(**kwargs)
        self._logits_callback = None

    @property
    def logits_callback(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]:
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
    def logits_callback(self, func: Callable | None) -> None:
        self._logits_callback = func

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
        :param model_embeddings: model output of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param feature_tensors: a dictionary of tensors from dataloader.
            This dictionary is expected to contain item popularity under ``feature_name``.
            Expected popularity shape: ``(batch_size, vocabulary_size)``.
        :param positive_labels: labels of positive events
            of shape ``(batch_size, sequence_length, num_positives)``.
        :param target_padding_mask: padding mask corresponding for `positive_labels`
            of shape ``(batch_size, sequence_length, num_positives)``.
        :return: a computed loss value.
        """
        if positive_labels.size(-1) != 1:
            msg = "The case of multi-positive labels is not supported in the PR-Entmax loss"
            raise NotImplementedError(msg)
        logits: torch.Tensor = self.logits_callback(model_embeddings)  # [batch_size, seq_len, vocab_size]
        popularity = feature_tensors[self.feature_name].unsqueeze(-2)
        logits = logits + self.gamma * torch.log(popularity + self.eps)
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


class PREntmaxSampled(SampledLossBase):
    """
    Sampled PR-Entmax loss with negative sampling.
    Calculates loss between one positive item and K negatively sampled items.

    The loss supports the calculation of logits for the case of multi-positive labels
    (there are several labels for each position in the sequence).
    """

    def __init__(
        self,
        feature_name: str,
        gamma: float,
        eps: float = 1e-6,
        negative_labels_ignore_index: int = -100,
        **kwargs,
    ):
        """
        To calculate the loss, ``Entmax15Loss`` is used.
        You can pass all parameters for initializing the object via kwargs.

        :param feature_name: Name of the feature containing item popularity.
        :param gamma: Weight of the popularity correction.
        :param eps: Small value added before logarithm to avoid log of zero.
        :param negative_labels_ignore_index: a padding value for negative labels.
            This may be the case when negative labels
            are formed at the preprocessing level, rather than the negative sampler.
            The index is ignored and does not contribute to the loss.
            Default: ``-100``.
        """
        super().__init__()
        self.feature_name = feature_name
        self.gamma = gamma
        self.eps = eps
        self.negative_labels_ignore_index = negative_labels_ignore_index
        self._loss = Entmax15Loss(**kwargs)
        self._logits_callback = None

    @property
    def logits_callback(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]:
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
    def logits_callback(self, func: Callable | None) -> None:
        self._logits_callback = func

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

        :param model_embeddings: a model output of shape ``(batch_size, sequence_length, embedding_dim)``.
        :param feature_tensors: a dictionary of tensors from dataloader.
            This dictionary is expected to contain item popularity under ``feature_name``.
            Expected popularity shape: ``(batch_size, vocabulary_size)``.
        :param positive_labels: labels of positive events
            of shape ``(batch_size, sequence_length, num_positives)``.
        :param negative_labels: labels of sampled negative events.

            Expected shape:
                - ``(batch_size, sequence_length, num_negatives)``
                - ``(batch_size, num_negatives)``
                - ``(num_negatives)`` - a case where the same negative events are used for the entire batch.
        :param target_padding_mask: a padding mask corresponding to ``positive_labels``
            of shape ``(batch_size, sequence_length, num_positives)``

        :return: a computed loss value.
        """
        target_batch_indices = target_padding_mask.nonzero(as_tuple=True)[0]
        target_labels = positive_labels[target_padding_mask]
        popularity = feature_tensors[self.feature_name]

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

        positive_popularity = popularity[target_batch_indices, target_labels].unsqueeze(-1)
        safe_negative_labels = negative_labels.masked_fill(
            negative_labels == self.negative_labels_ignore_index,
            0,
        )
        negative_popularity = popularity[
            target_batch_indices.unsqueeze(-1),
            safe_negative_labels,
        ]

        positive_logits = positive_logits + self.gamma * torch.log(positive_popularity + self.eps)
        negative_logits = negative_logits + self.gamma * torch.log(negative_popularity + self.eps)

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
