from typing import Any, Optional, Union

import torch

from replay.data.nn import TensorMap
from replay.nn.head import EmbeddingTyingHead, HeadProto
from replay.nn.loss import LossProto
from replay.nn.output import InferenceOutput, TrainOutput
from replay.nn.utils import warning_is_not_none


class BaseSequentialModel(torch.nn.Module):
    """Base class for sequential neural recommendation models.

    The class implements the common training and inference flows for models
    whose body returns a hidden state for every position in a sequence.
    """

    def __init__(
        self,
        body: Any,
        loss: LossProto,
        head: HeadProto | None = None,
    ) -> None:
        super().__init__()
        self.body = body
        self.head = EmbeddingTyingHead() if head is None else head
        self.loss = loss
        self.loss.logits_callback = self.get_logits

    @classmethod
    def from_params(cls, *args: Any) -> "BaseSequentialModel":
        raise NotImplementedError  # pragma: no cover

    def reset_parameters(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def get_logits(
        self,
        model_embeddings: torch.Tensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        item_embeddings: torch.Tensor = self.body.embedder.get_item_weights(candidates_to_score)
        logits: torch.Tensor = self.head(model_embeddings, item_embeddings)
        return logits

    def forward_train(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> TrainOutput:
        hidden_states: torch.Tensor = self.body(feature_tensors, padding_mask)
        assert hidden_states.dim() == 3

        loss: torch.Tensor = self.loss(
            model_embeddings=hidden_states,
            feature_tensors=feature_tensors,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            padding_mask=padding_mask,
            target_padding_mask=target_padding_mask,
        )

        return {
            "loss": loss,
            "hidden_states": (hidden_states,),
        }

    def forward_inference(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> InferenceOutput:
        hidden_states: torch.Tensor = self.body(feature_tensors, padding_mask)
        assert hidden_states.dim() == 3

        last_hidden_state = hidden_states[:, -1, :].contiguous()
        logits = self.get_logits(last_hidden_state, candidates_to_score)

        return {
            "logits": logits,
            "hidden_states": (hidden_states,),
        }

    def forward(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
        positive_labels: Optional[torch.LongTensor] = None,
        negative_labels: Optional[torch.LongTensor] = None,
        target_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Union[TrainOutput, InferenceOutput]:
        """
        :param feature_tensors: a dictionary of tensors to generate embeddings.
        :param padding_mask: A mask of shape ``(batch_size, sequence_length)``
            indicating which elements within ``key`` to ignore for the purpose of attention (i.e. treat as "padding").
            The ``False`` value indicates that the corresponding ``key`` value will be ignored.
        :param candidates_to_score: a tensor containing item IDs
            for which you need to get logits at the inference stage.\n
            **Note:** you must take into account the padding value when creating the tensor.\n
            The tensor participates in calculations only at the inference stage.
            You don't have to submit an argument at training stage,
            but if it is submitted, then no effect will be provided.\n
            Default: ``None``.
        :param positive_labels: a tensor containing positive labels for calculating the loss.\n
            You don't have to submit an argument at inference stage,
            but if it is submitted, then no effect will be provided.\n
            Default: ``None``.
        :param negative_labels: a tensor containing negative labels for calculating the loss.\n
            **Note:** Before running make sure that your loss supports calculations with negative labels.\n
            You don't have to submit an argument at inference stage,
            but if it is submitted, then no effect will be provided.\n
            Default: ``None``.
        :param target_padding_mask: A mask of shape ``(batch_size, sequence_length, num_positives)``
            indicating elements from ``positive_labels`` to ignore during loss calculation.
            The ``False`` value indicates that the corresponding value will be ignored.\n
            You don't have to submit an argument at inference stage,
            but if it is submitted, then no effect will be provided.\n
            Default: ``None``.
        :returns: During training, the model will return an object
            of the ``TrainOutput`` container class.
            At the inference stage, the ``InferenceOutput`` class will be returned.
        """
        if self.training:
            all(
                map(
                    warning_is_not_none("Variable `{}` is not None. This will have no effect at the training stage."),
                    [(candidates_to_score, "candidates_to_score")],
                )
            )
            return self.forward_train(
                feature_tensors=feature_tensors,
                padding_mask=padding_mask,
                positive_labels=positive_labels,
                negative_labels=negative_labels,
                target_padding_mask=target_padding_mask,
            )

        all(
            map(
                warning_is_not_none("Variable `{}` is not None. This will have no effect at the inference stage."),
                [
                    (positive_labels, "positive_labels"),
                    (negative_labels, "negative_labels"),
                    (target_padding_mask, "target_padding_mask"),
                ],
            )
        )
        return self.forward_inference(
            feature_tensors=feature_tensors,
            padding_mask=padding_mask,
            candidates_to_score=candidates_to_score,
        )
