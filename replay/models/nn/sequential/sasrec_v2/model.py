from collections.abc import Sequence
from typing import Optional, Protocol, Union

import torch

from replay.data.nn import TensorMap
from replay.models.nn.loss import LossProto
from replay.models.nn.output import InferenceOutput, TrainOutput
from replay.models.nn.sequential.common.agg import SequentialEmbeddingAggregatorProto
from replay.models.nn.sequential.common.head import EmbeddingTyingHead
from replay.models.nn.sequential.common.mask import AttentionMaskBuilderProto
from replay.models.nn.sequential.common.normalization import NormalizerProto
from replay.models.nn.utils import warning_is_not_none


class EmbedderProto(Protocol):
    def get_item_weights(
        self,
        indices: Optional[torch.LongTensor],
    ) -> torch.Tensor: ...

    def forward(
        self,
        feature_tensors: TensorMap,
        feature_names: Optional[Sequence[str]] = None,
    ) -> TensorMap: ...

    def reset_parameters(self) -> None: ...


class EncoderProto(Protocol):
    def forward(
        self,
        feature_tensors: TensorMap,
        input_embeddings: torch.Tensor,
        padding_mask: torch.BoolTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...


class SasRecBase(torch.nn.Module):
    def __init__(
        self,
        embedder: EmbedderProto,
        attn_mask_builder: AttentionMaskBuilderProto,
        embedding_aggregator: SequentialEmbeddingAggregatorProto,
        encoder: EncoderProto,
        output_normalization: NormalizerProto,
    ):
        super().__init__()
        self.embedder = embedder
        self.attn_mask_builder = attn_mask_builder
        self.embedding_aggregator = embedding_aggregator
        self.encoder = encoder
        self.output_normalization = output_normalization

    def reset_parameters(self) -> None:
        self.embedder.reset_parameters()
        self.embedding_aggregator.reset_parameters()
        self.encoder.reset_parameters()
        self.output_normalization.reset_parameters()

    def forward(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        embeddings = self.embedder(feature_tensors)
        agg_emb: torch.Tensor = self.embedding_aggregator(embeddings)
        assert agg_emb.dim() == 3

        attn_mask = self.attn_mask_builder(feature_tensors, padding_mask)

        hidden_state: torch.Tensor = self.encoder(
            feature_tensors=feature_tensors,
            input_embeddings=agg_emb,
            padding_mask=padding_mask,
            attention_mask=attn_mask,
        )
        assert agg_emb.size() == hidden_state.size()

        hidden_state = self.output_normalization(hidden_state)
        return hidden_state


class SasRec(torch.nn.Module):
    def __init__(
        self,
        embedder: EmbedderProto,
        attn_mask_builder: AttentionMaskBuilderProto,
        embedding_aggregator: SequentialEmbeddingAggregatorProto,
        encoder: EncoderProto,
        output_normalization: NormalizerProto,
        loss: LossProto,
    ):
        super().__init__()
        self.body = SasRecBase(
            embedder=embedder,
            attn_mask_builder=attn_mask_builder,
            embedding_aggregator=embedding_aggregator,
            encoder=encoder,
            output_normalization=output_normalization,
        )
        self.head = EmbeddingTyingHead()
        self.loss = loss
        self.loss.logits_callback = self.get_logits

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.body.reset_parameters()

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

        return TrainOutput(
            loss=loss,
            hidden_states=(hidden_states,),
        )

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

        return InferenceOutput(
            logits=logits,
            hidden_states=(hidden_states,),
        )

    def forward(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
        positive_labels: Optional[torch.LongTensor] = None,
        negative_labels: Optional[torch.LongTensor] = None,
        target_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Union[TrainOutput, InferenceOutput]:
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
            feature_tensors=feature_tensors, padding_mask=padding_mask, candidates_to_score=candidates_to_score
        )
