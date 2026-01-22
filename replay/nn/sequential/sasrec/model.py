from collections.abc import Sequence
from typing import Literal, Optional, Protocol, Union

import torch

from replay.data.nn import TensorMap, TensorSchema
from replay.nn.agg import AggregatorProto
from replay.nn.head import EmbeddingTyingHead
from replay.nn.loss import LossProto
from replay.nn.mask import AttentionMaskProto
from replay.nn.normalization import NormalizerProto
from replay.nn.output import InferenceOutput, TrainOutput
from replay.nn.utils import warning_is_not_none


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


class SasRecBody(torch.nn.Module):
    """
    Implementation of the architecture of the SasRec model.\n
    It can include various self-written blocks for modifying the model,
    but the sequence of applying layers is fixed in accordance with the original architecture.

    Paper: https://arxiv.org/pdf/1808.09781.
    """

    def __init__(
        self,
        embedder: EmbedderProto,
        embedding_aggregator: AggregatorProto,
        attn_mask_builder: AttentionMaskProto,
        encoder: EncoderProto,
        output_normalization: NormalizerProto,
    ):
        """
        :param embedder: An object of a class that performs the logic of
            generating embeddings from an input set of tensors.
        :param embedding_aggregator: An object of a class that performs the logic of aggregating multiple embeddings.\n
            For example, it can be a ``sum``, a ``mean``, or a ``concatenation``.
        :param attn_mask_builder: An object of a class that performs the logic of
            generating an attention mask based on the features and padding mask given to the model.
        :param encoder: An object of a class that performs the logic of generating
            a hidden embedding representation based on
            features, padding masks, attention mask, and aggregated embedding.
        :param output_normalization: An object of a class that performs the logic of
            normalization of the hidden state obtained from the encoder.\n
            For example, it may be a ``torch.nn.LayerNorm`` or ``torch.nn.RMSNorm``.
        """
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
        """
        :param feature_tensors: a dictionary of tensors to generate embeddings.
        :param padding_mask: A mask of shape ``(batch_size, sequence_length)``
            indicating which elements within ``key`` to ignore for the purpose of attention (i.e. treat as "padding").
            ``False`` value indicates that the corresponding ``key`` value will be ignored.
        :returns: The final hidden state.\n
            Expected shape: ``(batch_size, sequence_length, embedding_dim)``
        """
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
    """
    A model using the SasRec architecture as a hidden state generator.
    The hidden states are multiplied by the item embeddings,
    resulting in logits for each of the items.

    Example:

    .. code-block:: python

        body = SasRecBody(
            embedder=SequenceEmbedding(
                schema=tensor_schema,
            ),
            embedding_aggregator=PositionAwareAggregator(
                embedding_aggregator=common_aggregator,
                max_sequence_length=100,
                dropout=0.2,
            ),
            attn_mask_builder=DefaultAttentionMask(
                reference_feature_name=tensor_schema.item_id_feature_name,
                num_heads=2,
            ),
            encoder=SasRecTransformerLayer(
                embedding_dim=256,
                num_heads=2,
                num_blocks=2,
                dropout=0.3,
                activation="relu",
            ),
            output_normalization=torch.nn.LayerNorm(256),
        )
        sasrec = SasRec(
            body=body,
            loss=CESampled(padding_idx=tensor_schema.item_id_features.item().padding_value)
        )

    """

    def __init__(
        self,
        body: SasRecBody,
        loss: LossProto,
    ):
        """
        :param body: An instance of SasRecBody.
        :param loss: An object of a class that performs loss calculation
            based on hidden states from the model, positive and optionally negative labels.
        """
        super().__init__()
        self.body = body
        self.head = EmbeddingTyingHead()
        self.loss = loss
        self.loss.logits_callback = self.get_logits

        self.reset_parameters()

    @classmethod
    def from_params(
        cls,
        schema: TensorSchema,
        embedding_dim: int = 192,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_sequence_length: int = 50,
        dropout: float = 0.3,
        excluded_features: Optional[list[str]] = None,
        categorical_list_feature_aggregation_method: Literal["sum", "mean", "max"] = "sum",
    ) -> "SasRec":
        from replay.nn.agg import SumAggregator
        from replay.nn.embedding import SequenceEmbedding
        from replay.nn.loss import CE
        from replay.nn.mask import DefaultAttentionMask

        from .agg import PositionAwareAggregator
        from .transformer import SasRecTransformerLayer

        excluded_features = [
            schema.query_id_feature_name,
            schema.timestamp_feature_name,
            *(excluded_features or []),
        ]
        excluded_features = list(set(excluded_features))

        body = SasRecBody(
            embedder=SequenceEmbedding(
                schema=schema,
                categorical_list_feature_aggregation_method=categorical_list_feature_aggregation_method,
                excluded_features=excluded_features,
            ),
            embedding_aggregator=PositionAwareAggregator(
                embedding_aggregator=SumAggregator(embedding_dim=embedding_dim),
                max_sequence_length=max_sequence_length,
                dropout=dropout,
            ),
            attn_mask_builder=DefaultAttentionMask(
                reference_feature_name=schema.item_id_feature_name,
                num_heads=num_heads,
            ),
            encoder=SasRecTransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_blocks=num_blocks,
                dropout=dropout,
                activation="relu",
            ),
            output_normalization=torch.nn.LayerNorm(embedding_dim),
        )
        return cls(
            body=body,
            loss=CE(padding_idx=schema.item_id_features.item().padding_value),
        )

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
            ``False`` value indicates that the corresponding ``key`` value will be ignored.
        :param candidates_to_score: a tensor containing item IDs
            for which you need to get logits at the inference stage.\n
            **Note:** you must take into account the padding value when creating the tensor.\n
            The tensor participates in calculations only on the inference stage.
            You don't have to submit an argument at training stage,
            but if it is submitted, then no effect will be provided.\n
            Default: ``None``.
        :param positive_labels: a tensor containing positive labels for calculating the loss.\n
            You don't have to submit an argument at inference stage,
            but if it is submitted, then no effect will be provided.\n
            Default: ``None``.
        :param negative_labels: a tensor containing negative labels for calculating the loss.\n
            **Note:** Before run make sure that your loss supports calculations with negative labels.\n
            You don't have to submit an argument at inference stage,
            but if it is submitted, then no effect will be provided.\n
            Default: ``None``.
        :param target_padding_mask: A mask of shape ``(batch_size, sequence_length, num_positives)``
            indicating elements from ``positive_labels`` to ignore during loss calculation.
            ``False`` value indicates that the corresponding value will be ignored.\n
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
