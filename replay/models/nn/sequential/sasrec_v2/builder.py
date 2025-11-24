from typing import Optional

import torch

from replay.data.nn import TensorSchema
from replay.models.nn.loss import LossProto
from replay.models.nn.sequential.common.agg import SequentialEmbeddingAggregatorProto
from replay.models.nn.sequential.common.mask import AttentionMaskBuilderProto
from replay.models.nn.sequential.common.normalization import NormalizerProto

from .model import (
    EmbedderProto,
    EncoderProto,
    SasRec,
)


class SasRecBuilder:
    def __init__(self) -> None:
        self._embedder = None
        self._attn_mask_builder = None
        self._embedding_aggregator = None
        self._encoder = None
        self._output_normalization = None
        self._loss = None

    def embedder(self, embedder: EmbedderProto) -> "SasRecBuilder":
        self._embedder = embedder
        return self

    def attn_mask_builder(self, attn_mask_builder: AttentionMaskBuilderProto) -> "SasRecBuilder":
        self._attn_mask_builder = attn_mask_builder
        return self

    def embedding_aggregator(self, embedding_aggregator: SequentialEmbeddingAggregatorProto) -> "SasRecBuilder":
        self._embedding_aggregator = embedding_aggregator
        return self

    def encoder(self, encoder: EncoderProto) -> "SasRecBuilder":
        self._encoder = encoder
        return self

    def output_normalization(self, output_normalization: NormalizerProto) -> "SasRecBuilder":
        self._output_normalization = output_normalization
        return self

    def loss(self, loss: LossProto) -> "SasRecBuilder":
        self._loss = loss
        return self

    def default(
        self,
        tensor_schema: TensorSchema,
        hidden_size: int = 192,
        head_count: int = 4,
        block_count: int = 2,
        seq_len: int = 50,
        dropout: float = 0.3,
        excluded_features: Optional[list[str]] = None,
        categorical_list_feature_aggregation_method: str = "sum",
    ) -> "SasRecBuilder":
        from replay.models.nn.loss import CE
        from replay.models.nn.sequential.common.agg import SumAggregator
        from replay.models.nn.sequential.common.embedding import SequentialEmbedder
        from replay.models.nn.sequential.common.mask import DefaultAttentionMaskBuilder

        from .agg import SasRecEmbeddingAggregator
        from .layers import SasRecBlock

        excluded_features = [
            tensor_schema.query_id_feature_name,
            tensor_schema.timestamp_feature_name,
            *(excluded_features or []),
        ]

        self.embedder(
            SequentialEmbedder(
                tensor_schema,
                hidden_size,
                categorical_list_feature_aggregation_method=categorical_list_feature_aggregation_method,
                excluded_features=excluded_features,
            )
        )
        self.attn_mask_builder(DefaultAttentionMaskBuilder(tensor_schema, head_count))

        self.embedding_aggregator(
            SasRecEmbeddingAggregator(
                SumAggregator(hidden_size),
                hidden_size,
                seq_len,
                dropout,
            )
        )
        self.encoder(SasRecBlock(hidden_size, head_count, block_count, dropout, "relu"))
        self.output_normalization(torch.nn.LayerNorm(hidden_size))
        self.loss(CE(padding_value=tensor_schema.item_id_features.item().padding_value))
        return self

    def _check_required_params(self) -> None:
        params = [
            self._embedder,
            self._attn_mask_builder,
            self._embedding_aggregator,
            self._encoder,
            self._output_normalization,
            self._loss,
        ]
        param_names = [
            "embedder",
            "attn_mask_builder",
            "embedding_aggregator",
            "encoder",
            "output_normalization",
            "loss",
        ]
        for name, param in zip(param_names, params):
            if param is None:
                msg = f"You can not build `SasRec` because the parameter `{name}` is not specified."
                raise ValueError(msg)

    def build(self) -> SasRec:
        self._check_required_params()
        return SasRec(
            embedder=self._embedder,
            attn_mask_builder=self._attn_mask_builder,
            embedding_aggregator=self._embedding_aggregator,
            encoder=self._encoder,
            output_normalization=self._output_normalization,
            loss=self._loss,
        )
