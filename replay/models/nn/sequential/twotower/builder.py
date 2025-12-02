from collections.abc import Sequence
from typing import Optional

import torch

from replay.data.nn import TensorSchema
from replay.models.nn.loss import LossProto
from replay.models.nn.sequential.common.mask import AttentionMaskBuilderProto

from .model import (
    ContextMergerProto,
    EmbedderProto,
    EmbeddingAggregatorProto,
    ItemEncoderProto,
    NormalizerProto,
    QueryEncoderProto,
    TwoTower,
)


class TwoTowerBuilder:
    """
    The builder class for the TwoTower model.
    It allows to construct a model in parts,
    and also provides the opportunity to build a model from standard blocks inside the library.
    """
    def __init__(self) -> None:
        self._schema = None
        self._embedder = None
        self._attn_mask_builder = None
        self._query_tower_feature_names = None
        self._item_tower_feature_names = None
        self._query_embedding_aggregator = None
        self._item_embedding_aggregator = None
        self._query_encoder = None
        self._query_tower_output_normalization = None
        self._item_encoder = None
        self._feature_mapping_path = None
        self._item_reference_path = None
        self._loss = None
        self._context_merger = None

    def schema(self, tensor_schema: TensorSchema) -> "TwoTowerBuilder":
        self._schema = tensor_schema
        return self

    def embedder(self, embedder: EmbedderProto) -> "TwoTowerBuilder":
        self._embedder = embedder
        return self

    def attn_mask_builder(self, attn_mask_builder: AttentionMaskBuilderProto) -> "TwoTowerBuilder":
        self._attn_mask_builder = attn_mask_builder
        return self

    def query_tower_feature_names(self, query_tower_feature_names: Sequence[str]) -> "TwoTowerBuilder":
        self._query_tower_feature_names = query_tower_feature_names
        return self

    def item_tower_feature_names(self, item_tower_feature_names: Sequence[str]) -> "TwoTowerBuilder":
        self._item_tower_feature_names = item_tower_feature_names
        return self

    def query_embedding_aggregator(self, query_embedding_aggregator: EmbeddingAggregatorProto) -> "TwoTowerBuilder":
        self._query_embedding_aggregator = query_embedding_aggregator
        return self

    def item_embedding_aggregator(self, item_embedding_aggregator: EmbeddingAggregatorProto) -> "TwoTowerBuilder":
        self._item_embedding_aggregator = item_embedding_aggregator
        return self

    def query_encoder(self, query_encoder: QueryEncoderProto) -> "TwoTowerBuilder":
        self._query_encoder = query_encoder
        return self

    def query_tower_output_normalization(self, query_tower_output_normalization: NormalizerProto) -> "TwoTowerBuilder":
        self._query_tower_output_normalization = query_tower_output_normalization
        return self

    def item_encoder(self, item_encoder: ItemEncoderProto) -> "TwoTowerBuilder":
        self._item_encoder = item_encoder
        return self

    def feature_mapping_path(self, feature_mapping_path: str) -> "TwoTowerBuilder":
        self._feature_mapping_path = feature_mapping_path
        return self

    def item_reference_path(self, item_reference_path: str) -> "TwoTowerBuilder":
        self._item_reference_path = item_reference_path
        return self

    def loss(self, loss: LossProto) -> "TwoTowerBuilder":
        self._loss = loss
        return self

    def context_merger(self, context_merger: Optional[ContextMergerProto]) -> "TwoTowerBuilder":
        self._context_merger = context_merger
        return self

    def default(
        self,
        tensor_schema: TensorSchema,
        feature_mapping_path: str,
        item_reference_path: str,
        hidden_size: int = 192,
        head_count: int = 4,
        block_count: int = 2,
        seq_len: int = 50,
        embedding_dropout_rate: float = 0.3,
        excluded_features: Optional[list[str]] = None,
        categorical_list_feature_aggregation_method: str = "sum",
    ) -> "TwoTowerBuilder":
        from replay.models.nn.loss import CESampled
        from replay.models.nn.sequential.common.agg import SumAggregator
        from replay.models.nn.sequential.common.diff_transformer import DiffTransformerLayer
        from replay.models.nn.sequential.common.embedding import SequentialEmbedder
        from replay.models.nn.sequential.common.ffn import SwiGLUEncoder
        from replay.models.nn.sequential.common.mask import DefaultAttentionMaskBuilder

        from .agg import QueryTowerEmbeddingAggregator

        excluded_features = [
            tensor_schema.query_id_feature_name,
            tensor_schema.timestamp_feature_name,
            *(excluded_features or []),
        ]
        feature_names = set(tensor_schema.names) - set(excluded_features)

        self.schema(tensor_schema)
        self.embedder(
            SequentialEmbedder(
                tensor_schema,
                hidden_size,
                categorical_list_feature_aggregation_method=categorical_list_feature_aggregation_method,
                excluded_features=excluded_features,
            )
        )
        self.attn_mask_builder(DefaultAttentionMaskBuilder(tensor_schema, head_count))
        self.query_tower_feature_names(feature_names)
        self.item_tower_feature_names(feature_names)

        common_aggregator = SumAggregator(hidden_size)
        self.query_embedding_aggregator(
            QueryTowerEmbeddingAggregator(
                common_aggregator,
                hidden_size,
                seq_len,
                embedding_dropout_rate,
            )
        )
        self.item_embedding_aggregator(common_aggregator)
        self.query_encoder(DiffTransformerLayer(hidden_size, head_count, block_count))
        self.query_tower_output_normalization(torch.nn.RMSNorm(hidden_size))
        self.item_encoder(SwiGLUEncoder(hidden_size))
        self.feature_mapping_path(feature_mapping_path)
        self.item_reference_path(item_reference_path)
        self.loss(CESampled(tensor_schema.item_id_features.item().padding_value))
        return self

    def _check_required_params(self) -> None:
        params = [
            self._schema,
            self._embedder,
            self._attn_mask_builder,
            self._query_tower_feature_names,
            self._item_tower_feature_names,
            self._query_embedding_aggregator,
            self._item_embedding_aggregator,
            self._query_encoder,
            self._query_tower_output_normalization,
            self._item_encoder,
            self._feature_mapping_path,
            self._item_reference_path,
            self._loss,
        ]
        param_names = [
            "schema",
            "embedder",
            "attn_mask_builder",
            "query_tower_feature_names",
            "item_tower_feature_names",
            "query_embedding_aggregator",
            "item_embedding_aggregator",
            "query_encoder",
            "query_tower_output_normalization",
            "item_encoder",
            "feature_mapping_path",
            "item_reference_path",
            "loss",
        ]
        for name, param in zip(param_names, params):
            if param is None:
                msg = f"You can not build `TwoTower` because the parameter `{name}` is not specified."
                raise ValueError(msg)

    def build(self) -> TwoTower:
        self._check_required_params()
        return TwoTower(
            schema=self._schema,
            embedder=self._embedder,
            attn_mask_builder=self._attn_mask_builder,
            query_tower_feature_names=self._query_tower_feature_names,
            item_tower_feature_names=self._item_tower_feature_names,
            query_embedding_aggregator=self._query_embedding_aggregator,
            item_embedding_aggregator=self._item_embedding_aggregator,
            query_encoder=self._query_encoder,
            query_tower_output_normalization=self._query_tower_output_normalization,
            item_encoder=self._item_encoder,
            feature_mapping_path=self._feature_mapping_path,
            item_reference_path=self._item_reference_path,
            loss=self._loss,
            context_merger=self._context_merger,
        )
