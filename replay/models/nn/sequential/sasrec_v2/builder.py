from typing import Literal, Optional

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
    """
    The builder class for the SasRec model.
    It allows you to construct a model in parts,
    and also provides the opportunity to build a model from standard blocks inside the library.
    """

    def __init__(self) -> None:
        self._embedder = None
        self._attn_mask_builder = None
        self._embedding_aggregator = None
        self._encoder = None
        self._output_normalization = None
        self._loss = None

    def embedder(self, embedder: EmbedderProto) -> "SasRecBuilder":
        """
        :param embedder: An object of a class that performs the logic of
            generating embeddings from an input set of tensors.
        """
        self._embedder = embedder
        return self

    def attn_mask_builder(self, attn_mask_builder: AttentionMaskBuilderProto) -> "SasRecBuilder":
        """
        :param attn_mask_builder: An object of a class that performs the logic of
            generating an attention mask based on the features and padding mask given to the model.
        """
        self._attn_mask_builder = attn_mask_builder
        return self

    def embedding_aggregator(self, embedding_aggregator: SequentialEmbeddingAggregatorProto) -> "SasRecBuilder":
        """
        :param embedding_aggregator: An object of a class that performs the logic of aggregating multiple embeddings.\n
            For example, it can be a ``sum``, a ``mean``, or a ``concatenation``.
        """
        self._embedding_aggregator = embedding_aggregator
        return self

    def encoder(self, encoder: EncoderProto) -> "SasRecBuilder":
        """
        :param encoder: An object of a class that performs the logic of generating
            a hidden embedding representation based on
            features, padding masks, attention mask, and aggregated embedding.
        """
        self._encoder = encoder
        return self

    def output_normalization(self, output_normalization: NormalizerProto) -> "SasRecBuilder":
        """
        :param output_normalization: An object of a class that performs the logic of
            normalization of the hidden state obtained from the encoder.\n
            For example, it can be a ``torch.nn.LayerNorm`` or ``torch.nn.RMSNorm``.
        """
        self._output_normalization = output_normalization
        return self

    def loss(self, loss: LossProto) -> "SasRecBuilder":
        """
        :param loss: An object of a class that performs loss calculation
            based on hidden states from the model, positive and negative labels.
        """
        self._loss = loss
        return self

    def default(
        self,
        tensor_schema: TensorSchema,
        embedding_dim: int = 192,
        head_count: int = 4,
        block_count: int = 2,
        max_sequence_length: int = 50,
        dropout: float = 0.3,
        excluded_features: Optional[list[str]] = None,
        categorical_list_feature_aggregation_method: Literal["sum", "mean", "max"] = "sum",
    ) -> "SasRecBuilder":
        """
        Initialization of standard model blocks based on generally accepted standard hyperparameters.

        :param tensor_schema: Tensor schema of features.
        :param embedding_dim: Dimensions of incoming embeddings.
            Default: ``192``.
        :param head_count: Number of Attention heads.
            Default: ``4``.
        :param block_count: Number of Transformer blocks.
            Default: ``2``.
        :param max_sequence_length: Max length of sequence.
            Default: ``50``.
        :param dropout: probability of an element to be zeroed.
            Default: ``0.3``.
        :param excluded_features: A list containing the names of features
            for which you do not need to generate an embedding.
            Fragments from this list are expected to be contained in `schema`.
            Default: ``None``.
        :param categorical_list_feature_aggregation_method: Mode to aggregate tokens
            in token item representation (categorical list only).
            Default: ``"sum"``.
        """
        from replay.models.nn.loss import CE
        from replay.models.nn.sequential.common.agg import SumAggregator
        from replay.models.nn.sequential.common.embedding import SequentialEmbedder
        from replay.models.nn.sequential.common.mask import DefaultAttentionMaskBuilder
        from replay.models.nn.sequential.common.transformer import TransformerLayer

        from .agg import SasRecEmbeddingAggregator

        excluded_features = [
            tensor_schema.query_id_feature_name,
            tensor_schema.timestamp_feature_name,
            *(excluded_features or []),
        ]
        excluded_features = list(set(excluded_features))

        self.embedder(
            SequentialEmbedder(
                tensor_schema,
                categorical_list_feature_aggregation_method=categorical_list_feature_aggregation_method,
                excluded_features=excluded_features,
            )
        )
        self.attn_mask_builder(DefaultAttentionMaskBuilder(tensor_schema, head_count))

        self.embedding_aggregator(
            SasRecEmbeddingAggregator(
                SumAggregator(embedding_dim),
                max_sequence_length,
                dropout,
            )
        )
        self.encoder(TransformerLayer(embedding_dim, head_count, block_count, dropout, "relu"))
        self.output_normalization(torch.nn.LayerNorm(embedding_dim))
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
        """
        :returns: Returns the initialized model.
        """
        self._check_required_params()
        return SasRec(
            embedder=self._embedder,
            embedding_aggregator=self._embedding_aggregator,
            attn_mask_builder=self._attn_mask_builder,
            encoder=self._encoder,
            output_normalization=self._output_normalization,
            loss=self._loss,
        )
