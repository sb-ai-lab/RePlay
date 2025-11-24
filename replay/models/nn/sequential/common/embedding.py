import contextlib
from collections.abc import Sequence
from typing import Literal, Optional, Union

import torch

from replay.data.nn.schema import TensorFeatureInfo, TensorMap, TensorSchema


class SequentialEmbedder(torch.nn.Module):
    def __init__(
        self,
        schema: TensorSchema,
        embed_size: int,
        categorical_list_feature_aggregation_method: Literal["sum", "mean", "max"] = "sum",
        excluded_features: Optional[list[str]] = None,
    ):
        super().__init__()
        self.excluded_features = excluded_features or []
        feature_embedders = {}

        for feature_name, tensor_info in schema.items():
            if feature_name in self.excluded_features:
                continue
            if not tensor_info.is_seq:
                msg = f"Non-sequential features is not yet supported. Got {feature_name}"
                raise NotImplementedError(msg)
            if tensor_info.is_cat:  # categorical feature
                feature_embedders[feature_name] = CategoricalEmbedding(
                    tensor_info,
                    categorical_list_feature_aggregation_method,
                )
            elif tensor_info.is_num:  # numerical feature
                feature_embedders[feature_name] = NumericalEmbedding(
                    tensor_info,
                    embed_size,
                )
            else:
                msg = (
                    "Preprocess only cat, num, cat_list, num_list features. "
                    f"Got {feature_name}: is_cat={tensor_info.is_cat}, is_num={tensor_info.is_num}"
                )
                raise ValueError(msg)

        self.feature_names = list(feature_embedders.keys())
        if not feature_embedders:
            msg = "Expected to have at least 1 feature name including item_id"
            raise ValueError(msg)
        self.feature_embedders: dict[str, Union[CategoricalEmbedding, NumericalEmbedding]] = torch.nn.ModuleDict(
            feature_embedders
        )
        self._item_feature_name = schema.item_id_feature_name

    def reset_parameters(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def forward(self, feature_tensor: TensorMap, feature_names: Optional[Sequence[str]] = None) -> TensorMap:
        return {
            feature_name: self.feature_embedders[feature_name](feature_tensor[feature_name])
            for feature_name in (feature_names or self.feature_names)
        }

    @property
    def embeddings_dim(self) -> dict[str, int]:
        return {name: emb.embedding_dim for name, emb in self.feature_embedders.items()}

    def get_weights(self, feature_name: str, indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
        if indices is not None:
            return self.feature_embedders[feature_name](indices)
        return self.feature_embedders[feature_name].weight

    def get_item_weights(self, indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
        return self.get_weights(self._item_feature_name, indices)


class CategoricalEmbedding(torch.nn.Module):
    """
    Categorical feature embedding.
    """

    def __init__(
        self,
        feature: TensorFeatureInfo,
        categorical_list_feature_aggregation_method: Literal["sum", "mean", "max"] = "sum",
    ) -> None:
        """
        :param feature: Categorical tensor feature.
        :param categorical_list_feature_aggregation_method: Mode to aggregate tokens
            in token item representation (categorical list only). One of {`sum`, `mean`, `max`}
            Default: ``"sum"``.
        """
        super().__init__()
        assert feature.cardinality
        assert feature.embedding_dim
        if feature.is_list:
            self.emb = torch.nn.EmbeddingBag(
                feature.cardinality + 1,
                feature.embedding_dim,
                padding_idx=feature.padding_value,
                mode=categorical_list_feature_aggregation_method,
            )
            self._get_embeddings = self._get_cat_list_embeddings
        else:
            self.emb = torch.nn.Embedding(
                feature.cardinality + 1,
                feature.embedding_dim,
                padding_idx=feature.padding_value,
            )
            self._get_embeddings = self._get_cat_embeddings

    @property
    def weight(self) -> torch.Tensor:
        return self.emb.weight

    def forward(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        :param indices: Items indices.

        :returns: Embeddings for specific items.
        """
        return self._get_embeddings(indices)

    @property
    def embedding_dim(self) -> int:
        return self.emb.embedding_dim

    def _get_cat_embeddings(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        :param indices: Items indices.

        :returns: Embeddings for specific items.
        """
        return self.emb(indices)

    def _get_cat_list_embeddings(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        :param indices: Items indices.

        :returns: Embeddings for specific items.
        """
        source_size = indices.size()
        if indices.dim() >= 3:
            indices = indices.view(-1, source_size[-1])
        embeddings: torch.Tensor = self.emb(indices)
        embeddings = embeddings.view(*source_size[:-1], -1)
        return embeddings


class NumericalEmbedding(torch.nn.Module):
    """
    Numerical feature embedding.
    """

    def __init__(self, feature: TensorFeatureInfo, embed_size: int) -> None:
        """
        :param feature: Numerical tensor feature.
        :param embed_size: Output embedding dim.
        """
        super().__init__()
        assert feature.tensor_dim
        if feature.is_list:
            self._get_embeddings = self._get_num_list_embeddings
            self._embedding_dim = feature.tensor_dim
        else:
            self._get_embeddings = self._get_num_embeddings
            self._embedding_dim = embed_size
        self.linear = torch.nn.Linear(feature.tensor_dim, embed_size)

    @property
    def weight(self) -> torch.Tensor:
        return self.linear.weight

    def forward(self, values: torch.FloatTensor) -> torch.Tensor:
        """
        :param values: feature values.

        :returns: Embeddings for specific items.
        """
        embeddings = self._get_embeddings(values)
        return embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _get_num_embeddings(self, values: torch.FloatTensor) -> torch.Tensor:
        """
        :param values: feature values.

        :returns: Embeddings for specific items.
        """
        if values.dim() == 2:
            values = values.unsqueeze(-1).contiguous()
        return self.linear(values)

    def _get_num_list_embeddings(self, values: torch.FloatTensor) -> torch.Tensor:
        """
        :param values: embedding values.

        :returns: Embeddings for specific items.
        """
        return values
