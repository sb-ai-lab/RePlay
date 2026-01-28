import contextlib
import warnings
from collections.abc import Sequence
from typing import Literal, Optional, Union

import torch

from replay.data.nn.schema import TensorFeatureInfo, TensorMap, TensorSchema


class SequenceEmbedding(torch.nn.Module):
    """
    The embedding generation class for all types of features given into the sequential models.

    The embedding size for each feature will be taken from ``TensorSchema`` (from field named ``embedding_dim``).
    For numerical features, it is expected that the last dimension of the tensor will be equal
    to ``tensor_dim`` field in ``TensorSchema``.
    """

    def __init__(
        self,
        schema: TensorSchema,
        excluded_features: Optional[list[str]] = None,
        categorical_list_feature_aggregation_method: Literal["sum", "mean", "max"] = "sum",
    ):
        """
        :param schema: TensorSchema containing meta information about all the features
            for which you need to generate an embedding.
        :param excluded_features: A list containing the names of features
            for which you do not need to generate an embedding.
            Fragments from this list are expected to be contained in ``schema``.
            Default: ``None``.
        :param categorical_list_feature_aggregation_method: Mode to aggregate tokens
            in token item representation (categorical list only).
            Default: ``"sum"``.
        """
        super().__init__()
        self.excluded_features = excluded_features or []
        feature_embedders = {}

        for feature_name, tensor_info in schema.items():
            if feature_name in self.excluded_features:
                continue
            if not tensor_info.is_seq:
                msg = f"Non-sequential features is not yet supported. Got {feature_name}"
                raise NotImplementedError(msg)
            if tensor_info.is_cat:
                feature_embedders[feature_name] = CategoricalEmbedding(
                    tensor_info,
                    categorical_list_feature_aggregation_method,
                )
            else:
                feature_embedders[feature_name] = NumericalEmbedding(tensor_info)

        self.feature_names = list(feature_embedders.keys())
        if not feature_embedders:
            msg = "Expected to have at least one feature name to generate embedding."
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
        """
        :param feature_tensor: a dictionary of tensors to generate embedding.
            It is expected that the keys from this dictionary match the names of the features in the given ``schema``.
        :param feature_names: A custom list of features for which embeddings need to be generated.
            It is expected that the values from this list match the names of the features in the given ``schema``.\n
            Default: ``None``. This means that the names of the features from the ``schema`` will be used.

        :returns: a dictionary with tensors that contains embeddings.
        """
        return {
            feature_name: self.feature_embedders[feature_name](feature_tensor[feature_name])
            for feature_name in (feature_names or self.feature_names)
        }

    @property
    def embeddings_dim(self) -> dict[str, int]:
        """
        Returns the embedding dimensions for each of the features in the `schema`.
        """
        return {name: emb.embedding_dim for name, emb in self.feature_embedders.items()}

    def get_item_weights(self, indices: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Getting the embedding weights for a feature that matches the item id feature
        with the name specified in the ``schema``.
        It is expected that embeddings for this feature will definitely exist.
        **Note**: the row corresponding to the padding will be excluded from the returned weights.
        This logic will work if given ``indices`` is ``None``.

        :param indices: Items indices.
        :returns: Embeddings for specific items.
        """
        if indices is None:
            return self.feature_embedders[self._item_feature_name].weight
        return self.feature_embedders[self._item_feature_name](indices)


class CategoricalEmbedding(torch.nn.Module):
    """
    The embedding generation class for categorical features.
    It supports working with single features for each event in sequence, as well as several (categorical list).

    When using this class, keep in mind that the embedding size will be 1 more than the cardinality.
    This is necessary to take into account the padding value.
    """

    def __init__(
        self,
        feature_info: TensorFeatureInfo,
        categorical_list_feature_aggregation_method: Literal["sum", "mean", "max"] = "sum",
    ) -> None:
        """
        :param feature_info: Meta information about the feature.
        :param categorical_list_feature_aggregation_method: Mode to aggregate tokens
            in token item representation (categorical list only). One of {`sum`, `mean`, `max`}
            Default: ``"sum"``.
        """
        super().__init__()
        assert feature_info.cardinality
        assert feature_info.embedding_dim

        self._expect_padding_value_setted = True
        if feature_info.cardinality != feature_info.padding_value:
            self._expect_padding_value_setted = False
            msg = (
                f"The padding value={feature_info.padding_value} is set for the feature={feature_info.name}. "
                f"The expected padding value for this feature should be {feature_info.cardinality}. "
                "Keep this in mind when getting the weights via the `weight` property, "
                "because the weights are returned there without padding row. "
                "Therefore, during the IDs scores generating, "
                "all the IDs that greater than the padding value should be increased by 1."
            )
            warnings.warn(msg, stacklevel=2)

        if feature_info.is_list:
            self.emb = torch.nn.EmbeddingBag(
                feature_info.cardinality + 1,
                feature_info.embedding_dim,
                padding_idx=feature_info.padding_value,
                mode=categorical_list_feature_aggregation_method,
            )
            self._get_embeddings = self._get_cat_list_embeddings
        else:
            self.emb = torch.nn.Embedding(
                feature_info.cardinality + 1,
                feature_info.embedding_dim,
                padding_idx=feature_info.padding_value,
            )
            self._get_embeddings = self._get_cat_embeddings

    @property
    def weight(self) -> torch.Tensor:
        """
        Returns the weights of the embedding layer,
        excluding the row that corresponds to the padding.
        """
        if not self._expect_padding_value_setted:
            msg = (
                "The weights are returned there do not contain padding row. "
                "Therefore, during the IDs scores generating, "
                "all the IDs that greater than the padding value should be increased by 1."
            )
            warnings.warn(msg, stacklevel=2)

        mask_without_padding = torch.ones(
            size=(self.emb.weight.size(0),),
            dtype=torch.bool,
            device=self.emb.weight.device,
        )
        mask_without_padding[self.emb.padding_idx].zero_()
        return self.emb.weight[mask_without_padding]

    def forward(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        :param indices: Items indices.

        :returns: Embeddings for specific items.
        """
        return self._get_embeddings(indices)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension after applying the layer"""
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
        assert indices.dim() >= 2
        if indices.dim() == 2:
            embeddings: torch.Tensor = self.emb(indices)
        else:
            source_size = indices.size()
            indices = indices.view(-1, source_size[-1])
            embeddings = self.emb(indices)
            embeddings = embeddings.view(*source_size[:-1], -1)
        return embeddings


class NumericalEmbedding(torch.nn.Module):
    """
    The embedding generation class for numerical features.
    It supports working with single features for each event in sequence, as well as several (numerical list).

    **Note**: if the ``embedding_dim`` field in ``TensorFeatureInfo`` for an incoming feature matches its last dimension
    (``tensor_dim`` field in ``TensorFeatureInfo``), then transformation will not be applied.
    """

    def __init__(self, feature_info: TensorFeatureInfo) -> None:
        """
        :param feature_info: Meta information about the feature.
        """
        super().__init__()
        assert feature_info.tensor_dim
        assert feature_info.embedding_dim
        self._tensor_dim = feature_info.tensor_dim
        self._embedding_dim = feature_info.embedding_dim
        self.linear = torch.nn.Linear(feature_info.tensor_dim, self.embedding_dim)

        if feature_info.is_list:
            if self.embedding_dim == feature_info.tensor_dim:
                torch.nn.init.eye_(self.linear.weight.data)
                torch.nn.init.zeros_(self.linear.bias.data)

                self.linear.weight.requires_grad = False
                self.linear.bias.requires_grad = False
        else:
            assert feature_info.tensor_dim == 1
            self.linear = torch.nn.Linear(feature_info.tensor_dim, self.embedding_dim)

    @property
    def weight(self) -> torch.Tensor:
        """
        Returns the weight of the applied layer.
        If ``embedding_dim`` matches ``tensor_dim``, then the identity matrix will be returned.
        """
        return self.linear.weight

    def forward(self, values: torch.FloatTensor) -> torch.Tensor:
        """
        Numerical embedding forward pass.\n
        **Note**: if the ``embedding_dim`` for an incoming feature matches its last dimension (``tensor_dim``),
        then transformation will not be applied.

        :param values: feature values.
        :returns: Embeddings for specific items.
        """
        if values.dim() <= 2 and self._tensor_dim == 1:
            values = values.unsqueeze(-1).contiguous()

        assert values.size(-1) == self._tensor_dim
        if self._tensor_dim != self.embedding_dim:
            return self.linear(values)
        return values

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension after applying the layer"""
        return self._embedding_dim
