import pickle
from collections.abc import Generator, Sequence
from typing import Literal, Optional, Protocol, Union

import pandas as pd
import torch

from amazme.replay.data.nn import TensorMap, TensorSchema
from amazme.replay.models.nn.loss import LossProto
from amazme.replay.models.nn.output import InferenceOutput, TrainOutput
from amazme.replay.models.nn.sequential.common.head import EmbeddingTyingHead
from amazme.replay.models.nn.sequential.common.mask import AttentionMaskBuilderProto
from amazme.replay.models.nn.sequential.common.normalization import NormalizerProto
from amazme.replay.models.nn.utils import warning_is_not_none

FeatureDesc = dict[Union[str, int, float], int]


class EmbedderProto(Protocol):
    @property
    def feature_names(self) -> Sequence[str]: ...

    def forward(
        self,
        feature_tensors: TensorMap,
        feature_names: Sequence[str],
    ) -> TensorMap: ...

    def reset_parameters(self) -> None: ...


class EmbeddingAggregatorProto(Protocol):
    @property
    def feature_names(self) -> Sequence[str]: ...

    def forward(
        self,
        feature_tensors: TensorMap,
    ) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...


class QueryEncoderProto(Protocol):
    def forward(
        self,
        feature_tensors: TensorMap,
        input_embeddings: torch.Tensor,
        padding_mask: torch.BoolTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...


class ItemEncoderProto(Protocol):
    def forward(
        self,
        feature_tensors: TensorMap,
        input_embeddings: torch.Tensor,
    ) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...


class QueryTower(torch.nn.Module):
    def __init__(
        self,
        embedder: EmbedderProto,
        feature_names: Sequence[str],
        attn_mask_builder: AttentionMaskBuilderProto,
        embedding_aggregator: EmbeddingAggregatorProto,
        encoder: QueryEncoderProto,
        output_normalization: NormalizerProto,
    ):
        super().__init__()
        self.embedder = embedder
        self.attn_mask_builder = attn_mask_builder
        self.feature_names = feature_names
        self.embedding_aggregator = embedding_aggregator
        self.encoder = encoder
        self.output_normalization = output_normalization

    def reset_parameters(self) -> None:
        self.embedding_aggregator.reset_parameters()
        self.encoder.reset_parameters()
        self.output_normalization.reset_parameters()

    def forward(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        embeddings: TensorMap = self.embedder(feature_tensors, self.feature_names)
        agg_emb: torch.Tensor = self.embedding_aggregator(embeddings)
        assert agg_emb.dim() == 3

        attn_mask = self.attn_mask_builder(feature_tensors, padding_mask)
        print(attn_mask.shape)
        print(attn_mask)
        hidden_state: torch.Tensor = self.encoder(
            feature_tensors=feature_tensors,
            input_embeddings=agg_emb,
            padding_mask=padding_mask,
            attention_mask=attn_mask,
        )
        assert agg_emb.size() == hidden_state.size()

        hidden_state = self.output_normalization(hidden_state)
        return hidden_state


class ItemReference:
    def __init__(self, item_reference: dict[str, dict[Union[str, int, float], int]]):
        # {feature_name: {value0: idx0, value1: idx1, ...}, ...}
        self.item_reference = item_reference

    def keys(self) -> Generator[str, None, None]:
        return self.item_reference.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.item_reference

    def __getitem__(self, key: str) -> dict[Union[str, int, float], int]:
        return self.item_reference[key]

    @classmethod
    def load_feature_mapping(
        cls,
        schema: TensorSchema,
        feature_mapping_path: str,
    ) -> dict[str, dict[Union[str, int, float], int]]:
        if not feature_mapping_path:
            return {}
        feature_mapping = cls._load_file(feature_mapping_path)

        return cls.update_feature_mapping(schema, feature_mapping)

    @classmethod
    def load_item_reference(
        cls,
        schema: TensorSchema,
        item_reference_path: str,
        feature_mapping: dict[str, dict[Union[str, int, float], int]],
    ) -> "ItemReference":
        if not item_reference_path:
            return {schema.item_id_feature_name: list(range(schema.item_id_features.item().cardinality))}
        if not feature_mapping:
            msg = "Expected to have feature_mapping if passing item_reference_path for future mapping"
            raise ValueError(msg)
        item_reference = cls._load_file(item_reference_path, read_mode="pandas")
        return cls.get_item_reference(schema, item_reference, feature_mapping)

    @classmethod
    def update_feature_mapping(
        cls,
        schema: TensorSchema,
        feature_mapping: dict[str, FeatureDesc],
    ) -> dict[str, dict[Union[str, int, float], int]]:
        inverse_feature_names_mapping = cls._get_inverse_feature_names_mapping(schema)
        for feature_name in list(feature_mapping.keys()):
            new_feature_name = inverse_feature_names_mapping[feature_name]
            if new_feature_name == feature_name:
                continue
            feature_mapping[new_feature_name] = feature_mapping.pop(feature_name)
        return feature_mapping

    @classmethod
    def get_item_reference(
        cls,
        schema: TensorSchema,
        item_reference: pd.DataFrame,
        feature_mapping: dict[str, dict[Union[str, int, float], int]],
    ) -> "ItemReference":
        inverse_feature_names_mapping = cls._get_inverse_feature_names_mapping(schema)
        item_reference = item_reference.rename(columns=inverse_feature_names_mapping)
        item_reference: pd.DataFrame = (
            item_reference.loc[
                item_reference[schema.item_id_feature_name].isin(
                    list(feature_mapping[schema.item_id_feature_name].keys())
                )
            ]
            .sort_values(schema.item_id_feature_name, key=lambda x: x.map(feature_mapping[schema.item_id_feature_name]))
            .reset_index(drop=True)
        )
        item_reference = {
            feature_name: item_reference[feature_name].tolist() for feature_name in item_reference.columns
        }

        output_item_reference = {}
        for feature_name in item_reference:
            if feature_name not in feature_mapping:  # num features and other features
                output_item_reference[feature_name] = item_reference[feature_name].copy()
            elif isinstance(item_reference[feature_name][0], (int, float, str)):
                output_item_reference[feature_name] = [
                    feature_mapping[feature_name][x] for x in item_reference[feature_name]
                ]
            else:
                output_item_reference[feature_name] = [
                    [feature_mapping[feature_name].get(xx, schema.get(feature_name).padding_value) for xx in x]
                    for x in item_reference[feature_name]
                ]

        return cls(output_item_reference)

    @staticmethod
    def _get_inverse_feature_names_mapping(schema: TensorSchema) -> dict[str, str]:
        return {schema.get(feature_name).feature_source.column: feature_name for feature_name in schema}

    @staticmethod
    def _load_file(path: str, read_mode: Literal["pandas", "pickle"] = "pickle"):
        if read_mode == "pandas":
            if path.endswith(".parquet"):
                output = pd.read_parquet(path)
            elif path.endswith((".pkl", ".pickle")):
                output = pd.read_pickle(path)
            elif path.endswith(".csv"):
                output = pd.read_csv(path)
            else:
                msg = "Got filepath with unexpected file extension. Expected to get only parquet, pickle, csv files"
                raise ValueError(msg)
        elif read_mode == "pickle":
            with open(path, "rb") as f:
                output = pickle.load(f)
        else:
            msg = "Expected to read only pandas and pickle files"
            raise ValueError(msg)
        return output


class ItemTower(torch.nn.Module):
    def __init__(
        self,
        schema: TensorSchema,
        embedder: EmbedderProto,
        embedding_aggregator: EmbeddingAggregatorProto,
        encoder: ItemEncoderProto,
        feature_names: Sequence[str],
        feature_mapping_path: str,
        item_reference_path: str,
    ):
        super().__init__()
        self.embedder = embedder
        self.feature_names = feature_names
        self.embedding_aggregator = embedding_aggregator
        self.encoder = encoder

        feature_mapping = ItemReference.load_feature_mapping(schema, feature_mapping_path)
        self.item_reference = ItemReference.load_item_reference(schema, item_reference_path, feature_mapping)
        for feature_name, tensor_info in schema.items():
            if not tensor_info.is_seq:
                msg = "Non-sequential features is not yet supported"
                raise NotImplementedError(msg)
            if feature_name not in self.feature_names:
                continue

            dtype = torch.float32 if tensor_info.is_num else torch.int64
            buffer = torch.asarray(self.item_reference[feature_name], dtype=dtype)
            self.register_buffer(f"item_reference_{feature_name}", buffer)

        self.cache = None

    def reset_parameters(self) -> None:
        self.embedding_aggregator.reset_parameters()
        self.encoder.reset_parameters()

    def get_feature_buffer(self, feature_name: str) -> torch.Tensor:
        buffer_name = f"item_reference_{feature_name}"
        return self.get_buffer(buffer_name)

    def forward(
        self,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ):
        if self.training:
            self.cache = None

        if not self.training and self.cache is not None:
            if candidates_to_score is None:
                return self.cache
            return self.cache[candidates_to_score]

        if candidates_to_score is None:
            feature_tensors = {
                feature_name: self.get_feature_buffer(feature_name) for feature_name in self.feature_names
            }
        else:
            feature_tensors = {
                feature_name: self.get_feature_buffer(feature_name)[candidates_to_score]
                for feature_name in self.feature_names
            }

        embeddings: TensorMap = self.embedder(feature_tensors, self.feature_names)
        agg_emb: torch.Tensor = self.embedding_aggregator(embeddings)

        hidden_state: torch.Tensor = self.encoder(
            feature_tensors=feature_tensors,
            input_embeddings=agg_emb,
        )
        assert agg_emb.size() == hidden_state.size()

        if not self.training and self.cache is None and candidates_to_score is None:
            self.cache = hidden_state
        return hidden_state


class TwoTowerBase(torch.nn.Module):
    def __init__(
        self,
        schema: TensorSchema,
        embedder: EmbedderProto,
        attn_mask_builder: AttentionMaskBuilderProto,
        query_tower_feature_names: Sequence[str],
        item_tower_feature_names: Sequence[str],
        query_embedding_aggregator: EmbeddingAggregatorProto,
        item_embedding_aggregator: EmbeddingAggregatorProto,
        query_encoder: QueryEncoderProto,
        query_tower_output_normalization: NormalizerProto,
        item_encoder: ItemEncoderProto,
        feature_mapping_path: str,
        item_reference_path: str,
    ):
        super().__init__()
        self.embedder = embedder
        feature_names_union = set(query_tower_feature_names) | set(item_tower_feature_names)
        feature_names_not_in_emb = feature_names_union - set(self.embedder.feature_names)
        if len(feature_names_not_in_emb) != 0:
            msg = f"Feature names found that embedder does not support {list(feature_names_not_in_emb)}"
            raise ValueError(msg)

        self.query_tower = QueryTower(
            embedder,
            query_tower_feature_names,
            attn_mask_builder,
            query_embedding_aggregator,
            query_encoder,
            query_tower_output_normalization,
        )
        self.item_tower = ItemTower(
            schema,
            embedder,
            item_embedding_aggregator,
            item_encoder,
            item_tower_feature_names,
            feature_mapping_path,
            item_reference_path,
        )

    def reset_parameters(self) -> None:
        self.embedder.reset_parameters()
        self.query_tower.reset_parameters()
        self.item_tower.reset_parameters()


class ContextMergerProto(Protocol):
    def forward(
        self,
        model_hidden_state: torch.Tensor,
        feature_tensors: TensorMap,
    ) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...


class TwoTower(torch.nn.Module):
    def __init__(
        self,
        schema: TensorSchema,
        embedder: EmbedderProto,
        attn_mask_builder: AttentionMaskBuilderProto,
        query_tower_feature_names: Sequence[str],
        item_tower_feature_names: Sequence[str],
        query_embedding_aggregator: EmbeddingAggregatorProto,
        item_embedding_aggregator: EmbeddingAggregatorProto,
        query_encoder: QueryEncoderProto,
        query_tower_output_normalization: NormalizerProto,
        item_encoder: ItemEncoderProto,
        feature_mapping_path: str,
        item_reference_path: str,
        loss: LossProto,
        context_merger: Optional[ContextMergerProto] = None,
    ):
        super().__init__()
        self.body = TwoTowerBase(
            schema=schema,
            embedder=embedder,
            attn_mask_builder=attn_mask_builder,
            query_tower_feature_names=query_tower_feature_names,
            item_tower_feature_names=item_tower_feature_names,
            query_embedding_aggregator=query_embedding_aggregator,
            item_embedding_aggregator=item_embedding_aggregator,
            query_encoder=query_encoder,
            query_tower_output_normalization=query_tower_output_normalization,
            item_encoder=item_encoder,
            feature_mapping_path=feature_mapping_path,
            item_reference_path=item_reference_path,
        )
        self.head = EmbeddingTyingHead()
        self.loss = loss
        self.context_merger = context_merger
        self.loss.logits_callback = self.get_logits

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.body.reset_parameters()

    def get_logits(
        self,
        model_embeddings: torch.Tensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        item_embeddings: torch.Tensor = self.body.item_tower(candidates_to_score)
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
        hidden_states = ()
        query_hidden_states: torch.Tensor = self.body.query_tower(
            feature_tensors,
            padding_mask,
        )
        assert query_hidden_states.dim() == 3
        hidden_states += (query_hidden_states,)

        if self.context_merger is not None:
            query_hidden_states: torch.Tensor = self.context_merger(
                model_hidden_state=query_hidden_states,
                feature_tensors=feature_tensors,
            )
            assert query_hidden_states.dim() == 3
            hidden_states += (query_hidden_states,)

        loss: torch.Tensor = self.loss(
            model_embeddings=query_hidden_states,
            feature_tensors=feature_tensors,
            positive_labels=positive_labels,
            negative_labels=negative_labels,
            padding_mask=padding_mask,
            target_padding_mask=target_padding_mask,
        )

        return TrainOutput(
            loss=loss,
            hidden_states=hidden_states,
        )

    def forward_inference(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> InferenceOutput:
        hidden_states = ()
        query_hidden_states: torch.Tensor = self.body.query_tower(
            feature_tensors,
            padding_mask,
        )
        assert query_hidden_states.dim() == 3

        hidden_states += (query_hidden_states,)

        if self.context_merger is not None:
            query_hidden_states: torch.Tensor = self.context_merger(
                model_hidden_state=query_hidden_states,
                feature_tensors=feature_tensors,
            )
            assert query_hidden_states.dim() == 3
            hidden_states += (query_hidden_states,)

        last_hidden_state = query_hidden_states[:, -1, :].contiguous()
        logits = self.get_logits(last_hidden_state, candidates_to_score)

        return InferenceOutput(
            logits=logits,
            hidden_states=hidden_states,
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
