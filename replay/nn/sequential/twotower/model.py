from collections.abc import Generator, Sequence
from typing import Optional, Protocol, Union

import pandas as pd
import torch

from replay.data import FeatureSource
from replay.data.nn import TensorMap, TensorSchema
from replay.nn import (
    AggregatorProto,
    AttentionMaskProto,
    EmbeddingTyingHead,
    InferenceOutput,
    NormalizerProto,
    TrainOutput,
)
from replay.nn.loss import LossProto
from replay.nn.utils import warning_is_not_none


class EmbedderProto(Protocol):
    @property
    def feature_names(self) -> Sequence[str]: ...

    def forward(
        self,
        feature_tensors: TensorMap,
        feature_names: Optional[Sequence[str]] = None,
    ) -> TensorMap: ...

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
    """
    Query Tower of Two-Tower model.
    """

    def __init__(
        self,
        embedder: EmbedderProto,
        feature_names: Sequence[str],
        attn_mask_builder: AttentionMaskProto,
        embedding_aggregator: AggregatorProto,
        encoder: QueryEncoderProto,
        output_normalization: NormalizerProto,
    ):
        """
        :param embedder: An object of a class that performs the logic of
            generating embeddings from an input set of tensors.\n
            An embedder with the same arguments is used for both towers, but each tower has its own instance.
        :param feature_names: sequence of names used in query tower.
        :param attn_mask_builder: An object of a class that performs the logic of
            generating an attention mask based on the features and padding mask given to the model.
        :param embedding_aggregator: An object of a class that performs
            the logic of aggregating multiple embeddings of query tower.
        :param encoder: An object of a class that performs the logic of generating
            a query hidden embedding representation based on
            features, padding masks, attention mask, and aggregated embedding of ``query_tower_feature_names``.
            It's supposed to be a transformer.
        :param output_normalization: An object of a class that performs the logic of
            normalization of the hidden state obtained from the query encoder.\n
            For example, it can be a ``torch.nn.LayerNorm`` or ``torch.nn.RMSNorm``.
        """
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
        """
        :param feature_tensors: a dictionary of tensors to generate embeddings.
        :param padding_mask: A mask of shape ``(batch_size, sequence_length)``
            indicating which elements within ``key`` to ignore for the purpose of attention (i.e. treat as "padding").
            ``False`` value indicates that the corresponding ``key`` value will be ignored.
        :returns: The final hidden state.\n
            Expected shape: ``(batch_size, sequence_length, embedding_dim)``
        """
        embeddings: TensorMap = self.embedder(feature_tensors, self.feature_names)
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


class ItemReference:
    """
    Prepares a dict of item features values that will be used for training and inference of the Item Tower.
    """

    def __init__(self, schema: TensorSchema, item_reference_path: str):
        """
        :param schema: the same tensor schema used in TwoTower model.
        :param item_reference_path: path to parquet with dataframe of item features.\n
            **Note:**\n
            1. Dataframe columns must be already encoded via the same encoders used in `query_encoder` (user "tower").\n
            2. Item reference is constructed only on features with source of FeatureSource.ITEM_FEATURES
            in tensor schema so an identificator of items ("item_id") should be marked as FeatureSource.ITEM_FEATURES too.
        """
        inverse_feature_names_mapping = {
            schema.get(feature_name).feature_source.column: feature_name
            for feature_name in schema
            if feature_name in schema.item_id_features
            or schema.get(feature_name).feature_source.source == FeatureSource.ITEM_FEATURES
        }

        item_reference = pd.read_parquet(item_reference_path)
        item_reference = item_reference.rename(columns=inverse_feature_names_mapping)
        item_reference = item_reference.loc[:, inverse_feature_names_mapping.values()]
        item_reference: pd.DataFrame = item_reference.sort_values(schema.item_id_feature_name).reset_index(drop=True)
        item_reference = {
            feature_name: item_reference[feature_name].tolist() for feature_name in item_reference.columns
        }
        self.item_reference = item_reference

    def keys(self) -> Generator[str, None, None]:
        return self.item_reference.keys()

    def __contains__(self, key: str) -> bool:
        return key in self.item_reference

    def __getitem__(self, key: str) -> dict[Union[str, int, float], int]:
        return self.item_reference[key]


class ItemTower(torch.nn.Module):
    """
    Item Tower of Two-Tower model.

    **Note**: ItemTower loads feature tensors of all items into memory.
    """

    def __init__(
        self,
        schema: TensorSchema,
        embedder: EmbedderProto,
        embedding_aggregator: AggregatorProto,
        encoder: ItemEncoderProto,
        feature_names: Sequence[str],
        item_reference_path: str,
    ):
        """
        :param schema: tensor schema object with metainformation about features.
        :param embedder: An object of a class that performs the logic of
            generating embeddings from an input set of tensors.\n
            An embedder with the same arguments is used for both towers, but each tower has its own instance.
        :param embedding_aggregator: An object of a class that performs
            the logic of aggregating multiple embeddings of item tower.
        :param encoder: An object of a class that performs the logic of generating
            an item hidden embedding representation based on
            features and aggregated embeddings of ``item_tower_feature_names``.
            Item encoder uses item reference which is created based on ``item_reference_path``.
        :param feature_names: sequence of names used in item tower.
        :param item_reference_path: Path to dataframe with
            all items with features used in ``item_encoder`` (item "tower").
        """
        super().__init__()
        self.embedder = embedder
        self.feature_names = feature_names
        self.embedding_aggregator = embedding_aggregator
        self.encoder = encoder

        self.item_reference = ItemReference(schema, item_reference_path)
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
        """
        :param candidates_to_score: IDs of items using for obtaining item embeddings from item tower.
            If is setted to ``None``, all item embeddings from item tower will be returned.
            Default: ``None``.
        :return: item embeddings.\n
            Expected shape:\n
                - ``(candidates_to_score, embedding_dim)``,
                - ``(items_num, embedding_dim)`` if ``candidates_to_score`` is ``None``.
        """
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


class TwoTowerBody(torch.nn.Module):
    """
    Foundation for Two-Tower model which creates query "tower" and item "tower".\n

    For usage of two tower model, it should be added a forward pass with any scoring function
    for the hidden states of both towers, like a dot product.
    """

    def __init__(
        self,
        schema: TensorSchema,
        embedder: EmbedderProto,
        attn_mask_builder: AttentionMaskProto,
        query_tower_feature_names: Sequence[str],
        item_tower_feature_names: Sequence[str],
        query_embedding_aggregator: AggregatorProto,
        item_embedding_aggregator: AggregatorProto,
        query_encoder: QueryEncoderProto,
        query_tower_output_normalization: NormalizerProto,
        item_encoder: ItemEncoderProto,
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
    """
    Implementation generic Two-Tower architecture with two independent "towers" (encoders)
    which encode separate inputs. In recommender systems they are typically query tower and item tower.\n
    The output hidden states of each "tower" are fused via dot product in the model head.

    Paper: https://doi.org/10.1145/3366424.3386195
    """

    def __init__(
        self,
        schema: TensorSchema,
        embedder: EmbedderProto,
        attn_mask_builder: AttentionMaskProto,
        query_tower_feature_names: Sequence[str],
        item_tower_feature_names: Sequence[str],
        query_embedding_aggregator: AggregatorProto,
        item_embedding_aggregator: AggregatorProto,
        query_encoder: QueryEncoderProto,
        query_tower_output_normalization: NormalizerProto,
        item_encoder: ItemEncoderProto,
        item_reference_path: str,
        loss: LossProto,
        context_merger: Optional[ContextMergerProto] = None,
    ):
        """
        :param schema: tensor schema object with metainformation about features.
        :param embedder: An object of a class that performs the logic of
            generating embeddings from an input set of tensors.\n
            An embedder with the same arguments is used for both towers, but each tower has its own instance.
        :param attn_mask_builder: An object of a class that performs the logic of
            generating an attention mask based on the features and padding mask given to the model.
        :param query_tower_feature_names: sequence of names used in query tower.
        :param item_tower_feature_names: sequence of names used in item tower.
        :param query_embedding_aggregator: An object of a class that performs
            the logic of aggregating multiple embeddings of query tower.
        :param item_embedding_aggregator: An object of a class that performs
            the logic of aggregating multiple embeddings of item tower.
        :param query_encoder: An object of a class that performs the logic of generating
            a query hidden embedding representation based on
            features, padding masks, attention mask, and aggregated embedding of ``query_tower_feature_names``.
            It's supposed to be a transformer.
        :param query_tower_output_normalization: An object of a class that performs the logic of
            normalization of the hidden state obtained from the query encoder.\n
            For example, it can be a ``torch.nn.LayerNorm`` or ``torch.nn.RMSNorm``.
        :param item_encoder: An object of a class that performs the logic of generating
            an item hidden embedding representation based on
            features and aggregated embeddings of ``item_tower_feature_names``.
            Item encoder uses item reference which is created based on ``item_reference_path``.
        :param item_reference_path: Path to dataframe
            with all items with features used in ``item_encoder`` (item "tower").
            **Note:**\n
            1. Dataframe columns must be already encoded via the same encoders used in `query_encoder` (user "tower").\n
            2. Item reference is constructed only on features with source of FeatureSource.ITEM_FEATURES in tensor
            schema so an identificator of items ("item_id") should be marked as FeatureSource.ITEM_FEATURES too.
        :param loss: An object of a class that performs loss calculation
            based on hidden states from the model, positive and optionally negative labels.
        :param context_merger: An object of class that performs fusing query encoder hidden state
            with input feature tensors.
            Default: ``None``.
        """
        super().__init__()
        self.body = TwoTowerBody(
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
            item_reference_path=item_reference_path,
        )
        self.head = EmbeddingTyingHead()
        self.loss = loss
        self.context_merger = context_merger
        self.loss.logits_callback = self.get_logits

        self.reset_parameters()

    @classmethod
    def from_params(
        cls,
        schema: TensorSchema,
        item_reference_path: str,
        embedding_dim: int = 192,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_sequence_length: int = 50,
        dropout: float = 0.3,
        excluded_features: Optional[list[str]] = None,
        categorical_list_feature_aggregation_method: str = "sum",
    ) -> "TwoTower":
        """
        Class method for fast creating an instance of TwoTower with typical types
        of blocks and user provided parameters.\n
        The item "tower" is a SwiGLU encoder (MLP with SwiGLU activation),\n
        the user "tower" is a SasRec transformer layers, and loss is Cross-Entropy loss.\n
        Embeddings of every feature in both "towers" are aggregated via sum.
        The same features are be used in both "towers",
        that is, the features specified in the tensor schema with the exception of `excluded_features`.\n
        To create an instance of TwoTower with other types of blocks, use the class constructor.

        :param schema: tensor schema object with metainformation about features.
        :param item_reference_path: Path to dataframe
            with all items with features used in ``item_encoder`` (item "tower").
            **Note:**\n
            1. Dataframe columns must be already encoded via the same encoders used in `query_encoder` (user "tower").\n
            2. Item reference is constructed only on features with source of FeatureSource.ITEM_FEATURES in tensor
            schema so an identificator of items ("item_id") should be marked as FeatureSource.ITEM_FEATURES too.
        :param embedding_dim: embeddings dimension in both towers. Default: ``192``.
        :param num_heads: number of heads  in user tower SasRec layers. Default: ``4``.
        :param num_blocks: number of blocks  in user tower SasRec layers. Default: ``2``.
        :param max_sequence_length: maximun length of sequence in user tower SasRec layers. Default: ``50``.
        :param dropout: dropout value in both towers. Default: ``0.3``
        :param excluded_features: A list containing the names of features
            for which you do not need to generate an embedding.
            Fragments from this list are expected to be contained in ``schema``.
            Default: ``None``.
        :param categorical_list_feature_aggregation_method: Mode to aggregate tokens
            in token item representation (categorical list only).
            Default: ``"sum"``.
        :return: an instance of TwoTower class.
        """
        from replay.nn import DefaultAttentionMask, SequenceEmbedding, SumAggregator, SwiGLUEncoder
        from replay.nn.loss import CE
        from replay.nn.sequential import PositionAwareAggregator, SasRecTransformerLayer

        # check 463-468
        # excluded_features = list(set(excluded_features or []))

        excluded_features = [
            schema.query_id_feature_name,
            schema.timestamp_feature_name,
            *(excluded_features or []),
        ]
        excluded_features = list(set(excluded_features))

        feature_names = set(schema.names) - set(excluded_features)

        common_aggregator = SumAggregator(embedding_dim=embedding_dim)
        return cls(
            schema=schema,
            embedder=SequenceEmbedding(
                schema=schema,
                categorical_list_feature_aggregation_method=categorical_list_feature_aggregation_method,
                excluded_features=excluded_features,
            ),
            attn_mask_builder=DefaultAttentionMask(
                reference_feature_name=schema.item_id_feature_name,
                num_heads=num_heads,
            ),
            query_tower_feature_names=feature_names,
            item_tower_feature_names=feature_names,
            query_embedding_aggregator=PositionAwareAggregator(
                embedding_aggregator=common_aggregator,
                max_sequence_length=max_sequence_length,
                dropout=dropout,
            ),
            item_embedding_aggregator=common_aggregator,
            query_encoder=SasRecTransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_blocks=num_blocks,
                dropout=dropout,
                activation="relu",
            ),
            query_tower_output_normalization=torch.nn.LayerNorm(embedding_dim),
            item_encoder=SwiGLUEncoder(embedding_dim=embedding_dim, hidden_dim=2*embedding_dim),
            item_reference_path=item_reference_path,
            loss=CE(padding_idx=schema.item_id_features.item().padding_value),
            context_merger=None,
        )

    def reset_parameters(self) -> None:
        self.body.reset_parameters()

    def get_logits(
        self,
        model_embeddings: torch.Tensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        Function for tying last hidden states of query "tower" and set of item embeddings from item "tower"
        via dot product in the model head.

        :param model_embeddings: last hidden state of query tower.
        :param candidates_to_score: IDs of items to be scored.
            These IDs are used for obtaining item embeddings from item tower.
            If is setted to ``None``, all item embeddings from item tower will be used.
            Default: ``None``.
        :return: logits.
        """
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
            feature_tensors=feature_tensors, padding_mask=padding_mask, candidates_to_score=candidates_to_score
        )
