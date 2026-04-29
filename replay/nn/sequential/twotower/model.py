from collections.abc import Sequence
from typing import Protocol

import torch

from replay.data.nn import TensorMap, TensorSchema
from replay.nn.agg import AggregatorProto
from replay.nn.head import EmbeddingTyingHead
from replay.nn.loss import LossProto
from replay.nn.mask import AttentionMaskProto
from replay.nn.normalization import NormalizerProto
from replay.nn.output import InferenceOutput, TrainOutput
from replay.nn.utils import warning_is_not_none

from .reader import FeaturesReaderProtocol


class EmbedderProto(Protocol):
    @property
    def feature_names(self) -> Sequence[str]: ...

    def forward(
        self,
        feature_tensors: TensorMap,
        feature_names: Sequence[str] | None = None,
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
    The Query Tower of TwoTower model.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        embedder: EmbedderProto,
        embedding_aggregator: AggregatorProto,
        attn_mask_builder: AttentionMaskProto,
        encoder: QueryEncoderProto,
        output_normalization: NormalizerProto,
    ):
        """
        :param feature_names: a sequence of names used in a query tower.
        :param embedder: An object of a class that performs the logic of
            generating embeddings from an input batch.
        :param embedding_aggregator: An object of a class that performs
            the logic of aggregating multiple embeddings of the query tower.
        :param attn_mask_builder: An object of a class that performs the logic of
            generating an attention mask based on the features and padding mask given to the model.
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
            The ``False`` value indicates that the corresponding ``key`` value will be ignored.
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


class ItemTower(torch.nn.Module):
    """
    The Item Tower of the TwoTower model.

    **Note**: ItemTower loads feature tensors of all items to memory.
    """

    FEATURE_BUFFER_PREFIX = "item_reference_"

    def __init__(
        self,
        schema: TensorSchema,
        item_features_reader: FeaturesReaderProtocol,
        embedder: EmbedderProto,
        embedding_aggregator: AggregatorProto,
        encoder: ItemEncoderProto,
    ):
        """
        :param schema: a tensor schema object with meta information on features.
        :param item_features_reader: A class that implements reading features,
            processing them, and converting them to ``torch.Tensor`` for ItemTower.
            You can use :class:`replay.nn.sequential.twotower.FeaturesReader` as a standard class.\n
            But you can implement your own feature processing,
            just follow the :class:`replay.nn.sequential.twotower.FeaturesReaderProtocol` protocol.
        :param feature_names: a sequence of names used in the item tower.
        :param embedder: An object of a class that performs the logic of
            generating embeddings from input data.
        :param embedding_aggregator: An object of a class that performs
            the logic of aggregating multiple embeddings.
        :param encoder: An object of a class that performs the logic of generating
            an item hidden embedding representation based for
            the features got from ``item_features_reader``.
        """
        super().__init__()
        self.embedder = embedder
        self.feature_names = item_features_reader.feature_names
        self.embedding_aggregator = embedding_aggregator
        self.encoder = encoder

        for feature_name in schema:
            if feature_name not in self.feature_names:
                continue
            self.register_buffer(
                f"{self.FEATURE_BUFFER_PREFIX}{feature_name}", item_features_reader[feature_name], persistent=True
            )

        self.register_buffer("cache", None, persistent=True)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        cache_key = f"{prefix}cache"
        cache = state_dict.pop(cache_key, None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if cache is not None:
            assert cache.shape[0] == self._get_any_feature_buffer().shape[0]
            assert cache.shape[1] == self.embedding_aggregator.embedding_dim
            self.cache = cache

    @classmethod
    def from_item_features(
        cls,
        item_features: dict[str, torch.Tensor],
        embedder: EmbedderProto,
        embedding_aggregator: AggregatorProto,
        encoder: ItemEncoderProto,
    ) -> "ItemTower":
        """
        Build :class:`ItemTower` from preloaded item feature tensors.
        Unlike the constructor, this method does not use a reader object
        and therefore skips the reader's internal input-processing logic.
        It expects the already processed result in the `item_features` argument.`

        :param item_features: Mapping from feature name to a tensor with values for all items.
            Every tensor is registered as a persistent :attr:`FEATURE_BUFFER_PREFIX` buffer.
        :param embedder: An object of a class that performs the logic of
            generating embeddings from input data.
        :param embedding_aggregator: An object of a class that performs
            the logic of aggregating multiple embeddings.
        :param encoder: An object of a class that performs the logic of generating
            an item hidden embedding representation based for
            the features got from ``item_features_reader``.
        :returns: Initialized :class:`ItemTower` instance with item reference buffers and empty cache.
        """
        model = cls.__new__(cls)
        torch.nn.Module.__init__(model)

        model.embedder = embedder
        model.feature_names = list(item_features)
        model.embedding_aggregator = embedding_aggregator
        model.encoder = encoder

        for feature_name, feature_tensor in item_features.items():
            model.register_buffer(f"{cls.FEATURE_BUFFER_PREFIX}{feature_name}", feature_tensor, persistent=True)

        model.register_buffer("cache", None, persistent=True)
        return model

    @classmethod
    def from_checkpoint(
        cls,
        state_dict: dict[str, torch.Tensor],
        embedder: EmbedderProto,
        embedding_aggregator: AggregatorProto,
        encoder: ItemEncoderProto,
        **kwargs,
    ) -> "ItemTower":
        """
        Restore :class:`ItemTower` from checkpoint state dictionary.

        The method infers required item reference buffers from :attr:`FEATURE_BUFFER_PREFIX*` entries
        in ``state_dict``, creates a new :class:`ItemTower` instance, and loads parameters
        and buffers via :meth:`torch.nn.Module.load_state_dict`.

        A checkpoint can also be loaded in the standard way by constructing :class:`ItemTower` via the constructor
        and then calling :meth:`torch.nn.Module.load_state_dict`.
        This method is a convenience wrapper that avoids explicit creation of a feature reader instance.

        :param state_dict: A checkpoint state dictionary containing module parameters and buffers.
        :param embedder: An object of a class used to embed item feature tensors.
            Must match to the ``state_dict`` embedder.
        :param embedding_aggregator: An object of a class used to aggregate per-feature embeddings.
            Must match to the ``state_dict`` embedding_aggregator.
        :param encoder: An object of a class used to encode aggregated item embeddings.
            Must match to the ``state_dict`` encoder.
        :param kwargs: Additional keyword arguments forwarded to
            :meth:`torch.nn.Module.load_state_dict` (for example, ``strict``).
        :returns: Restored :class:`ItemTower` instance.
        """
        item_features = {
            key.removeprefix(cls.FEATURE_BUFFER_PREFIX): value
            for key, value in state_dict.items()
            if key.startswith(cls.FEATURE_BUFFER_PREFIX)
        }
        if not item_features:
            msg = f"Checkpoint does not contain {cls.FEATURE_BUFFER_PREFIX=} buffers."
            raise ValueError(msg)

        model = cls.from_item_features(
            item_features=item_features,
            embedder=embedder,
            embedding_aggregator=embedding_aggregator,
            encoder=encoder,
        )
        model.load_state_dict(state_dict, **kwargs)
        return model

    def reset_parameters(self) -> None:
        self.embedding_aggregator.reset_parameters()
        self.encoder.reset_parameters()

    def get_feature_buffer(self, feature_name: str) -> torch.Tensor:
        buffer_name = f"{self.FEATURE_BUFFER_PREFIX}{feature_name}"
        return self.get_buffer(buffer_name)

    def _get_any_feature_buffer(self) -> torch.Tensor:
        feature_name = next(iter(self.feature_names))
        return self.get_feature_buffer(feature_name)

    def forward(
        self,
        candidates_to_score: torch.LongTensor | None = None,
    ):
        """
        :param candidates_to_score: IDs of items used for obtaining item embeddings from the item tower.
            If it is set to ``None``, all item embeddings from the item tower will be returned.
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
    Foundation for the TwoTower model that creates the “tower” query and “tower” item.

    For usage of the two tower model, an instance of this class should be passed to
    :class:`replay.nn.sequential.twotower.TwoTower` with any loss from :ref:`Losses <Losses>`.
    """

    def __init__(
        self,
        schema: TensorSchema,
        embedder: EmbedderProto,
        attn_mask_builder: AttentionMaskProto,
        query_tower_feature_names: Sequence[str],
        query_embedding_aggregator: AggregatorProto,
        item_embedding_aggregator: AggregatorProto,
        query_encoder: QueryEncoderProto,
        query_tower_output_normalization: NormalizerProto,
        item_encoder: ItemEncoderProto,
        item_features_reader: FeaturesReaderProtocol,
    ):
        """
        :param schema: tensor schema object with metainformation about features.
        :param embedder: An object of a class that performs the logic of
            generating embeddings from an input batch.\n
            The same object is used to generate embeddings in different towers.
        :param query_tower_feature_names: sequence of names used in query tower.
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
        :param attn_mask_builder: An object of a class that performs the logic of
            generating an attention mask based on the features and padding mask given to the model.
        :param item_encoder: An object of a class that performs the logic of generating
            an item hidden embedding representation based on
            features and aggregated embeddings of ``item_tower_feature_names``.
            Item encoder uses item reference which is created based on ``item_features_path``.
        :param item_features_reader: A class that implements reading features,
            processing them, and converting them to ``torch.Tensor`` for ItemTower.
            You can use :class:`replay.nn.sequential.twotower.FeaturesReader` as a standard class.\n
            But you can implement your own feature processing,
            just follow the :class:`replay.nn.sequential.twotower.FeaturesReaderProtocol` protocol.

        """
        super().__init__()
        self.embedder = embedder
        feature_names_union = set(query_tower_feature_names) | set(item_features_reader.feature_names)
        feature_names_not_in_emb = feature_names_union - set(self.embedder.feature_names)
        if len(feature_names_not_in_emb) != 0:
            msg = f"Feature names found that embedder does not support {list(feature_names_not_in_emb)}"
            raise ValueError(msg)

        self.query_tower = QueryTower(
            query_tower_feature_names,
            embedder,
            query_embedding_aggregator,
            attn_mask_builder,
            query_encoder,
            query_tower_output_normalization,
        )
        self.item_tower = ItemTower(
            schema,
            item_features_reader,
            embedder,
            item_embedding_aggregator,
            item_encoder,
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
    Generic implementation TwoTower architecture with two independent “towers” (encoders)
    which encode separate inputs. In recommender systems they are typically a query tower and item tower.
    The output hidden states of each "tower" are fused via the dot product in the model head.

    Source paper: https://doi.org/10.1145/3366424.3386195

    Example:

    .. code-block:: python

        >>> import pandas as pd
        >>> from replay.data import FeatureHint, FeatureSource, FeatureType
        >>> from replay.data.nn import TensorFeatureInfo, TensorFeatureSource, TensorSchema
        >>> from replay.nn.agg import SumAggregator
        >>> from replay.nn.embedding import SequenceEmbedding
        >>> from replay.nn.ffn import SwiGLUEncoder
        >>> from replay.nn.mask import DefaultAttentionMask
        >>> from replay.nn.loss import CESampled
        >>> from replay.nn.sequential import PositionAwareAggregator, SasRecTransformerLayer
        >>> from replay.nn.sequential.twotower import FeaturesReader, TwoTowerBody, TwoTower
        ...
        >>> NUM_UNIQUE_ITEMS = 200 # number of unique item_id in the item catalog
        >>> tensor_schema = TensorSchema(
        ...     [
        ...         TensorFeatureInfo(
        ...             "item_id",
        ...             is_seq=True,
        ...             feature_type=FeatureType.CATEGORICAL,
        ...             embedding_dim=256,
        ...             padding_value=NUM_UNIQUE_ITEMS,
        ...             cardinality=NUM_UNIQUE_ITEMS,
        ...             feature_hint=FeatureHint.ITEM_ID,
        ...             feature_sources=[TensorFeatureSource(FeatureSource.INTERACTIONS, "item_id")]
        ...         ),
        ...     ]
        ... )
        >>> # encoded item features including item_id
        >>> ITEM_FEATURES_PATH = "item_catalog_encoded.parquet"
        >>> item_catalog_encoded = pd.DataFrame({"item_id": [i for i in range(NUM_UNIQUE_ITEMS)]})
        >>> item_catalog_encoded.to_parquet(ITEM_FEATURES_PATH)
        ...
        >>> common_aggregator = SumAggregator(embedding_dim=256)
        ...
        >>> body = TwoTowerBody(
        ...     schema=tensor_schema,
        ...     embedder=SequenceEmbedding(schema=tensor_schema),
        ...     attn_mask_builder=DefaultAttentionMask(
        ...         reference_feature_name=tensor_schema.item_id_feature_name,
        ...         num_heads=2,
        ...     ),
        ...     query_tower_feature_names=tensor_schema.names,
        ...     query_embedding_aggregator=PositionAwareAggregator(
        ...         embedding_aggregator=common_aggregator,
        ...         max_sequence_length=100,
        ...         dropout=0.2,
        ...     ),
        ...     item_embedding_aggregator=common_aggregator,
        ...     query_encoder=SasRecTransformerLayer(
        ...        embedding_dim=256,
        ...        num_heads=2,
        ...        num_blocks=2,
        ...        dropout=0.3,
        ...        activation="relu",
        ...     ),
        ...     query_tower_output_normalization=torch.nn.LayerNorm(256),
        ...     item_encoder=SwiGLUEncoder(embedding_dim=256, hidden_dim=2*256),
        ...     item_features_reader=FeaturesReader(
        ...         schema=tensor_schema,
        ...         metadata={"item_id": {}},
        ...         path=ITEM_FEATURES_PATH,
        ...     ),
        ... )
        >>> twotower = TwoTower(
        ...     body=body,
        ...     loss=CESampled(ignore_index=tensor_schema[tensor_schema.item_id_feature_name].padding_value),
        ... )

    """

    def __init__(
        self,
        body: TwoTowerBody,
        loss: LossProto,
        context_merger: ContextMergerProto | None = None,
    ):
        """
        :param body: An instance of TwoTowerBody.
        :param loss: An object of a class that performs loss calculation
            based on hidden states from the model, positive and optionally negative labels.
        :param context_merger: An object of a class that performs fusing query encoder hidden state
            with input feature tensors.
            Default: ``None``.
        """
        super().__init__()
        self.body = body
        self.head = EmbeddingTyingHead()
        self.loss = loss
        self.context_merger = context_merger
        self.loss.logits_callback = self.get_logits

        self.reset_parameters()

    @classmethod
    def from_params(
        cls,
        schema: TensorSchema,
        item_features_reader: FeaturesReaderProtocol,
        embedding_dim: int = 192,
        num_heads: int = 4,
        num_blocks: int = 2,
        max_sequence_length: int = 50,
        dropout: float = 0.3,
        excluded_features: list[str] | None = None,
        categorical_list_feature_aggregation_method: str = "sum",
    ) -> "TwoTower":
        """
        A class method for fast creation of the TwoTower instance.\n
        The item "tower" is a SwiGLU encoder (MLP with SwiGLU activation),\n
        the user "tower" is a SasRec transformer layers, and loss is Cross-Entropy loss.\n
        Embeddings of every feature in both "towers" are aggregated via sum.
        The same features are used in both "towers",
        that is, the features specified in the tensor schema with the exception of `excluded_features`.\n
        To create an instance of TwoTower with other types of blocks, please use the class constructor.

        :param schema: a tensor schema object with meta information on features.
        :param item_features_reader: A class that implements reading features,
            processing them, and converting them to ``torch.Tensor`` for ItemTower.
            You can use :class:`replay.nn.sequential.twotower.FeaturesReader` as a standard class.\n
            But you can implement your own feature processing,
            just follow the :class:`replay.nn.sequential.twotower.FeaturesReaderProtocol` protocol.
        :param embedding_dim: embeddings dimension in both towers. Default: ``192``.
        :param num_heads: number of heads in user tower layers. Default: ``4``.
        :param num_blocks: number of blocks in user tower layers. Default: ``2``.
        :param max_sequence_length: maximun length of a sequence in user tower SasRec layers. Default: ``50``.
        :param dropout: a dropout value in both towers. Default: ``0.3``
        :param excluded_features: A list containing the names of features
            for which you do not need to generate an embedding.
            Fragments from this list are expected to be contained in ``schema``.
            Default: ``None``.
        :param categorical_list_feature_aggregation_method: Mode to aggregate tokens
            in token item representation (categorical list only).
            Default: ``"sum"``.
        :return: an instance of TwoTower class.
        """
        from replay.nn.agg import SumAggregator
        from replay.nn.embedding import SequenceEmbedding
        from replay.nn.ffn import SwiGLUEncoder
        from replay.nn.loss import CE
        from replay.nn.mask import DefaultAttentionMask
        from replay.nn.sequential import PositionAwareAggregator, SasRecTransformerLayer

        excluded_features = [
            schema.query_id_feature_name,
            schema.timestamp_feature_name,
            *(excluded_features or []),
        ]
        excluded_features = list(set(excluded_features))

        feature_names = set(schema.names) - set(excluded_features)

        common_aggregator = SumAggregator(embedding_dim=embedding_dim)
        return cls(
            TwoTowerBody(
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
                item_encoder=SwiGLUEncoder(embedding_dim=embedding_dim, hidden_dim=2 * embedding_dim),
                item_features_reader=item_features_reader,
            ),
            loss=CE(ignore_index=schema.item_id_features.item().padding_value),
            context_merger=None,
        )

    def reset_parameters(self) -> None:
        self.body.reset_parameters()

    def get_logits(
        self,
        model_embeddings: torch.Tensor,
        candidates_to_score: torch.LongTensor | None = None,
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
        candidates_to_score: torch.LongTensor | None = None,
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
        candidates_to_score: torch.LongTensor | None = None,
        positive_labels: torch.LongTensor | None = None,
        negative_labels: torch.LongTensor | None = None,
        target_padding_mask: torch.BoolTensor | None = None,
    ) -> TrainOutput | InferenceOutput:
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
            feature_tensors=feature_tensors, padding_mask=padding_mask, candidates_to_score=candidates_to_score
        )
