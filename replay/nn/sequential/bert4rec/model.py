from collections.abc import Sequence
from typing import Literal, Optional, Protocol, Union

import torch

from replay.data.nn import TensorMap, TensorSchema
from replay.nn.agg import AggregatorProto
from replay.nn.head import EmbeddingTyingHead, HeadProto
from replay.nn.loss import LossProto
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


class BERT4RecBody(torch.nn.Module):
    """
    Implementation of the body of the BERT4Rec model.\n
    It can include various self-written blocks for modifying the model,
    but the sequence of applying layers is fixed in accordance with the BERT4Rec architecture.
    """

    def __init__(
        self,
        embedder: EmbedderProto,
        embedding_aggregator: AggregatorProto,
        encoder: EncoderProto,
    ):
        """
        :param embedder: An object of a class that performs the logic of
            generating embeddings from an input set of tensors.
        :param embedding_aggregator: An object of a class that performs the logic of aggregating multiple embeddings.\n
            For example, it can be a ``sum`` or a ``concatenation``. For BERT4Rec, the aggregator is also expected
            to apply token masking based on the ``token_mask`` tensor from ``feature_tensors``.
        :param encoder: An object of a class that performs the logic of generating
            a hidden embedding representation based on
            features, padding masks, and aggregated embeddings.
        """
        super().__init__()
        self.embedder = embedder
        self.embedding_aggregator = embedding_aggregator
        self.encoder = encoder

    def reset_parameters(self) -> None:
        self.embedder.reset_parameters()
        self.embedding_aggregator.reset_parameters()
        self.encoder.reset_parameters()

    def forward(
        self,
        feature_tensors: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        :param feature_tensors: a dictionary of tensors to generate embeddings.
            It must contain a ``token_mask`` key where ``False`` marks masked tokens.
        :param padding_mask: A mask of shape ``(batch_size, sequence_length)``
            indicating which elements within ``key`` to ignore for the purpose of attention (i.e. treat as "padding").
            ``False`` value indicates that the corresponding ``key`` value will be ignored.
        :returns: The final hidden state.\n
            Expected shape: ``(batch_size, sequence_length, embedding_dim)``
        """
        embeddings = self.embedder(feature_tensors)
        agg_emb: torch.Tensor = self.embedding_aggregator(embeddings, feature_tensors["token_mask"])
        assert agg_emb.dim() == 3

        hidden_state: torch.Tensor = self.encoder(
            feature_tensors=feature_tensors,
            input_embeddings=agg_emb,
            padding_mask=padding_mask,
        )
        assert agg_emb.size() == hidden_state.size()

        return hidden_state


class BERT4Rec(torch.nn.Module):
    """
    A model using the BERT4Rec architecture as a hidden state generator.
    The hidden states are multiplied by the item embeddings,
    resulting in logits for each of the items.

    Source paper: https://arxiv.org/pdf/1904.06690.

    Example:

    .. code-block:: python

        >>> from replay.data import FeatureHint, FeatureSource, FeatureType
        >>> from replay.data.nn import TensorFeatureInfo, TensorFeatureSource, TensorSchema
        >>> from replay.nn.agg import SumAggregator
        >>> from replay.nn.embedding import SequenceEmbedding
        >>> from replay.nn.loss import CE
        >>> from replay.nn.sequential import (
        ...     BERT4Rec,
        ...     BERT4RecBody,
        ...     BERT4RecTransformerLayer,
        ...     BERT4RecPositionAwareAggregator,
        ... )
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
        ...
        >>> body = BERT4RecBody(
        ...    embedder=SequenceEmbedding(
        ...        schema=tensor_schema,
        ...    ),
        ...    embedding_aggregator=BERT4RecPositionAwareAggregator(
        ...        embedding_aggregator=SumAggregator(embedding_dim=256),
        ...        max_sequence_length=100,
        ...        dropout=0.2,
        ...    ),
        ...    encoder=BERT4RecTransformerLayer(
        ...        embedding_dim=256,
        ...        num_heads=2,
        ...        num_blocks=2,
        ...        feed_forward_hidden=1024,
        ...        dropout=0.3,
        ...        activation="relu",
        ...    ),
        ... )
        >>> bert4rec = BERT4Rec(
        ...     body=body,
        ...     loss=CE(ignore_index=tensor_schema[tensor_schema.item_id_feature_name].padding_value),
        ... )

    """

    def __init__(
        self,
        body: BERT4RecBody,
        loss: LossProto,
        head: HeadProto | None = None,
    ):
        """
        :param body: An instance of BERT4RecBody.
        :param loss: An object of a class that performs loss calculation
            based on hidden states from the model, positive and optionally negative labels.
        :param head: An object of a class that performs logits calculation from hidden states and item embeddings.
            If ``None``, ``EmbeddingTyingHead`` is used.
        """
        super().__init__()
        self.body = body
        self.head = EmbeddingTyingHead() if head is None else head
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
        feed_forward_hidden: Optional[int] = None,
        activation: Literal["relu", "gelu"] = "gelu",
        excluded_features: Optional[list[str]] = None,
        categorical_list_feature_aggregation_method: Literal["sum", "mean", "max"] = "sum",
        head: HeadProto | None = None,
    ) -> "BERT4Rec":
        """
        Build a BERT4Rec model from common hyperparameters.

        :param schema: Tensor schema of input sequence features.
        :param embedding_dim: Common hidden size used by embeddings, aggregation, and transformer blocks.
        :param num_heads: Number of attention heads.
        :param num_blocks: Number of transformer blocks.
        :param max_sequence_length: Maximum sequence length for learned positional embeddings.
        :param dropout: Dropout probability.
        :param feed_forward_hidden: Hidden size of the transformer feed-forward network.
            If ``None``, it is set to ``4 * embedding_dim``.
        :param activation: Activation function used in the feed-forward network.
        :param excluded_features: Additional schema feature names to exclude from embedding generation.
            Query id and timestamp features are always excluded.
        :param categorical_list_feature_aggregation_method: Method to aggregate categorical-list feature values.
        :param head: Optional logits head. If ``None``, ``EmbeddingTyingHead`` is used.
        :returns: Initialized BERT4Rec model with CE loss.
        """
        from replay.nn.agg import SumAggregator
        from replay.nn.embedding import SequenceEmbedding
        from replay.nn.loss import CE

        from .agg import BERT4RecPositionAwareAggregator
        from .transformer import BERT4RecTransformerLayer

        excluded_features = [
            schema.query_id_feature_name,
            schema.timestamp_feature_name,
            *(excluded_features or []),
        ]
        excluded_features = list(set(excluded_features))

        body = BERT4RecBody(
            embedder=SequenceEmbedding(
                schema=schema,
                categorical_list_feature_aggregation_method=categorical_list_feature_aggregation_method,
                excluded_features=excluded_features,
            ),
            embedding_aggregator=BERT4RecPositionAwareAggregator(
                embedding_aggregator=SumAggregator(embedding_dim=embedding_dim),
                max_sequence_length=max_sequence_length,
                dropout=dropout,
            ),
            encoder=BERT4RecTransformerLayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                num_blocks=num_blocks,
                feed_forward_hidden=feed_forward_hidden or 4 * embedding_dim,
                dropout=dropout,
                activation=activation,
            ),
        )
        return cls(
            body=body,
            loss=CE(ignore_index=schema.item_id_features.item().padding_value),
            head=head,
        )

    def reset_parameters(self) -> None:
        self.body.reset_parameters()
        self.head.reset_parameters()

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
            It must contain a ``token_mask`` key where ``False`` marks masked tokens.
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
