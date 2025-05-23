import contextlib
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from replay.data.nn import TensorFeatureInfo, TensorMap, TensorSchema


class Bert4RecModel(torch.nn.Module):
    """
    BERT model
    """

    def __init__(
        self,
        schema: TensorSchema,
        max_len: int = 100,
        hidden_size: int = 256,
        num_blocks: int = 2,
        num_heads: int = 4,
        num_passes_over_block: int = 1,
        dropout: float = 0.1,
        enable_positional_embedding: bool = True,
        enable_embedding_tying: bool = False,
    ) -> None:
        """
        :param schema: Tensor schema of features.
        :param max_len: Max length of sequence.
            Default: ``100``.
        :param hidden_size: Hidden size of transformer.
            Default: ``256``.
        :param num_blocks: Number of Transformer blocks.
            Default: ``2``.
        :param num_heads: Number of Attention heads.
            Default: ``4``.
        :param num_passes_over_block: Number of times to pass data over each Transformer block.
            Default: ``1``.
        :param dropout: Dropout rate.
            Default: ``0.1``.
        :param enable_positional_embedding: Add positional embedding to the result.
            Default: ``True``.
        :param enable_embedding_tying: Use embedding tying head.
            Default: ``False``.
        """
        super().__init__()

        self.schema = schema
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_passes_over_block = num_passes_over_block
        self.dropout = dropout
        self.enable_positional_embedding = enable_positional_embedding
        self.enable_embedding_tying = enable_embedding_tying
        self.item_count = schema.item_id_features.item().cardinality
        assert self.item_count

        self.item_embedder = BertEmbedding(
            schema,
            max_len=max_len,
            dropout=dropout,
            enable_positional_embedding=enable_positional_embedding,
        )

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size,
                    num_heads,
                    4 * hidden_size,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self._head: Union[ClassificationHead, EmbeddingTyingHead]
        if self.enable_embedding_tying:
            self._head = EmbeddingTyingHead(self.item_embedder, self.item_count)
        else:
            self._head = ClassificationHead(hidden_size, self.item_count)

        self._init()

    def forward(self, inputs: TensorMap, pad_mask: torch.BoolTensor, token_mask: torch.BoolTensor) -> torch.Tensor:
        """
        :param inputs: Batch of features.
        :param pad_mask: Padding mask where 0 - <PAD>, 1 otherwise.
        :param token_mask: Token mask where 0 - <MASK> tokens, 1 otherwise.

        :returns: Calculated scores.
        """
        output_embeddings = self.forward_step(inputs, pad_mask, token_mask)
        all_scores = self.get_logits(output_embeddings)

        return all_scores  # [B x L x E]

    def predict(
        self,
        inputs: TensorMap,
        pad_mask: torch.BoolTensor,
        token_mask: torch.BoolTensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        :param inputs: Batch of features.
        :param pad_mask: Padding mask where 0 - <PAD>, 1 otherwise.
        :param token_mask: Token mask where 0 - <MASK> tokens, 1 otherwise.
        :param candidates_to_score: Item ids to calculate scores.
            if `None` predicts for all items

        :returns: Calculated scores among canditates_to_score items.
        """
        # final_emb: [B x E]
        final_emb = self.get_query_embeddings(inputs, pad_mask, token_mask)
        candidate_scores = self.get_logits(final_emb, candidates_to_score)
        return candidate_scores

    def forward_step(self, inputs: TensorMap, pad_mask: torch.BoolTensor, token_mask: torch.BoolTensor) -> torch.Tensor:
        """

        :param inputs (TensorMap): Batch of features.
        :param pad_mask (torch.BoolTensor): Padding mask where 0 - <PAD>, 1 otherwise.
        :param token_mask (torch.BoolTensor): Token mask where 0 - <MASK> tokens, 1 otherwise.

        :returns: Output embeddings.
        """

        # B - batch size
        # L - sequence length (max_len)
        # E - embedding size for tokens fed into transformer

        # (B x L x E)
        x = self.item_embedder(inputs, token_mask)

        # Running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            for _ in range(self.num_passes_over_block):
                x = transformer(x, pad_mask)

        return x

    def get_logits(self, out_embeddings: torch.Tensor, item_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Apply head to output embeddings of `forward_step`.

        :param out_embeddings: Embeddings after `forward step`.
        :param item_ids: Item ids to calculate scores.
            Default: ``None``.

        :returns: Logits for each element in `item_ids`.
        """
        return self._head(out_embeddings, item_ids)

    def get_query_embeddings(self, inputs: TensorMap, pad_mask: torch.BoolTensor, token_mask: torch.BoolTensor):
        """
        :param inputs: Batch of features.
        :param pad_mask: Padding mask where 0 - <PAD>, 1 otherwise.
        :param token_mask: Token mask where 0 - <MASK> tokens, 1 otherwise.

        :returns: Query embeddings.
        """
        return self.forward_step(inputs, pad_mask, token_mask)[:, -1, :]

    def _init(self) -> None:
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)


class BertEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        sum of all these features are output of BertEmbedding
    """

    def __init__(
        self,
        schema: TensorSchema,
        max_len: int,
        dropout: float = 0.1,
        enable_positional_embedding: bool = True,
        aggregation_method: str = "sum",
    ) -> None:
        """
        :param schema: Tensor schema of features.
        :param max_len: Max length of sequence.
        :param dropout: Dropout rate.
            Default: ``0.1``.
        :param enable_positional_embedding: Add positional embedding to the result.
            Default: ``True``.
        :param aggregation_method: Aggregation method for result embedding.
            Possible values: `"sum"`.
            Default: ``sum``.
        """
        super().__init__()

        self.schema = schema
        self.max_len = max_len
        self.enable_positional_embedding = enable_positional_embedding
        self.aggregation_method = aggregation_method
        self.cat_embeddings = torch.nn.ModuleDict()

        common_dim = None

        for feature_name, tensor_info in schema.items():
            if not tensor_info.is_seq:
                msg = "Non-sequential features is not yet supported"
                raise NotImplementedError(msg)

            dim = tensor_info.embedding_dim if tensor_info.is_cat else tensor_info.tensor_dim

            if aggregation_method == "sum":
                if common_dim is None:
                    common_dim = dim

                if dim != common_dim:
                    msg = "Dimension of all features must be the same for sum aggregation"
                    raise ValueError(msg)
            else:
                raise NotImplementedError()

            if tensor_info.is_cat:
                self.cat_embeddings[feature_name] = CatFeatureEmbedding(tensor_info)

        assert common_dim

        self.embedding_dim = common_dim
        self.dropout = torch.nn.Dropout(p=dropout)
        self.mask_embedding = torch.nn.Embedding(1, common_dim)

        if self.enable_positional_embedding:
            self.position = PositionalEmbedding(max_len=max_len, d_model=common_dim)

    def forward(self, inputs: TensorMap, token_mask: torch.BoolTensor) -> torch.Tensor:
        """
        :param inputs: Batch of features.
        :param token_mask: Token mask where 0 - <MASK> tokens, 1 otherwise.

        :returns: Embeddings for input features.
        """
        if self.aggregation_method == "sum":
            aggregated_embedding: torch.Tensor = None

            for feature_name in self.schema.categorical_features:
                x = inputs[feature_name]
                embedding = self.cat_embeddings[feature_name](x)

                if aggregated_embedding is None:
                    aggregated_embedding = embedding
                else:
                    aggregated_embedding += embedding

            for feature_name in self.schema.numerical_features:
                aggregated_embedding += inputs[feature_name]

        else:
            raise NotImplementedError()

        batch_size = aggregated_embedding.size(0)
        seq_len = aggregated_embedding.size(1)
        embedding_dim = aggregated_embedding.size(2)

        assert seq_len == self.max_len
        assert embedding_dim == self.embedding_dim

        # (B x L) -> (B x L x E)
        #
        # [[0, 1],        [ [[0, 0, 0],
        #  [1, 0]]   -->     [1, 1, 1]],
        #                   [[1, 1, 1],
        #                    [0, 0, 0]] ]
        expanded_mask = token_mask.unsqueeze(-1).expand(-1, -1, embedding_dim)

        # (1 x E) -> (B x L x E)
        #
        # [[1, 2, 3]] -> [ [[1, 2, 3],
        #                   [1, 2, 3]],
        #                  [[1, 2, 3],
        #                   [1, 2, 3]] ]
        expanded_embedding = self.mask_embedding.weight.expand(batch_size, seq_len, -1)

        # Fill masked token embeddings with embedding of [mask] token
        full_embedding = aggregated_embedding.where(expanded_mask, expanded_embedding)

        x = full_embedding
        if self.enable_positional_embedding:
            x += self.position(full_embedding)

        x = self.dropout(x)

        return x

    @property
    def item_embeddings(self) -> torch.Tensor:
        """
        :returns: Item embeddings.
        """
        return self.cat_embeddings[self.schema.item_id_feature_name].weight

    def get_all_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        :returns: copy of all embeddings presented in this layer as a dict.
        """
        embeddings = {
            "item_embedding": self.item_embeddings.data.detach().clone(),
        }
        for feature_name in self.schema:
            if feature_name != self.schema.item_id_feature_name:
                embeddings[feature_name] = self.cat_embeddings[feature_name].weight.data.detach().clone()
        if self.enable_positional_embedding:
            embeddings["positional_embedding"] = self.position.pe.weight.data.detach().clone()

        return embeddings


class CatFeatureEmbedding(torch.nn.Embedding):
    """
    Categorical feature embedding.
    """

    def __init__(self, feature: TensorFeatureInfo) -> None:
        """
        :param feature: Categorical tensor feature.
        """
        assert feature.cardinality
        assert feature.embedding_dim
        super().__init__(feature.cardinality, feature.embedding_dim)


class PositionalEmbedding(torch.nn.Module):
    """
    Positional embedding.
    """

    def __init__(self, max_len: int, d_model: int) -> None:
        """
        :param max_len: Max sequence length.
        :param d_model: Embedding dimension.
        """
        super().__init__()
        self.pe = torch.nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input embedding.

        :returns: Positional embedding.
        """
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class BaseHead(ABC, torch.nn.Module):
    """
    Base abstract head
    """

    def forward(
        self,
        out_embeddings: torch.Tensor,
        item_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        :param out_embeddings: Embeddings after `forward step`.
        :param item_ids: Item ids to calculate scores.
            Default: ``None``.

        :returns: Calculated logits.
        """
        item_embeddings = self.get_item_embeddings()
        bias = self.get_bias()
        if item_ids is not None:
            item_embeddings = item_embeddings[item_ids]
            bias = bias[item_ids]

        logits = torch.nn.functional.linear(out_embeddings, item_embeddings, bias)
        return logits

    @abstractmethod
    def get_item_embeddings(self) -> torch.Tensor:  # pragma: no cover
        """
        :returns: Item embeddings.
        """

    @abstractmethod
    def get_bias(self) -> torch.Tensor:  # pragma: no cover
        """
        :returns: Bias tensor.
        """


class EmbeddingTyingHead(BaseHead):
    """
    Head that calculate logits for all item_ids given output embeddings.
    """

    def __init__(self, item_embedder: BertEmbedding, n_items: int):
        """
        :param item_embedder: Bert embedding.
        :param n_items: Number of items.
        """
        super().__init__()
        self._item_embedder = item_embedder
        self.out_bias = torch.nn.Parameter(torch.Tensor(n_items))
        self.out_bias.data.normal_(0, 0.01)

    def get_item_embeddings(self) -> torch.Tensor:
        """
        :returns: Item embeddings.
        """
        return self._item_embedder.item_embeddings

    def get_bias(self) -> torch.Tensor:
        """
        :returns: Bias tensor.
        """
        return self.out_bias


class ClassificationHead(BaseHead):
    """
    Classification head with linear output
    """

    def __init__(self, hidden_size: int, n_items: int) -> None:
        """
        :param hidden_size: Hidden size of transformer.
        :param n_items: Number of items.
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden_size, n_items, bias=True)

    def get_item_embeddings(self) -> torch.Tensor:
        """
        :returns: Item embeddings.
        """
        return self.linear.weight

    def get_bias(self) -> torch.Tensor:
        """
        :returns: Bias tensor.
        """
        return self.linear.bias


class TransformerBlock(torch.nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(
        self,
        hidden_size: int,
        attn_heads: int,
        feed_forward_hidden: int,
        dropout: float,
    ) -> None:
        """
        :param hidden_size: Hidden size of transformer.
        :param attn_heads: Head sizes of multi-head attention.
        :param feed_forward_hidden: Feed_forward_hidden, usually 4*hidden_size.
        :param dropout: Dropout rate.
        """
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(hidden_size, attn_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.attention_norm = torch.nn.LayerNorm(hidden_size)

        self.pff = PositionwiseFeedForward(d_model=hidden_size, d_ff=feed_forward_hidden, dropout=dropout)
        self.pff_dropout = torch.nn.Dropout(dropout)
        self.pff_norm = torch.nn.LayerNorm(hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        :param x: Input bert embedding.
        :param mask: Mask where 0 - <MASK>, 1 - otherwise.

        :returns: Embedding after Transformer block applied.
        """
        # Attention + skip-connection
        x_norm = self.attention_norm(x)
        attent_emb, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=~mask, need_weights=False)
        y = x + self.attention_dropout(attent_emb)

        # PFF + skip-connection
        z = y + self.pff_dropout(self.pff(self.pff_norm(y)))

        return self.dropout(z)


class PositionwiseFeedForward(torch.nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        :param d_mode: Embedding dimension.
        :param d_ff: Feed forward dimension, usually 4*d_model.
        :param dropout: Dropout rate.
            Default: ``0.1``.
        """
        super().__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor.

        :returns: Position wised output.
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
