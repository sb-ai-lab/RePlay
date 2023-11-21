import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, cast

import torch

from replay.data.nn import TensorFeatureInfo, TensorMap, TensorSchema


class BertModel(torch.nn.Module):
    """
    BERT model
    """

    def __init__(
        self,
        schema: TensorSchema,
        max_len: int = 100,
        embed_size: int = 256,
        num_blocks: int = 2,
        num_heads: int = 4,
        num_passes_over_block: int = 1,
        dropout: float = 0.1,
        enable_positional_embedding: bool = True,
        enable_embedding_tying: bool = False,
    ) -> None:
        """
        Args:
            schema (TensorSchema): Tensor schema of features.
            max_len (int): Max length of sequence.
                Default: ``100``.
            embed_size (int): Embedding size.
                Default: ``256``.
            num_blocks (int): Number of Transformer blocks.
                Default: ``2``.
            num_heads (int): Number of Attention heads.
                Default: ``4``.
            num_passes_over_block (int): Number of times to pass data over each Transformer block.
                Default: ``1``.
            dropout (float): Dropout rate.
                Default: ``0.1``.
            enable_positional_embedding (bool): Add positional embedding to the result.
                Default: ``True``.
            enable_embedding_tying (bool): Use embedding tying head.
                Default: ``False``.
        """
        super().__init__()

        self.schema = schema
        self.max_len = max_len
        self.embed_size = embed_size
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
                    embed_size,
                    num_heads,
                    4 * embed_size,
                    dropout,
                )
                for _ in range(num_blocks)
            ]
        )

        self._head: Union[ClassificationHead, EmbeddingTyingHead]
        if self.enable_embedding_tying:
            self._head = EmbeddingTyingHead(self.item_embedder, self.item_count)
        else:
            self._head = ClassificationHead(embed_size, self.item_count)

        self._init()

    def forward(self, inputs: TensorMap, pad_mask: torch.BoolTensor, token_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Args:
            inputs (TensorMap): Batch of features.
            pad_mask (torch.BoolTensor): Padding mask where 0 - <PAD>, 1 otherwise.
            token_mask (torch.BoolTensor): Token mask where 0 - <MASK> tokens, 1 otherwise.

        Returns:
            torch.Tensor: Calculated scores.
        """
        output_embeddings = self.forward_step(inputs, pad_mask, token_mask)
        all_scores = self.get_logits(output_embeddings)

        return all_scores  # [B x L x E]

    def forward_step(self, inputs: TensorMap, pad_mask: torch.BoolTensor, token_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Args:
            inputs (TensorMap): Batch of features.
            pad_mask (torch.BoolTensor): Padding mask where 0 - <PAD>, 1 otherwise.
            token_mask (torch.BoolTensor): Token mask where 0 - <MASK> tokens, 1 otherwise.

        Returns:
            torch.Tensor: Output embeddings.
        """
        # TODO: Check if input tensors matches provided schema

        # B - batch size
        # L - sequence length (max_len)
        # E - embedding size for tokens fed into transformer

        # (B x L x E)
        x = self.item_embedder(inputs, token_mask)

        # (B x 1 x L x L)
        pad_mask_for_attention = self._get_attention_mask_from_padding(pad_mask)

        # Running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            for _ in range(self.num_passes_over_block):
                x = transformer(x, pad_mask_for_attention)

        return x

    def get_logits(self, out_embeddings: torch.Tensor, item_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Apply head to output embeddings of `forward_step`.

        Args:
            out_embeddings (torch.Tensor): Embeddings after `forward step`.
            item_ids (Optional[torch.LongTensor]): Item ids to calculate scores.
                Default: ``None``.

        Returns:
            torch.Tensor: Logits for each element in `item_ids`.
        """
        return self._head(out_embeddings, item_ids)

    def _get_attention_mask_from_padding(self, pad_mask: torch.BoolTensor) -> torch.BoolTensor:
        # (B x L) -> (B x 1 x L x L)
        pad_mask_for_attention = pad_mask.unsqueeze(1).repeat(1, self.max_len, 1).unsqueeze(1)
        return cast(torch.BoolTensor, pad_mask_for_attention)

    def _init(self) -> None:
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except ValueError:
                pass


class BertEmbedding(torch.nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
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
        Args:
            schema (TensorSchema): Tensor schema of features.
            max_len (int): Max length of sequence.
            dropout (float): Dropout rate.
                Default: ``0.1``.
            enable_positional_embedding (bool): Add positional embedding to the result.
                Default: ``True``.
            aggregation_method (str): Aggregation method for result embedding.
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
                # TODO: Support non-sequential embeddings
                raise NotImplementedError("Non-sequential features is not yet supported")

            if tensor_info.is_cat:
                dim = tensor_info.embedding_dim
            else:
                dim = tensor_info.tensor_dim

            if aggregation_method == "sum":
                if common_dim is None:
                    common_dim = dim

                if dim != common_dim:
                    # TODO: Add linear projection for features?
                    raise ValueError("Dimension of all features must be the same for sum aggregation")
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
        Args:
            inputs (TensorMap): Batch of features.
            token_mask (torch.BoolTensor): Token mask where 0 - <MASK> tokens, 1 otherwise.

        Returns:
            torch.Tensor: Embeddings for input features.
        """
        if self.aggregation_method == "sum":
            aggregated_embedding: torch.Tensor = None  # type: ignore

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
        Returns:
            torch.Tensor: Item embeddings.
        """
        return self.cat_embeddings[self.schema.item_id_feature_name].weight


class CatFeatureEmbedding(torch.nn.Embedding):
    """
    Categorical feature embedding.
    """

    def __init__(self, feature: TensorFeatureInfo) -> None:
        """
        Args:
            feature (TensorFeatureInfo): Categorical tensor feature.
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
        Args:
            max_len (int): Max sequence length.
            d_model (int): Embedding dimension.
        """
        super().__init__()
        self.pe = torch.nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input embedding.

        Returns:
            torch.Tensor: Positional embedding.
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
        Args:
            out_embeddings (torch.Tensor): Embeddings after `forward step`.
            item_ids (Optional[torch.LongTensor]): Item ids to calculate scores.
                Default: ``None``.

        Returns:
            torch.Tensor: Calculated logits.
        """
        item_embeddings = self.get_item_embeddings()
        bias = self.get_bias()
        if item_ids is not None:
            item_embeddings = item_embeddings[item_ids]
            bias = bias[item_ids]

        logits = item_embeddings.matmul(out_embeddings.unsqueeze(-1)).squeeze(-1) + bias
        return logits

    @abstractmethod
    def get_item_embeddings(self) -> torch.Tensor:  # pragma: no cover
        pass

    @abstractmethod
    def get_bias(self) -> torch.Tensor:  # pragma: no cover
        pass


class EmbeddingTyingHead(BaseHead):
    """
    Head that calculate logits for all item_ids given output embeddings.
    """

    def __init__(self, item_embedder: BertEmbedding, n_items: int):
        """
        Args:
            item_embedder (BertEmbedding): Bert embedding.
            n_items (int): Number of items.
        """
        super().__init__()
        self._item_embedder = item_embedder
        self.out_bias = torch.nn.Parameter(torch.Tensor(n_items))
        self.out_bias.data.normal_(0, 0.01)

    def get_item_embeddings(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Item embeddings.
        """
        return self._item_embedder.item_embeddings

    def get_bias(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Bias tensor.
        """
        return self.out_bias


class ClassificationHead(BaseHead):
    """
    Classification head with linear output
    """

    def __init__(self, embed_size: int, n_items: int) -> None:
        """
        Args:
            embed_size (int): Embedding size.
            n_items (int): Number of items.
        """
        super().__init__()
        self.linear = torch.nn.Linear(embed_size, n_items, bias=True)

    def get_item_embeddings(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Item embeddings.
        """
        return self.linear.weight

    def get_bias(self) -> torch.Tensor:
        """
        Returns:
            torch.Tensor: Bias tensor.
        """
        return self.linear.bias


class TransformerBlock(torch.nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(
        self,
        hidden: int,
        attn_heads: int,
        feed_forward_hidden: int,
        dropout: float,
    ) -> None:
        """
        Args:
            hidden (int): Hidden size of transformer.
            attn_heads (int): Head sizes of multi-head attention.
            feed_forward_hidden (int): Feed_forward_hidden, usually 4*hidden_size.
            dropout (float): Dropout rate.
            disentangled (bool): Use disentagled attention instead of default attention.
                Default: ``False``.
        """
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
        self.attention_dropout = torch.nn.Dropout(dropout)
        self.attention_norm = LayerNorm(hidden)

        self.pff = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.pff_dropout = torch.nn.Dropout(dropout)
        self.pff_norm = LayerNorm(hidden)

        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input bert embedding.
            mask (torch.BoolTensor): Mask where 0 - <MASK>, 1 - otherwise.
            pos_embeddings (Optional[torch.Tensor]): Positional embedding.
                Default: ``None``.
            pos_indices (Optional[torch.LongTensor]): Positional indices.
                Default: ``None``.

        Returns:
            torch.Tensor: Embedding after Transformer block applied.
        """
        # Attention + skip-connection
        x_norm = self.attention_norm(x)
        y = x + self.attention_dropout(self.attention(x_norm, x_norm, x_norm, mask))

        # PFF + skip-connection
        z = y + self.pff_dropout(self.pff(self.pff_norm(y)))

        return self.dropout(z)


class Attention(torch.nn.Module):
    """
    Compute Scaled Dot Product Attention
    """

    def __init__(self, dropout: float) -> None:
        """
        Args:
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query (torch.Tensor): Query feature vector.
            key (torch.Tensor): Key feature vector.
            value (torch.Tensor): Value feature vector.
            mask (torch.BoolTensor): Mask where 0 - <MASK>, 1 - otherwise.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of scaled dot product attention
                and attention logits for each element.
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h: int, d_model: int, dropout: float = 0.1) -> None:
        """
        Args:
            h (int): Head sizes of multi-head attention.
            d_model (int): Embedding dimension.
            dropout (float): Dropout rate.
                Default: ``0.1``.
        """
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        # 3 linear projections for Q, K, V
        self.qkv_linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])

        # 2 linear projections for P -> P_q, P_k
        self.pos_linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(2)])

        self.output_linear = torch.nn.Linear(d_model, d_model)

        self.attention = Attention(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Args:
            query (torch.Tensor): Query feature vector.
            key (torch.Tensor): Key feature vector.
            value (torch.Tensor): Value feature vector.
            mask (torch.BoolTensor): Mask where 0 - <MASK>, 1 - otherwise.

        Returns:
            torch.Tensor: Attention outputs.
        """
        batch_size = query.size(0)

        # B - batch size
        # L - sequence length (max_len)
        # E - embedding size for tokens fed into transformer
        # K - max relative distance
        # H - attention head count

        # Do all the linear projections in batch from d_model => h x d_k
        # (B x L x E) -> (B x H x L x (E / H))
        query, key, value = [
            layer(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for layer, x in zip(self.qkv_linear_layers, (query, key, value))
        ]

        x, _ = self.attention(query, key, value, mask)

        # Concat using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class LayerNorm(torch.nn.Module):
    """
    Construct a layernorm module (See citation for details).
    """

    def __init__(self, features: int, eps: float = 1e-6):
        """
        Args:
            features (int): Number of features.
            eps (float): A value added to the denominator for numerical stability.
                Default: ``1e-6``.
        """
        super().__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized input tensor.
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(torch.nn.Module):
    """
    Implements FFN equation.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        """
        Args:
            d_model (int): Embedding dimension.
            d_ff (int): Feed forward dimension, usually 4*d_model.
            dropout (float): Dropout rate.
                Default: ``0.1``.
        """
        super().__init__()
        self.w_1 = torch.nn.Linear(d_model, d_ff)
        self.w_2 = torch.nn.Linear(d_ff, d_model)
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Position wised output.
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class GELU(torch.nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated input tensor.
        """
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
