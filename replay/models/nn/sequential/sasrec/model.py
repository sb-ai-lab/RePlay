import abc
from typing import Any, Optional, Tuple, Union, cast, Dict

import torch

from replay.data.nn import TensorMap, TensorSchema


# pylint: disable=too-many-instance-attributes
class SasRecModel(torch.nn.Module):
    """
    SasRec model
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        schema: TensorSchema,
        num_blocks: int = 2,
        num_heads: int = 1,
        hidden_size: int = 50,
        max_len: int = 200,
        dropout: float = 0.2,
        ti_modification: bool = False,
        time_span: int = 256,
    ) -> None:
        """
        :param schema: Tensor schema of features.
        :param num_blocks: Number of Transformer blocks.
            Default: ``2``.
        :param num_heads: Number of Attention heads.
            Default: ``1``.
        :param hidden_size: Hidden size of transformer.
            Default: ``50``.
        :param max_len: Max length of sequence.
            Default: ``200``.
        :param dropout: Dropout rate.
            Default: ``0.2``.
        :param ti_modification: Enable time relation.
            Default: ``False``.
        :param time_span: Time span if `ti_modification` is `True`.
            Default: ``256``.
        """
        super().__init__()

        # Hyperparams
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.dropout = dropout
        self.ti_modification = ti_modification
        self.time_span = time_span

        item_count = schema.item_id_features.item().cardinality
        assert item_count
        self.item_count = item_count
        self.padding_idx = item_count

        assert schema.item_id_feature_name
        self.item_feature_name = schema.item_id_feature_name
        self.register_buffer("candidates_to_score", torch.LongTensor(list(range(self.item_count))))

        # Model blocks
        self.masking = SasRecMasks(
            schema=schema,
            padding_idx=self.padding_idx,
        )
        self.item_embedder: Union[TiSasRecEmbeddings, SasRecEmbeddings]
        self.sasrec_layers: torch.nn.Module

        if self.ti_modification:
            self.item_embedder = TiSasRecEmbeddings(
                schema=schema,
                embed_size=self.hidden_size,
                max_len=self.max_len,
                dropout=self.dropout,
                padding_idx=self.padding_idx,
                time_span=self.time_span,
            )
            self.sasrec_layers = TiSasRecLayers(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_blocks=self.num_blocks,
                dropout=self.dropout,
            )
        else:
            self.item_embedder = SasRecEmbeddings(
                schema=schema,
                embed_size=self.hidden_size,
                max_len=self.max_len,
                dropout=self.dropout,
                padding_idx=self.padding_idx,
            )
            self.sasrec_layers = SasRecLayers(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                num_blocks=self.num_blocks,
                dropout=self.dropout,
            )
        self.output_normalization = SasRecNormalizer(
            hidden_size=self.hidden_size,
        )
        self._head = EmbeddingTyingHead(self.item_embedder)
        self._init()

    def forward(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Calculated scores.
        """
        output_embeddings = self.forward_step(feature_tensor, padding_mask)
        all_scores = self.get_logits(output_embeddings)

        return all_scores  # [B x L x E]

    def predict(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.
        :param candidates_to_score: Item ids to calculate scores.
            if `None` predicts for all items

        :returns: Prediction among canditates_to_score items.
        """
        output_emb = self.forward_step(feature_tensor, padding_mask)

        # output_emb: [B x L x E]
        # final_emb: [B x E]
        final_emb = output_emb[:, -1, :]  # last item
        candidate_scores = self.get_logits(final_emb, candidates_to_score)
        return candidate_scores

    def forward_step(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Output embeddings.
        """
        device = feature_tensor[self.item_feature_name].device
        attention_mask, padding_mask, feature_tensor = self.masking(feature_tensor, padding_mask)
        if self.ti_modification:
            seqs, ti_embeddings = self.item_embedder(feature_tensor, padding_mask)
            seqs = self.sasrec_layers(seqs, attention_mask, padding_mask, ti_embeddings, device)
        else:
            seqs = self.item_embedder(feature_tensor, padding_mask)
            seqs = self.sasrec_layers(seqs, attention_mask, padding_mask)
        output_emb = self.output_normalization(seqs)

        return output_emb

    def get_logits(self, out_embeddings: torch.Tensor, item_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Apply head to output embeddings of `forward_step`.

        :param out_embeddings: Embeddings after `forward step`.
        :param item_ids: Item ids to calculate scores.
            Default: ``None``.

        :returns: Logits for each element in `item_ids`.
        """
        return self._head(out_embeddings, item_ids)

    def _init(self) -> None:
        for _, param in self.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except ValueError:
                pass


# pylint: disable=too-few-public-methods
class SasRecMasks:
    """
    SasRec Masks
        1. AttentionMask
        2. PaddingMask
    """

    def __init__(self, schema: TensorSchema, padding_idx: int) -> None:
        """
        :param schema: Tensor schema of features.
        :param padding_idx: Padding indices.
        """
        assert schema.item_id_feature_name
        self.schema = schema
        self.item_feature_name = schema.item_id_feature_name
        self.padding_idx = padding_idx

    def __call__(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor, TensorMap]:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Attention mask, unsqueezed padding mask and output feature tensor.
        """
        input_sequence = feature_tensor[self.item_feature_name]

        attention_mask = ~torch.tril(
            torch.ones((input_sequence.shape[1], input_sequence.shape[1]), dtype=torch.bool)
        ).to(padding_mask.device)

        output_feature_tensor = dict(feature_tensor)
        output_feature_tensor[self.item_feature_name] = input_sequence.masked_fill(
            mask=(~padding_mask),
            value=self.padding_idx,
        )

        padding_mask_unsqueezed = padding_mask.unsqueeze(-1)

        return (
            cast(torch.BoolTensor, attention_mask),
            cast(torch.BoolTensor, padding_mask_unsqueezed),
            output_feature_tensor,
        )


class BaseSasRecEmbeddings(abc.ABC):
    """
    Base SasRec embedding class
    """

    @abc.abstractmethod
    def get_item_weights(self, indices: torch.LongTensor) -> torch.Tensor:  # pragma: no cover
        """
        :param indices: Items indices.

        :returns: Item weights for specific items.
        """

    @abc.abstractmethod
    def get_all_item_weights(self) -> torch.Tensor:  # pragma: no cover
        """
        :returns: Item weights for all items.
        """

    @abc.abstractmethod
    def get_all_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        :returns: copy of all embeddings presented in a layer as a dict.
        """


class EmbeddingTyingHead(torch.nn.Module):
    """
    Head that calculate logits for all item_ids given output embeddings
    """

    def __init__(self, item_embedder: BaseSasRecEmbeddings):
        """
        :param item_embedder: SasRec embedding.
        """
        super().__init__()
        self._item_embedder = item_embedder

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
        if item_ids is not None:
            item_embeddings = self._item_embedder.get_item_weights(item_ids)
        else:
            item_embeddings = self._item_embedder.get_all_item_weights()

        if len(item_embeddings.shape) > 2:  # global_uniform, negative sharing=False, train only
            logits = (item_embeddings * out_embeddings.unsqueeze(-2)).sum(dim=-1)
        else:
            logits = item_embeddings.matmul(out_embeddings.unsqueeze(-1)).squeeze(-1)
        return logits


class SasRecEmbeddings(torch.nn.Module, BaseSasRecEmbeddings):
    """
    SasRec Embedding:
        1. ItemEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information

    Link: https://arxiv.org/pdf/1808.09781.pdf
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        schema: TensorSchema,
        embed_size: int,
        padding_idx: int,
        max_len: int,
        dropout: float,
    ) -> None:
        """
        :param schema Tensor schema of features.
        :param embed_size: Embedding size.
        :param padding_idx: Padding index.
        :param max_len: Max length of sequence.
        :param dropout: Dropout rate.
            Default: ``0.1``.
        """
        super().__init__()
        item_count = schema.item_id_features.item().cardinality
        assert item_count

        self.item_emb = torch.nn.Embedding(item_count + 1, embed_size, padding_idx=padding_idx)
        self.pos_emb = SasRecPositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.item_emb_dropout = torch.nn.Dropout(p=dropout)

        assert schema.item_id_feature_name
        self.item_feature_name = schema.item_id_feature_name

    def forward(self, feature_tensor: TensorMap, padding_mask: torch.BoolTensor) -> torch.Tensor:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Embeddings for input features.
        """
        seqs = self.item_emb(feature_tensor[self.item_feature_name]) * (self.item_emb.embedding_dim**0.5)
        seqs += self.pos_emb(seqs)
        seqs = self.item_emb_dropout(seqs)
        seqs *= padding_mask
        return seqs

    def get_item_weights(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        :param indices: Items indices.

        :returns: Item weights for specific items.
        """
        return self.item_emb(indices)

    def get_all_item_weights(self) -> torch.Tensor:
        """
        :returns: Item weights for all items.
        """
        # Last one is reserved for padding, so we remove it
        return self.item_emb.weight[:-1, :]

    def get_all_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        :returns: copy of all embeddings presented in this layer as a dict.
        """
        return {
            "item_embedding": self.item_emb.weight.data[:-1, :].detach().clone(),
            "positional_embedding": self.pos_emb.pe.weight.data.detach().clone(),
        }


class SasRecLayers(torch.nn.Module):
    """
    SasRec vanilla layers:
        1. SelfAttention layers
        2. FeedForward layers

    Link: https://arxiv.org/pdf/1808.09781.pdf
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        """
        :param hidden_size: Hidden size of transformer.
        :param num_heads: Number of Attention heads.
        :param num_blocks: Number of Transformer blocks.
        :param dropout: Dropout rate.
        """
        super().__init__()
        self.attention_layers = self._layers_stacker(
            torch.nn.MultiheadAttention(hidden_size, num_heads, dropout), num_blocks
        )
        self.attention_layernorms = self._layers_stacker(torch.nn.LayerNorm(hidden_size, eps=1e-8), num_blocks)
        self.forward_layers = self._layers_stacker(SasRecPointWiseFeedForward(hidden_size, dropout), num_blocks)
        self.forward_layernorms = self._layers_stacker(torch.nn.LayerNorm(hidden_size, eps=1e-8), num_blocks)

    def forward(
        self,
        seqs: torch.Tensor,
        attention_mask: torch.BoolTensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        :param seqs: Item embeddings.
        :param attention_mask: Attention mask.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Output embeddings.
        """
        length = len(self.attention_layers)
        for i in range(length):
            seqs = torch.transpose(seqs, 0, 1)
            query = self.attention_layernorms[i](seqs)
            attent_emb, _ = self.attention_layers[i](query, seqs, seqs, attn_mask=attention_mask)
            seqs = query + attent_emb
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= padding_mask

        return seqs

    def _layers_stacker(self, layer: Any, num_blocks: int) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([layer] * num_blocks)


class SasRecNormalizer(torch.nn.Module):
    """
    SasRec notmilization layers

    Link: https://arxiv.org/pdf/1808.09781.pdf
    """

    def __init__(
        self,
        hidden_size: int,
    ) -> None:
        """
        :param hidden_size: Hidden size of transformer.
        """
        super().__init__()
        self.last_layernorm = torch.nn.LayerNorm(hidden_size, eps=1e-8)

    def forward(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        :param seqs: Input embeddings.

        :returns: Normalized embeddings.
        """
        output_emb = self.last_layernorm(seqs)

        return output_emb


class SasRecPointWiseFeedForward(torch.nn.Module):
    """
    Point wise feed forward layers

    Link: https://arxiv.org/pdf/1808.09781.pdf
    """

    def __init__(self, hidden_units: int, dropout: float) -> None:
        """
        :param hidden_units: Hidden size.
        :param dropout: Dropout rate.
        """
        super().__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout)

    def forward(self, inputs: torch.LongTensor) -> torch.LongTensor:
        """
        :param inputs: Query feature vector.

        :returns: Output tensors.
        """
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs

        return outputs


class SasRecPositionalEmbedding(torch.nn.Module):
    """
    Positional embedding.
    """

    # pylint: disable=invalid-name
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


class TiSasRecEmbeddings(torch.nn.Module, BaseSasRecEmbeddings):
    """
    TiSasRec Embedding:
        1. ItemEmbedding : normal embedding matrix
        2. TimeRelativeEmbedding: based on TiSasRec architecture
        3. TimeRelativePositionalEmbedding: based on TiSasRec architecture

    Link: https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        schema: TensorSchema,
        embed_size: int,
        padding_idx: int,
        max_len: int,
        time_span: int,
        dropout: float,
    ) -> None:
        """
        :param schema: Tensor schema of features.
        :param embed_size: Embedding size.
        :param padding_idx: Padding index.
        :param max_len: Max length of sequence.
        :param time_span: Time span value.
        :param dropout: Dropout rate.
        """
        super().__init__()
        self.time_span = time_span

        item_count = schema.item_id_features.item().cardinality
        assert item_count

        self.item_emb = torch.nn.Embedding(item_count + 1, embed_size, padding_idx=padding_idx)
        self.abs_pos_k_emb = SasRecPositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.abs_pos_v_emb = SasRecPositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.time_matrix_k_emb = torch.nn.Embedding(time_span + 1, embed_size)
        self.time_matrix_v_emb = torch.nn.Embedding(time_span + 1, embed_size)

        self.item_emb_dropout = torch.nn.Dropout(p=dropout)
        self.abs_pos_k_emb_dropout = torch.nn.Dropout(p=dropout)
        self.abs_pos_v_emb_dropout = torch.nn.Dropout(p=dropout)
        self.time_matrix_k_dropout = torch.nn.Dropout(p=dropout)
        self.time_matrix_v_dropout = torch.nn.Dropout(p=dropout)

        assert schema.item_id_feature_name
        self.item_feature_name = schema.item_id_feature_name

        assert schema.timestamp_feature_name
        self.timestamp_feature_name = schema.timestamp_feature_name

    def forward(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Output embeddings, key and value time interval matrices, key and value positional embeddings.
        """
        time_matrix = self._time_relative_matrix(feature_tensor[self.timestamp_feature_name])

        seqs = self.item_emb(feature_tensor[self.item_feature_name]) * (self.item_emb.embedding_dim**0.5)
        abs_pos_k = self.abs_pos_k_emb(seqs)
        abs_pos_v = self.abs_pos_v_emb(seqs)
        time_matrix_k = self.time_matrix_k_emb(time_matrix)
        time_matrix_v = self.time_matrix_v_emb(time_matrix)

        seqs = self.item_emb_dropout(seqs)
        seqs *= padding_mask

        abs_pos_k = self.abs_pos_k_emb_dropout(abs_pos_k)
        abs_pos_v = self.abs_pos_v_emb_dropout(abs_pos_v)
        time_matrix_k = self.time_matrix_k_dropout(time_matrix_k)
        time_matrix_v = self.time_matrix_v_dropout(time_matrix_v)

        ti_embeddings = (time_matrix_k, time_matrix_v, abs_pos_k, abs_pos_v)

        return seqs, ti_embeddings

    def _time_relative_matrix(self, time_seq: torch.Tensor) -> torch.LongTensor:
        time_matrix = torch.abs(time_seq.unsqueeze(-1) - time_seq.unsqueeze(1))
        if time_matrix.dtype != torch.int64:
            time_matrix = torch.floor(time_matrix).long()
        time_matrix = time_matrix.masked_fill(time_matrix > self.time_span, self.time_span)
        return cast(torch.LongTensor, time_matrix)

    def get_item_weights(self, indices: torch.LongTensor) -> torch.Tensor:
        """
        :param indices: Items indices.

        :returns: Item weights for specific items.
        """
        return self.item_emb(indices)

    def get_all_item_weights(self) -> torch.Tensor:
        """
        :returns: Item weights for all items.
        """
        # Last one is reserved for padding, so we remove it
        return self.item_emb.weight[:-1, :]

    def get_all_embeddings(self) -> Dict[str, torch.Tensor]:
        """
        :returns: copy of all embeddings presented in this layer as a dict.
        """
        return {
            "item_embedding": self.item_emb.weight.data[:-1, :].detach().clone(),
            "abs_pos_k_emb": self.abs_pos_k_emb.pe.weight.data.detach().clone(),
            "abs_pos_v_emb": self.abs_pos_v_emb.pe.weight.data.detach().clone(),
            "time_matrix_k_emb": self.time_matrix_k_emb.weight.data.detach().clone(),
            "time_matrix_v_emb": self.time_matrix_v_emb.weight.data.detach().clone(),
        }


class TiSasRecLayers(torch.nn.Module):
    """
    TiSasRec layers:
        1. Time-relative SelfAttention layers
        2. FeedForward layers

    Link: https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        """
        :param hidden_size: Hidden size of transformer.
        :param num_heads: Number of Attention heads.
        :param num_blocks: Number of Transformer blocks.
        :param dropout: Dropout rate.
        """
        super().__init__()
        self.attention_layers = self._layers_stacker(TiSasRecAttention(hidden_size, num_heads, dropout), num_blocks)
        self.forward_layers = self._layers_stacker(SasRecPointWiseFeedForward(hidden_size, dropout), num_blocks)
        self.attention_layernorms = self._layers_stacker(torch.nn.LayerNorm(hidden_size, eps=1e-8), num_blocks)
        self.forward_layernorms = self._layers_stacker(torch.nn.LayerNorm(hidden_size, eps=1e-8), num_blocks)

    # pylint: disable=too-many-arguments
    def forward(
        self,
        seqs: torch.Tensor,
        attention_mask: torch.BoolTensor,
        padding_mask: torch.BoolTensor,
        ti_embeddings: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        :param seqs: Item embeddings.
        :param attention_mask: Attention mask.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.
        :param ti_embeddings: Output embeddings, key and value time interval matrices,
            key and value positional embeddings.
        :param device: Selected device.

        :returns: Output embeddings.
        """
        length = len(self.attention_layers)
        for i in range(length):
            query = self.attention_layernorms[i](seqs)
            attent_emb = self.attention_layers[i](query, seqs, ~padding_mask, attention_mask, ti_embeddings, device)
            seqs = query + attent_emb
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= padding_mask

        return seqs

    def _layers_stacker(self, layer: Any, num_blocks: int) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([layer] * num_blocks)


class TiSasRecAttention(torch.nn.Module):
    """
    Time interval aware multihead attention

    Link: https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf
    """

    def __init__(self, hidden_size: int, head_num: int, dropout_rate: float) -> None:
        """
        :param hidden_size: Hidden embedding size.
        :param head_num: Head numbers.
        :param dropout_rate: Dropout rate.
        """
        super().__init__()
        self.query_w = torch.nn.Linear(hidden_size, hidden_size)
        self.key_w = torch.nn.Linear(hidden_size, hidden_size)
        self.value_w = torch.nn.Linear(hidden_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.softmax = torch.nn.Softmax(dim=-1)

        self.hidden_size = hidden_size
        self.head_num = head_num
        self.head_size = hidden_size // head_num
        self.dropout_rate = dropout_rate

    # pylint: disable=too-many-arguments, invalid-name, too-many-locals
    def forward(
        self,
        queries: torch.LongTensor,
        keys: torch.LongTensor,
        time_mask: torch.LongTensor,
        attn_mask: torch.LongTensor,
        ti_embeddings: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor, torch.LongTensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        :param queries: Queries feature vector.
        :param keys: Keys feature vector.
        :param time_mask: Time interval mask.
        :param attn_mask: Attention mask.
        :param ti_embeddings: Output embeddings, key and value time interval matrices,
            key and value positional embeddings..
        :param device: Selected device.

        :returns: Attention outputs.
        """
        time_matrix_k, time_matrix_v, abs_pos_k, abs_pos_v = ti_embeddings
        query, key, value = self.query_w(queries), self.key_w(keys), self.value_w(keys)

        # head dim * batch dim for parallelization (h*N, T, C/h)
        q_ = torch.cat(torch.split(query, self.head_size, dim=2), dim=0)
        k_ = torch.cat(torch.split(key, self.head_size, dim=2), dim=0)
        v_ = torch.cat(torch.split(value, self.head_size, dim=2), dim=0)

        time_matrix_k_ = torch.cat(torch.split(time_matrix_k, self.head_size, dim=3), dim=0)
        time_matrix_v_ = torch.cat(torch.split(time_matrix_v, self.head_size, dim=3), dim=0)
        abs_pos_k_ = torch.cat(torch.split(abs_pos_k, self.head_size, dim=2), dim=0)
        abs_pos_v_ = torch.cat(torch.split(abs_pos_v, self.head_size, dim=2), dim=0)

        # batched channel wise matmul to gen attention weights
        attn_weights = q_.matmul(torch.transpose(k_, 1, 2))
        attn_weights += q_.matmul(torch.transpose(abs_pos_k_, 1, 2))
        attn_weights += time_matrix_k_.matmul(q_.unsqueeze(-1)).squeeze(-1)

        # seq length adaptive scaling
        attn_weights = attn_weights / (k_.shape[-1] ** 0.5)

        t_mask = time_mask.repeat(self.head_num, 1, 1)
        t_mask = t_mask.expand(-1, -1, attn_weights.shape[-1])
        a_mask = attn_mask.unsqueeze(0).expand(attn_weights.shape[0], -1, -1)
        paddings = torch.ones(attn_weights.shape) * (-(2**32) + 1)  # -1e23 # float('-inf')
        paddings = paddings.to(device)
        attn_weights = torch.where(t_mask, paddings, attn_weights)
        attn_weights = torch.where(a_mask, paddings, attn_weights)

        attn_weights = self.softmax(attn_weights)
        attn_weights = self.dropout(attn_weights)

        outputs = attn_weights.matmul(v_)
        outputs += attn_weights.matmul(abs_pos_v_)
        outputs += attn_weights.unsqueeze(2).matmul(time_matrix_v_).reshape(outputs.shape).squeeze(2)

        outputs = torch.cat(torch.split(outputs, query.shape[0], dim=0), dim=2)

        return cast(torch.LongTensor, outputs)
