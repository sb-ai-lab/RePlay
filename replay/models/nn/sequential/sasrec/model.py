import contextlib
from typing import Any, Optional, Protocol

import torch

from replay.data.nn import TensorMap, TensorSchema
from replay.models.nn.ffn import PointWiseFeedForward

from .embedding import SasRecEmbedding, SasRecEmbeddingProtocol
from .head import EmbeddingTyingHead, SasRecHeadProtocol
from .mask_builder import SasRecAttentionMaskBuilder, SasRecAttentionMaskBuilderProtocol


class SasRecLayersProtocol(Protocol):
    def forward(
        self,
        seqs: torch.Tensor,
        attention_mask: torch.BoolTensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...

class SasRecNormalizerProtocol(Protocol):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...


class SasRecLayers(torch.nn.Module, SasRecLayersProtocol):
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
        self.num_blocks = num_blocks
        self.attention_layers = self._layers_stacker(
            num_blocks,
            torch.nn.MultiheadAttention,
            hidden_size,
            num_heads,
            dropout,
            batch_first=True,
        )
        self.attention_layernorms = self._layers_stacker(num_blocks, torch.nn.LayerNorm, hidden_size, eps=1e-8)
        self.forward_layers = self._layers_stacker(num_blocks, PointWiseFeedForward, hidden_size, dropout)
        self.forward_layernorms = self._layers_stacker(num_blocks, torch.nn.LayerNorm, hidden_size, eps=1e-8)

    def reset_parameters(self):
        for _, param in self.named_parameters():
            with contextlib.suppress(ValueError):
                torch.nn.init.xavier_normal_(param.data)

    def _layers_stacker(self, num_blocks: int, layer_class: Any, *args, **kwargs) -> torch.nn.ModuleList:
        return torch.nn.ModuleList([layer_class(*args, **kwargs) for _ in range(num_blocks)])

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
        for i in range(self.num_blocks):
            query = self.attention_layernorms[i](seqs)
            attent_emb, _ = self.attention_layers[i](
                query,
                seqs,
                seqs,
                key_padding_mask=padding_mask.logical_not(),
                attn_mask=attention_mask,
                need_weights=False,
            )
            seqs = query + attent_emb

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        return seqs


class SasRecModel(torch.nn.Module):
    """
    SasRec model
    """

    def __init__(
        self,
        attn_mask_builder: SasRecAttentionMaskBuilderProtocol,
        item_embedder: SasRecEmbeddingProtocol,
        layers: SasRecLayersProtocol,
        output_normalization: SasRecNormalizerProtocol,
        head: SasRecHeadProtocol,
    ) -> None:
        super().__init__()

        # Model blocks
        self.attn_mask_builder = attn_mask_builder

        self.item_embedder = item_embedder
        self.sasrec_layers = layers
        self.output_normalization = output_normalization
        self.head = head
        self.reset_parameters()

    def reset_parameters(self):
        self.item_embedder.reset_parameters()
        self.sasrec_layers.reset_parameters()
        self.output_normalization.reset_parameters()
        self.head.reset_parameters()

    def forward_step(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Output embeddings. Shape: (batch_size, sequence_length, embedding_dim)
        """
        seqs = self.item_embedder(feature_tensor)
        attention_mask = self.attn_mask_builder(feature_tensor, padding_mask)
        seqs = self.sasrec_layers(seqs, attention_mask, padding_mask)
        output_emb = self.output_normalization(seqs)

        return output_emb

    def forward(
        self,
        feature_tensor: TensorMap,
        padding_mask: torch.BoolTensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        :param feature_tensor: Batch of features.
        :param padding_mask: Padding mask where 0 - <PAD>, 1 otherwise.

        :returns: Calculated scores.
        """
        output_embeddings = self.forward_step(feature_tensor, padding_mask)
        if not self.training:
            output_embeddings = output_embeddings[:, -1, :]
            scores = self.head(output_embeddings, candidates_to_score)
        else:
            scores = self.head(output_embeddings)

        return scores

    @classmethod
    def build_default_model(
        cls,
        schema: TensorSchema,
        num_blocks: int = 2,
        num_heads: int = 1,
        hidden_size: int = 50,
        max_len: int = 200,
        dropout: float = 0.2,
    ):
        item_embedder=SasRecEmbedding(schema, max_len, dropout)
        return cls(
            attn_mask_builder=SasRecAttentionMaskBuilder(schema),
            item_embedder=item_embedder,
            layers=SasRecLayers(hidden_size, num_heads, num_blocks, dropout),
            output_normalization=torch.nn.LayerNorm(hidden_size, eps=1e-8),
            head=EmbeddingTyingHead(item_embedder),
        )


