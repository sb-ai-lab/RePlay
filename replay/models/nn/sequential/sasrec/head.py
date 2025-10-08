from typing import Optional, Protocol

import torch

from .embedding import SasRecEmbeddingProtocol


class SasRecHeadProtocol(Protocol):
    def forward(
        self,
        out_embeddings: torch.Tensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor: ...

    def reset_parameters(self) -> None: ...

class EmbeddingTyingHead(torch.nn.Module, SasRecHeadProtocol):
    """
    Head that calculate logits for all item_ids given output embeddings
    """

    def __init__(self, item_embedder: SasRecEmbeddingProtocol):
        """
        :param item_embedder: SasRec embedding.
        """
        super().__init__()
        self._item_embedder = item_embedder

    def reset_parameters(self) -> None:
        """
        It makes no sense to change the initialization in `item_embedder` here,
        because it is used here to get weights and it is initialized elsewhere.
        """
        return

    def forward(
        self,
        out_embeddings: torch.Tensor,
        candidates_to_score: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """
        :param out_embeddings: Embeddings after `forward step`.
            Possible shapes:
                - (batch_size, sequence_length, embedding_dim)
                - (batch_size, embedding_dim)
        :param candidates_to_score: Item ids to calculate scores.
            Default: ``None``, means calculating scores for all items.

        :returns: Calculated logits.
            Possible shapes that depend on the incoming parameters:
                - (batch_size, sequence_length, item_count)
                - (batch_size, item_count)
        """
        item_embeddings = self._item_embedder.get_item_weights(candidates_to_score)
        assert item_embeddings.size() == 2
        return torch.matmul(out_embeddings, item_embeddings.t())
