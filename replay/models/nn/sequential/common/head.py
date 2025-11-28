import torch


class EmbeddingTyingHead(torch.nn.Module):
    """
    The model head for calculating the output logits as a dot product
    between the model hidden state and the item embeddings.
    The module supports both 2-d and 3-d tensors for the hidden state and the item embeddings.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param hidden_states: hidden state of shape
            (``batch_size``, ``embedding_dim``) or (``batch_size``, ``sequence_length``, ``embedding_dim``).
        :param item_embeddings: item embeddings of shape
            (``num_items``, ``embedding_dim``) or (``batch_size``, ``num_items``, ``embedding_dim``).
        :return: logits of shape (``batch_size``, ``num_items``)
            or (``batch_size``, ``sequence_length``, ``num_items``).
        """
        if item_embeddings.dim() == 2:
            item_embeddings = item_embeddings.transpose(-1, -2).contiguous()
            # hidden_states shape [B, *, E]
            # item embeddings shape [I, E]
            # [B, *, E] x [E, I] -> [B, *, I]
            return hidden_states.matmul(item_embeddings)
        elif item_embeddings.dim() == 3 and hidden_states.dim() == 2:
            item_embeddings = item_embeddings.transpose(-1, -2).contiguous()
            # out_embeddings shape [B, E]
            # item embeddings shape [B, I, E]
            # [B, E] x [B, E, I] -> [B, I]
            hidden_states = hidden_states.unsqueeze(-2)
            logits = hidden_states.matmul(item_embeddings)
            return logits.squeeze(-2)
        # out_embeddings shape: [B, *, E]
        # item embeddings shape [B, *, E]
        # [*, 1, E] x [*, E, 1] -> [B, *]
        return torch.bmm(
            hidden_states.view(-1, 1, hidden_states.size(-1)), item_embeddings.view(-1, item_embeddings.size(-1), 1)
        ).view(hidden_states.size(0), *item_embeddings.shape[1:-1])
