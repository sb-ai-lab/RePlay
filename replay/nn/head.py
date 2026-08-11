import torch


class EmbeddingTyingHead(torch.nn.Module):
    """
    The model head for calculating the output logits as a dot product
    between the model hidden state and the item embeddings.
    The module supports both 2-d and 3-d tensors for the hidden state and the item embeddings.

    As a result of the work, the scores for each item will be obtained.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param hidden_states: a hidden state of shape
            ``(batch_size, embedding_dim)`` or ``(batch_size, sequence_length, embedding_dim)``.
        :param item_embeddings: item embeddings of shape
            ``(num_items, embedding_dim)`` or ``(batch_size, num_items, embedding_dim)``.
        :return: logits of shape ``(batch_size, num_items)``
            or ``(batch_size, sequence_length, num_items)``.
        """
        if item_embeddings.dim() == 2:
            # [B, *, E] x [I, E] -> [B, *, I]
            return torch.nn.functional.linear(hidden_states, item_embeddings)
        if item_embeddings.dim() == 3 and hidden_states.dim() == 2:
            # [B, I, E] x [B, E, 1] -> [B, I]
            return torch.bmm(item_embeddings, hidden_states.unsqueeze(-1)).squeeze(-1)
        # Pairwise dot product for tensors with matching leading dimensions.
        return torch.bmm(
            hidden_states.view(-1, 1, hidden_states.size(-1)),
            item_embeddings.view(-1, item_embeddings.size(-1), 1),
        ).view(hidden_states.size(0), *item_embeddings.shape[1:-1])
