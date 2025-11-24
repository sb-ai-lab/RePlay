import torch


class EmbeddingTyingHead(torch.nn.Module):
    """
    Head that calculate logits for all items given output embeddings
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        item_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        🚨🚨🚨
        Please don't look for ways to combine all these cases
        into one and write a universal method.

        The implementation is written in such a way to get maximum performance.
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
        # out_embeddings shape: [B, E] or [B, P, E]
        # item embeddings shape [B, I, E]
        # [*, 1, E] x [*, E, 1] -> [B, *, I]
        return torch.bmm(
            hidden_states.view(-1, 1, hidden_states.size(-1)), item_embeddings.view(-1, item_embeddings.size(-1), 1)
        ).view(hidden_states.size(0), *item_embeddings.shape[1:-1])
