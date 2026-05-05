from dataclasses import dataclass
from typing import Literal

import torch

from .ffn import PointWiseFeedForward


@dataclass(frozen=True)
class TopKMixtureOfExpertsConfig:
    """
    Configuration for Top-K Mixture-of-Experts feed-forward network.
    """

    num_experts: int = 8
    k: int = 1

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            msg = "Parameter ``num_experts`` must be positive."
            raise ValueError(msg)
        if self.k <= 0:
            msg = "Parameter ``k`` must be positive."
            raise ValueError(msg)
        if self.k > self.num_experts:
            msg = "Parameter ``k`` must be less than or equal to ``num_experts``."
            raise ValueError(msg)


class TopKMixtureOfExpertsBlock(torch.nn.Module):
    """
    Sparse Top-K Mixture-of-Experts wrapper over point-wise feed-forward experts.
    """

    def __init__(
        self,
        config: TopKMixtureOfExpertsConfig,
        embedding_dim: int,
        dropout: float,
        activation: Literal["relu", "gelu"] = "gelu",
    ) -> None:
        """
        :param config: top-K Mixture-of-Experts configuration.
        :param embedding_dim: Dimension of the input features.
        :param dropout: probability of an element to be zeroed.
        :param activation: the name of the activation function.
            Default: ``"gelu"``.
        """
        super().__init__()

        self.config = config
        self.router = torch.nn.Linear(embedding_dim, config.num_experts, bias=False)
        self.experts = torch.nn.ModuleList(
            [
                PointWiseFeedForward(
                    embedding_dim=embedding_dim,
                    dropout=dropout,
                    activation=activation,
                    residual=False,
                )
                for _ in range(config.num_experts)
            ]
        )

    def reset_parameters(self) -> None:
        self.router.reset_parameters()

        for expert in self.experts:
            expert.reset_parameters()

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        """
        :param input_embeddings: An input tensor of shape ``(batch_size, sequence_length, embedding_dim)``.

        :returns: An output tensor of shape ``(batch_size, sequence_length, embedding_dim)``.
        """
        batch_size, sequence_length, embedding_dim = input_embeddings.shape
        tokens = input_embeddings.reshape(batch_size * sequence_length, embedding_dim)

        # NOTE: compute router probs in float32 for numerical stability.
        router_logits = torch.nn.functional.linear(
            input=tokens.float(),
            weight=self.router.weight.float(),
            bias=None if self.router.bias is None else self.router.bias.float(),
        )

        router_probs = torch.softmax(router_logits, dim=-1)
        route_weights, route_experts = torch.topk(router_probs, k=self.config.k, dim=-1)

        if self.config.k > 1:
            route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        route_weights = route_weights.to(tokens.dtype)

        flat_route_experts = route_experts.reshape(-1)
        flat_route_weights = route_weights.reshape(-1)

        flat_token_indices = torch.arange(tokens.size(0), device=tokens.device).repeat_interleave(self.config.k)

        output_tokens = torch.zeros_like(tokens)

        for expert_idx, expert in enumerate(self.experts):
            route_indices = torch.nonzero(flat_route_experts == expert_idx, as_tuple=False).squeeze(-1)

            token_indices = flat_token_indices.index_select(0, route_indices)
            expert_inputs = tokens.index_select(0, token_indices)

            expert_outputs = expert(expert_inputs.unsqueeze(1)).squeeze(1)

            expert_weights = flat_route_weights.index_select(0, route_indices).unsqueeze(-1)
            weighted_outputs = expert_outputs * expert_weights

            output_tokens = output_tokens.index_add(0, token_indices, weighted_outputs)

        return (tokens + output_tokens).reshape(batch_size, sequence_length, embedding_dim)
