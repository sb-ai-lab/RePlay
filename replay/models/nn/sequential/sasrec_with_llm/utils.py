import torch
import json

from torch import nn
import torch.nn.functional as F


def load_user_profile_embeddings(file_path: str, user_id_mapping: dict):
    """
    Load user profile embeddings from a single JSON file
    :param file_path: Path to the JSON file
    :param user_id_mapping: Mapping from user IDs to indices

    :return: Tuple of user profile embeddings tensor and binary mask for missing profiles
    """
    with open(file_path, 'r') as f:
        user_profiles_data = json.load(f)

    embedding_dim = len(next(iter(user_profiles_data.values())))
    max_idx = max(user_id_mapping.values()) + 1  # Ensure list can accommodate the highest index
    user_profiles_list = [[0.0] * embedding_dim for _ in range(max_idx)]
    existing_profile_binary_mask = [True for _ in range(max_idx)]

    for original_id, idx in user_id_mapping.items():
        embedding = user_profiles_data.get(str(original_id))
        if embedding is not None:
            user_profiles_list[idx] = embedding
        else:
            existing_profile_binary_mask[idx] = False

    user_profiles_tensor = torch.tensor(user_profiles_list, dtype=torch.float)
    existing_profile_binary_mask_tensor = torch.BoolTensor(existing_profile_binary_mask)
    return user_profiles_tensor, existing_profile_binary_mask_tensor


def mean_weightening(hidden_states: torch.Tensor) -> torch.Tensor:
    """
    :param hidden_states: [batch_size, seq_len, hidden_size]
    :return: [batch_size, hidden_size]
    """
    return hidden_states.mean(dim=1)


def exponential_weightening(hidden_states: torch.Tensor, weight_scale: float) -> torch.Tensor:
    """
    :param hidden_states: hidden states of SASRec layers - [batch_size, seq_len, hidden_size]
    :param weight_scale: scale factor for the exponential weights
    """
    device = hidden_states.device

    indices = torch.arange(hidden_states.shape[1]).float().to(device)
    weights = torch.exp(weight_scale * indices)
    weights = weights / weights.sum()
    weights = weights.view(1, hidden_states.shape[1], 1)

    weighted_tensor = hidden_states * weights
    result = weighted_tensor.sum(dim=1)
    return result


class SimpleAttentionAggregator(nn.Module):
    def __init__(self, hidden_units: int) -> None:
        super(SimpleAttentionAggregator, self).__init__()
        self.attention = nn.Linear(hidden_units, 1)

    def forward(self, x):
        """
        :param x: Input tensor of shape [batch_size, n_items, hidden_units]
        :returns: Aggregated tensor of shape [batch_size, hidden_units]
        """
        scores = self.attention(x)
        weights = F.softmax(scores, dim=1)
        weighted_sum = (x * weights).sum(dim=1)
        return weighted_sum