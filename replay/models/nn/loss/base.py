from typing import Callable, Optional, Protocol, TypedDict

import torch

from replay.data.nn import TensorMap


class LossProto(Protocol):
    @property
    def logits_callback(self) -> Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]: ...

    @logits_callback.setter
    def logits_callback(self, func: Optional[Callable]) -> None: ...

    def forward(
        self,
        model_embeddings: torch.Tensor,
        feature_tensors: TensorMap,
        positive_labels: torch.LongTensor,
        negative_labels: torch.LongTensor,
        padding_mask: torch.BoolTensor,
        target_padding_mask: torch.BoolTensor,
    ) -> torch.Tensor: ...


class SampledLossOutput(TypedDict):
    positive_logits: torch.Tensor
    negative_logits: torch.Tensor
    positive_labels: torch.LongTensor
    negative_labels: torch.LongTensor


class SampledLossBase(torch.nn.Module):
    @property
    def logits_callback(self) -> Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        raise NotImplementedError()

    def get_sampled_logits(
        self,
        model_embeddings: torch.Tensor,
        positive_labels: torch.LongTensor,  # [batch_size, seq_len, num_positives]
        negative_labels: torch.LongTensor,  # [num_negatives] or [batch_size, seq_len, num_negatives]
        target_padding_mask: torch.BoolTensor,  # [batch_size, seq_len, num_positives]
    ) -> SampledLossOutput:
        initial_positive_labels = positive_labels
        ################## SHAPE CHECKING STAGE START ##################
        batch_size, seq_len, num_positives = positive_labels.size()
        assert target_padding_mask.size() == (batch_size, seq_len, num_positives)
        num_negatives = negative_labels.size(-1)
        if negative_labels.dim() == 3:
            # [batch_size, seq_len, num_negatives] -> [batch_size, seq_len, 1, num_negatives]
            negative_labels = negative_labels.unsqueeze(-2)
            if num_positives != 1:
                # [batch_size, seq_len, num_negatives] -> [batch_size, seq_len, num_positives, num_negatives]
                negative_labels = negative_labels.repeat((1, 1, num_positives, 1))
        assert (
            negative_labels.size() == (batch_size, seq_len, num_positives, num_negatives) or negative_labels.dim() == 1
        )
        ################## SHAPE CHECKING STAGE END ##################

        # Get output embedding for every user event
        embedding_dim = model_embeddings.size(-1)
        assert model_embeddings.size() == (batch_size, seq_len, embedding_dim)

        # [batch_size, seq_len, emb_dim] ->  [batch_size, seq_len, 1, emb_dim]
        model_embeddings = model_embeddings.unsqueeze(-2)
        if num_positives != 1:  # multti positive branch
            model_embeddings = model_embeddings.repeat((1, 1, num_positives, 1))
        assert model_embeddings.size() == (batch_size, seq_len, num_positives, embedding_dim)

        # Apply target mask
        # [batch_size, seq_len, num_positives] -> [batch_size, seq_len]
        masked_batch_size = target_padding_mask.sum().item()

        # [batch_size, seq_len, num_positives] -> [masked_batch_size, 1]
        positive_labels = positive_labels[target_padding_mask].unsqueeze(-1)
        assert positive_labels.size() == (masked_batch_size, 1)

        if negative_labels.dim() != 1:
            # [batch_size, seq_len, num_positives, num_negatives] -> [masked_batch_size, num_negatives]
            negative_labels = negative_labels[target_padding_mask]
            assert negative_labels.size() == (masked_batch_size, num_negatives)

        # [batch_size, seq_len, num_positives, emb_dim] -> [masked_batch_size, emb_dim]
        model_embeddings = model_embeddings[target_padding_mask]
        assert model_embeddings.size() == (masked_batch_size, embedding_dim)

        # Get positive and negative logits
        positive_logits = self.logits_callback(model_embeddings, positive_labels)
        assert positive_logits.size() == (masked_batch_size, 1)

        negative_logits = self.logits_callback(model_embeddings, negative_labels)
        assert negative_logits.size() == (masked_batch_size, num_negatives)

        if num_positives != 1:
            # [batch_size, seq_len, num_positives] -> [batch_size * seq_len]
            masked_target_padding_mask = target_padding_mask.sum(-1).view(-1)
            # [batch_size, seq_len, num_positives] -> [masked_batch_size, num_positives]
            positive_labels = torch.repeat_interleave(
                initial_positive_labels.view(-1, num_positives), masked_target_padding_mask, dim=0
            )

        return {
            "positive_logits": positive_logits,
            "negative_logits": negative_logits,
            "positive_labels": positive_labels,
            "negative_labels": negative_labels,
        }


def weight_loss_with_sample_weight(
    feature_tensors: TensorMap,
    target_padding_mask: torch.BoolTensor,
    loss: torch.Tensor,
    sample_weight_feature_name: Optional[str] = None,
) -> torch.FloatTensor:
    if sample_weight_feature_name is not None:
        sample_weight = feature_tensors[sample_weight_feature_name]
        sample_weight = sample_weight[target_padding_mask]
        weighted_loss = (loss * sample_weight).mean()
        return weighted_loss
    return loss


def mask_negative_logits(
    negative_logits: torch.Tensor,
    negative_labels: torch.LongTensor,
    positive_labels: torch.LongTensor,
    explicit_negatives_padding_value: Optional[int] = None,
) -> torch.Tensor:
    """Assign very low values to logits according to masking rules
    negative_logits: [masked_batch_size, num_negatives]
    negative_labels: [masked_batch_size, num_negatives] or [num_negatives]
    positive_labels: [masked_batch_size, num_positives]
    """

    if explicit_negatives_padding_value is not None:
        # [masked_batch_size, num_negatives] or [batch_size, seq_len, num_negatives] - get mask for padding values
        mask = (negative_labels == explicit_negatives_padding_value) * 1e9
        negative_logits -= mask

    if negative_labels.dim() > 1:  # explicit_negatives
        # [masked_batch_size, num_negatives] -> [masked_batch_size, 1, num_negatives]
        negative_labels = negative_labels.unsqueeze(-2)

    # [masked_batch_size, num_positives] -> [masked_batch_size, num_positives, 1]
    positive_labels = positive_labels.unsqueeze(-1)
    negative_mask = positive_labels == negative_labels  # [masked_batch_size, num_positives, num_negatives]

    # [masked_batch_size, num_positives, num_negatives] -> [masked_batch_size, num_negatives]
    negative_mask = negative_mask.sum(-2)
    negative_logits = negative_logits - 1e9 * negative_mask
    return negative_logits
