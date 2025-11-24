import itertools
from typing import Any, Literal, Optional

import numpy as np
import torch

NegativeSamplingStrategy = Literal["global_uniform", "inbatch", "custom_weights"]


class SequentialNegativeSampler:
    def __init__(
        self,
        vocab_size: int,
        item_id_feature_name: str,
        negative_sampling_strategy: NegativeSamplingStrategy,
        num_negative_samples: int,
        sample_distribution: Optional[torch.Tensor] = None,
        feature_tensors_name: str = "feature_tensors",
        padding_mask_name: str = "padding_mask",
    ):
        """Sampler for negative items

        :param vocab_size: Number of items in vocabulary.
        :param item_id_feature_name: Name of feature in batch that corresponds to item ids sequences.
        :param negative_sampling_strategy: Type of sampling strategy.
        :param num_negative_samples: Number of negative item ids to get.
        :param sample_distribution: Distribution of weights for samples.
            The parameter can only be set if `negative_sampling_strategy` is equal to "custom_weights".
            Otherwise, you will get an exception.
            Default: ``None``.
        :param feature_tensors_name: The name of the key inside the batch.
            The key must contain the features that are given into the model.
            There should also be a feature called `item_id_feature_name`.
            Note: This parameter is needed only if the batch is a dictionary.
            Default: ``feature_tensors``.
        :param padding_mask_name: The name of the key inside the batch.
            The key must contain the padding mask that are given into the model.
            Note: This parameter is needed only if the batch is a dictionary.
            Default: ``padding_mask``.
        """

        assert negative_sampling_strategy in ["global_uniform", "inbatch", "custom_weights"]

        if negative_sampling_strategy == "custom_weights" and sample_distribution is None:
            msg = (
                "`negative_sampling_strategy` is equal to `custom_weights`, "
                "but the distribution of weights for samples is not set"
            )
            raise ValueError(msg)

        if negative_sampling_strategy != "custom_weights" and sample_distribution is not None:
            msg = (
                "The distribution of weights for samples is set, "
                f"but negative_sampling_strategy is equal to {negative_sampling_strategy}. "
                "If you really want to use a custom distribution of weights for samples "
                "then set `negative_sampling_strategy` to `custom_weights`."
            )
            raise ValueError(msg)

        if sample_distribution is not None and sample_distribution.size(0) != vocab_size:
            msg = (
                "The sample_distribution parameter has an incorrect size. "
                f"Got {sample_distribution.size(0)}, expected {vocab_size}."
            )
            raise ValueError(msg)

        self.item_id_feature_name = item_id_feature_name
        self.feature_tensors_name = feature_tensors_name
        self.padding_mask_name = padding_mask_name
        self.negative_sampling_strategy = negative_sampling_strategy
        self.num_negative_samples = num_negative_samples

        # Uniform sampling for negatives, PMI will be added later
        self.multinomial_sample_distribution: torch.Tensor
        if sample_distribution is not None:
            self.multinomial_sample_distribution = sample_distribution
        else:
            self.multinomial_sample_distribution = torch.ones(vocab_size)

    def get_uniform_negatives(self):
        """Get single vector of negatives for all samples in batch"""

        # [num_negatives] - shape of negatives
        negatives = torch.multinomial(
            self.multinomial_sample_distribution,
            num_samples=self.num_negative_samples,
            replacement=False,
        )

        return negatives

    def get_inbatch_negatives(self, batch: Any) -> torch.Tensor:
        """Get different vectors of negatives for samples in batch.

        Important points:
            - Repetitions are possible, without repetition sampling is slower
            - For extremely low number of samples in batch (e.g. 2) it's possible to
              have no candidates which results in exception. User may have only 1 event
              that disappers after right shift in Dataset
        """

        # [batch_size, seq_len] - tensor with input item_id's
        batch_positives = (
            batch[self.feature_tensors_name][self.item_id_feature_name]
            if isinstance(batch, dict)
            else batch.features[self.item_id_feature_name]  # for bc
        )

        # [batch_size, seq_len] - tensor with mask for input item_id's
        padding_mask = batch[self.padding_mask_name] if isinstance(batch, dict) else batch.padding_mask  # for bc

        # All item_id's as flat tensor
        batch_positives = batch_positives[padding_mask]

        # List with count of non-padding events in each sample in batch
        batch_session_len = torch.sum(padding_mask, dim=1).tolist()

        # Accumulate count so we can select subsets of batch_positives later
        positive_indices = itertools.accumulate(batch_session_len)

        # To correctly select subsets of batch_positives later
        positive_indices = [0, *list(positive_indices)]

        # Sampled negative item_id's
        negatives = []

        # For each sample in batch
        for i in range(len(positive_indices[:-1])):
            # Get item_id's not from current sample in batch [~1ms]
            candidates = torch.cat([batch_positives[: positive_indices[i]], batch_positives[positive_indices[i + 1] :]])

            # Sample indices of items from candidates, fastest if repetitions are possible [~2 ms]
            indices = np.random.randint(len(candidates), size=self.num_negative_samples)

            # Get tensor of negative item_id's from candidates [~3 ms]
            candidates = candidates[indices]

            # Add tensor to list of negatives
            negatives.append(candidates)

        # Create output tensor [batch_size, num_negatives]
        negatives = torch.stack(negatives)

        return negatives

    def get_negatives(self, batch: Any) -> torch.Tensor:
        """Get negatives based on configured strategy"""

        # Custom weights aren't supported yet, global uniform will be used instead
        if self.negative_sampling_strategy in ["global_uniform", "custom_weights"]:
            return self.get_uniform_negatives()

        elif self.negative_sampling_strategy == "inbatch":
            return self.get_inbatch_negatives(batch=batch)

        else:
            msg = f"Negative sampling strategy isn't supported: {self.negative_sampling_strategy}"
            raise ValueError(msg)
