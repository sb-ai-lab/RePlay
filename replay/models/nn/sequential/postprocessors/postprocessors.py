from typing import List, Optional, Set, Tuple, cast

import numpy as np
import pandas as pd
import torch

from replay.data.nn import SequentialDataset

from ._base import BasePostProcessor


class RemoveSeenItems(BasePostProcessor):
    """
    Filters out the items that already have been seen in dataset.
    """

    def __init__(self, sequential: SequentialDataset) -> None:
        super().__init__()
        self._sequential = sequential

    def on_validation(
        self, query_ids: torch.LongTensor, scores: torch.Tensor, ground_truth: torch.LongTensor
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        """
        Validation step.

        :param query_ids: query id sequence
        :param scores: calculated logits
        :param ground_truth: ground truth dataset

        :returns: modified query ids and scores and ground truth dataset
        """
        modified_scores = self._compute_scores(query_ids, scores)
        return query_ids, modified_scores, ground_truth

    def on_prediction(self, query_ids: torch.LongTensor, scores: torch.Tensor) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Prediction step.

        :param query_ids: query id sequence
        :param scores: calculated logits

        :returns: modified query ids and scores
        """
        modified_scores = self._compute_scores(query_ids, scores)
        return query_ids, modified_scores

    def _compute_scores(self, query_ids: torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
        flat_seen_item_ids = self._get_flat_seen_item_ids(query_ids)
        return self._fill_item_ids(scores, flat_seen_item_ids, -np.inf)

    def _fill_item_ids(
        self,
        scores: torch.Tensor,
        flat_item_ids: torch.LongTensor,
        value: float,
    ) -> torch.Tensor:
        flat_item_ids_on_device = flat_item_ids.to(scores.device)
        if scores.is_contiguous():
            scores.view(-1)[flat_item_ids_on_device] = value
        else:
            flat_scores = scores.flatten()
            flat_scores[flat_item_ids_on_device] = value
            scores = flat_scores.reshape(scores.shape)
        return scores

    def _get_flat_seen_item_ids(self, query_ids: torch.LongTensor) -> torch.LongTensor:
        query_ids_np = query_ids.flatten().cpu().numpy()

        item_count = self._sequential.schema.item_id_features.item().cardinality
        assert item_count
        item_id_feature_name = self._sequential.schema.item_id_feature_name
        assert item_id_feature_name

        item_id_sequences = self._sequential.get_sequence_by_query_id(query_ids_np, item_id_feature_name)

        for i in range(item_id_sequences.shape[0]):
            item_id_sequences[i] = item_id_sequences[i].copy() + i * item_count

        flat_seen_item_ids_np = np.concatenate(item_id_sequences)
        return torch.LongTensor(flat_seen_item_ids_np)


class SampleItems(BasePostProcessor):
    """
    Generates negative samples to compute sampled metrics
    """

    def __init__(
        self,
        grouped_validation_items: pd.DataFrame,
        user_col: str,
        item_col: str,
        items_list: np.ndarray,
        sample_count: int,
    ) -> None:
        self.items_set = set(items_list)
        self.sample_count = sample_count
        users = grouped_validation_items[user_col].to_numpy()
        items = grouped_validation_items[item_col].to_numpy()
        self.items_list: List[Set[int]] = [set() for _ in range(users.shape[0])]
        for i in range(users.shape[0]):
            self.items_list[users[i]] = set(items[i])

    def on_validation(
        self, query_ids: torch.LongTensor, scores: torch.Tensor, ground_truth: torch.LongTensor
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        """
        Validation step.

        :param query_ids: query id sequence
        :param scores: calculated logits
        :param ground_truth: ground truth dataset

        :returns: modified query ids and scores and ground truth dataset
        """
        modified_score = self._compute_score(query_ids, scores, ground_truth)
        return query_ids, modified_score, ground_truth

    def on_prediction(self, query_ids: torch.LongTensor, scores: torch.Tensor) -> Tuple[torch.LongTensor, torch.Tensor]:
        """
        Prediction step.

        :param query_ids: query id sequence
        :param scores: calculated logits

        :returns: modified query ids and scores
        """
        modified_score = self._compute_score(query_ids, scores, None)
        return query_ids, modified_score

    def _compute_score(
        self, query_ids: torch.LongTensor, scores: torch.Tensor, ground_truth: Optional[torch.LongTensor]
    ) -> torch.Tensor:
        batch_size = query_ids.shape[0]
        item_ids = ground_truth.cpu().numpy() if ground_truth is not None else None
        candidate_ids: List[torch.Tensor] = []
        candidate_labels: List[torch.Tensor] = []
        for user in range(batch_size):
            ground_truth_items = set(item_ids[user]) if ground_truth is not None else set()
            sample, label = self._generate_samples_for_user(ground_truth_items, self.items_list[user])
            candidate_ids.append(sample)
            candidate_labels.append(label)

        candidate_ids_torch = cast(torch.LongTensor, torch.stack(candidate_ids).to(scores.device).long())

        return self._fill_scores(candidate_ids_torch, scores, np.inf)

    def _fill_scores(self, ids: torch.LongTensor, scores: torch.Tensor, value: float) -> torch.Tensor:
        row_count = ids.shape[0]
        item_count = scores.shape[1]
        stride = torch.arange(0, row_count * item_count, item_count).to(scores.device).reshape(-1, 1)
        strided_ids = ids + stride
        new_scores = scores.flatten()
        new_scores[strided_ids.flatten()] = value

        return new_scores.reshape_as(scores)

    def _generate_samples_for_user(
        self, ground_truth_items: Set[int], input_items: Set[int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        negative_sample_count = self.sample_count - len(ground_truth_items)
        assert negative_sample_count > 0

        # Negative samples in the original paper excludes the input sequences.
        random_items = np.random.choice(list(self.items_set - input_items - ground_truth_items), negative_sample_count)
        samples_list = list(random_items) + list(ground_truth_items)

        samples = torch.tensor(samples_list, dtype=torch.long)
        labels = torch.cat(
            [
                torch.zeros(negative_sample_count, dtype=torch.bool),
                torch.ones(len(ground_truth_items), dtype=torch.bool),
            ]
        )

        return (
            samples,
            labels,
        )
