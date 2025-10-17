import abc

import torch


class BasePostProcessor(abc.ABC):  # pragma: no cover
    """
    Abstract base class for post processor
    """

    @abc.abstractmethod
    def on_prediction(self, query_ids: torch.LongTensor, scores: torch.Tensor) -> tuple[torch.LongTensor, torch.Tensor]:
        """
        Prediction step.

        :param query_ids: query id sequence
        :param scores: calculated logits

        :returns: modified query ids and scores
        """
        # it is necessary to return the tuple of modified query_ids and scores

    @abc.abstractmethod
    def on_validation(
        self, query_ids: torch.LongTensor, scores: torch.Tensor, ground_truth: torch.LongTensor
    ) -> tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
        """
        Validation step.

        :param query_ids: query id sequence
        :param scores: calculated logits
        :param ground_truth: ground truth dataset

        :returns: modified query ids and scores and ground truth dataset
        """
        # it is necessary to return the tuple of modified query_ids, scores and ground_truth
