import abc
from typing import Optional, Union

import torch


class PostprocessorBase(abc.ABC):  # pragma: no cover
    """
    Abstract base class for post processor
    """

    @abc.abstractmethod
    def on_validation(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        """
        The method is called externally inside the callback at the validation stage.

        :param batch: the batch sent to the model from the dataloader
        :param logits: logits from the model

        :returns: modified logits
        """

    @abc.abstractmethod
    def on_prediction(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        """
        The method is called externally inside the callback at the prediction (inference) stage.

        :param batch: the batch sent to the model from the dataloader
        :param logits: logits from the model

        :returns: modified logits
        """

    @property
    def candidates(self) -> Union[torch.LongTensor, None]:
        """
        Returns tensor of item ids to calculate scores.
        """
        return self._candidates

    @candidates.setter
    def candidates(self, candidates: Optional[torch.LongTensor] = None) -> None:
        """
        Sets tensor of item ids to calculate scores.
        :param candidates: Tensor of item ids to calculate scores.
        """
        if (candidates is not None) and (torch.unique(candidates).numel() != candidates.numel()):
            msg = "The tensor of candidates to score must be unique."
            raise ValueError(msg)

        self._candidates = candidates
