import abc

import torch


class BasePostProcessor(abc.ABC):  # pragma: no cover
    """
    Abstract base class for post processor
    """

    @abc.abstractmethod
    def __call__(self, batch: dict, logits: torch.Tensor) -> torch.Tensor:
        """
        :param batch: the batch sent to the model from the dataloader
        :param logits: logits from the model

        :returns: modified logits
        """
