import abc

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
