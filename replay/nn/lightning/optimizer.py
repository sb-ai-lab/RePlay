import abc
from collections.abc import Iterator
from typing import Literal

import torch


class BaseOptimizerFactory(abc.ABC):
    """
    Interface for optimizer factory
    """

    @abc.abstractmethod
    def create(self, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:  # pragma: no cover
        """
        Creates optimizer based on parameters.

        :param parameters: torch parameters to initialize optimizer

        :returns: torch optimizer
        """


class OptimizerFactory(BaseOptimizerFactory):
    """
    Factory that creates optimizer depending on passed parameters
    """

    def __init__(
        self,
        optimizer: Literal["adam", "sgd"] = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        sgd_momentum: float = 0.0,
        betas: tuple[float, float] = (0.9, 0.98),
    ) -> None:
        super().__init__()
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sgd_momentum = sgd_momentum
        self.betas = betas

    def create(self, parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
        """
        Creates optimizer based on parameters.

        :param parameters: torch parameters to initialize optimizer

        :returns: torch optimizer
        """
        if self.optimizer == "adam":
            return torch.optim.Adam(parameters, lr=self.learning_rate, weight_decay=self.weight_decay, betas=self.betas)
        if self.optimizer == "sgd":
            return torch.optim.SGD(
                parameters, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.sgd_momentum
            )

        msg = "Unexpected optimizer"
        raise ValueError(msg)
