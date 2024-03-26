import abc
from typing import Iterator, Tuple

import torch


class OptimizerFactory(abc.ABC):
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


class LRSchedulerFactory(abc.ABC):
    """
    Interface for learning rate scheduler factory
    """

    @abc.abstractmethod
    def create(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:  # pragma: no cover
        """
        Creates learning rate scheduler based on optimizer.

        :param optimizer: torch optimizer

        :returns: torch LRScheduler
        """


class FatOptimizerFactory(OptimizerFactory):
    """
    Factory that creates optimizer depending on passed parameters
    """

    def __init__(
        self,
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        sgd_momentum: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.98),
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


class FatLRSchedulerFactory(LRSchedulerFactory):
    """
    Factory that creates learning rate schedule depending on passed parameters
    """

    def __init__(
        self,
        decay_step: int = 25,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.decay_step = decay_step
        self.gamma = gamma

    def create(self, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Creates learning rate scheduler based on optimizer.

        :param optimizer: torch optimizer

        :returns: torch LRScheduler
        """
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.decay_step, gamma=self.gamma)
