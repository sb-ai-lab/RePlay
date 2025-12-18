import abc
import warnings
from collections.abc import Iterator
from typing import Literal

import torch


class OptimizerFactory(abc.ABC):
    """
    Interface for optimizer factory
    """

    @abc.abstractmethod
    def create(
        self, parameters: Iterator[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:  # pragma: no cover
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
    def create(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:  # pragma: no cover
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
            return torch.optim.Adam(
                parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.betas,
            )
        if self.optimizer == "sgd":
            return torch.optim.SGD(
                parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.sgd_momentum,
            )

        msg = "Unexpected optimizer"
        raise ValueError(msg)


class FatLRSchedulerFactory(LRSchedulerFactory):
    """
    Factory that creates learning rate schedule depending on passed parameters
    """

    def __init__(self, decay_step: int = 25, gamma: float = 1.0) -> None:
        super().__init__()
        self.decay_step = decay_step
        self.gamma = gamma

    def create(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Creates learning rate scheduler based on optimizer.

        :param optimizer: torch optimizer

        :returns: torch LRScheduler
        """
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.decay_step, gamma=self.gamma
        )


class LambdaLRSchedulerFactory(LRSchedulerFactory):
    """
    Factory that creates learning rate schedule depending on passed parameters
    """

    def __init__(
        self,
        warmup_steps: int,
        warmup_lr: float = 1.0,
        normal_lr: float = 0.1,
        update_interval: Literal["epoch", "step"] = "epoch",
    ) -> None:
        super().__init__()

        if normal_lr <= 0.0:
            msg = f"Normal LR must be positive. Got {normal_lr}"
            raise ValueError(msg)
        if warmup_lr <= 0.0:
            msg = f"Warmup LR must be positive. Got {warmup_lr}"
            raise ValueError(msg)
        if normal_lr >= warmup_lr:
            msg = f"Suspicious LR pair: {normal_lr=}, {warmup_lr=}"
            warnings.warn(msg, stacklevel=2)

        self.warmup_lr = warmup_lr
        self.normal_lr = normal_lr
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval

    def create(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Creates learning rate scheduler based on optimizer.

        :param optimizer: torch optimizer

        :returns: torch LambdaLR
        """
        return {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda),
            "interval": self.update_interval,
            "frequency": 1,
        }

    def lr_lambda(self, current_step: int) -> float:
        if current_step >= self.warmup_steps:
            return self.normal_lr
        return self.warmup_lr
