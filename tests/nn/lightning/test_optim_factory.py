from contextlib import nullcontext as no_exception

import pytest
import torch

from replay.nn.lightning import LightningModule
from replay.nn.lightning.optimizers import OptimizerFactory
from replay.nn.lightning.schedulers import LambdaLRSchedulerFactory, LRSchedulerFactory


@pytest.mark.parametrize(
    "optimizer_factory, lr_scheduler_factory, optimizer_type",
    [
        (None, None, torch.optim.Adam),
        (OptimizerFactory(), None, torch.optim.Adam),
        (OptimizerFactory("sgd"), None, torch.optim.SGD),
        (None, LRSchedulerFactory(), torch.optim.Adam),
        (OptimizerFactory(), LRSchedulerFactory(), torch.optim.Adam),
        (None, LambdaLRSchedulerFactory(6), torch.optim.Adam),
        (OptimizerFactory(), LambdaLRSchedulerFactory(6), torch.optim.Adam),
    ],
)
def test_model_configure_optimizers(sasrec_model, optimizer_factory, lr_scheduler_factory, optimizer_type):
    lit_model = LightningModule(
        sasrec_model,
        lr_scheduler_factory=lr_scheduler_factory,
        optimizer_factory=optimizer_factory,
    )

    parameters = lit_model.configure_optimizers()
    if isinstance(parameters, tuple):
        assert isinstance(parameters[0][0], optimizer_type)
        if isinstance(lr_scheduler_factory, LRSchedulerFactory):
            assert isinstance(parameters[1][0], torch.optim.lr_scheduler.StepLR)
        if isinstance(lr_scheduler_factory, LambdaLRSchedulerFactory):
            assert isinstance(parameters[1][0]["scheduler"], torch.optim.lr_scheduler.LambdaLR)
    else:
        assert isinstance(parameters, optimizer_type)


def test_model_get_set_optim_factory(sasrec_model):
    optim_factory = OptimizerFactory()
    lit_model = LightningModule(sasrec_model, optimizer_factory=optim_factory)

    assert lit_model._optimizer_factory is optim_factory
    new_factory = OptimizerFactory(learning_rate=0.1)
    lit_model._optimizer_factory = new_factory
    assert lit_model._optimizer_factory is new_factory


@pytest.mark.torch
def test_model_configure_wrong_optimizer(sasrec_model):
    lit_model = LightningModule(
        sasrec_model,
        optimizer_factory=OptimizerFactory(""),
    )

    with pytest.raises(ValueError) as exc:
        lit_model.configure_optimizers()

    assert str(exc.value) == "Unexpected optimizer"


@pytest.mark.parametrize(
    "warmup_lr, normal_lr, expected_exception",
    [
        (-1.0, 0.1, pytest.raises(ValueError)),
        (1.0, -0.1, pytest.raises(ValueError)),
        (0.0, 0.0, pytest.raises(ValueError)),
        (0.1, 0.01, no_exception()),
        (0.1, 1.0, no_exception()),
    ],
)
def test_configure_lambda_lr_scheduler(warmup_lr, normal_lr, expected_exception):
    with expected_exception:
        LambdaLRSchedulerFactory(warmup_steps=1, warmup_lr=warmup_lr, normal_lr=normal_lr)
