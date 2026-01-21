from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from .optimizer_factory import (
        BaseLRSchedulerFactory,
        BaseOptimizerFactory,
        LambdaLRSchedulerFactory,
        LRSchedulerFactory,
        OptimizerFactory,
    )
