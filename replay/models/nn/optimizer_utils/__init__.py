from replay.utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    from .optimizer_factory import FatLRSchedulerFactory, FatOptimizerFactory, LRSchedulerFactory, OptimizerFactory
