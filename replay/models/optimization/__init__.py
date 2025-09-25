"""
Hyperparameter optimization of models
"""

from replay.utils.types import OPTUNA_AVAILABLE

from .optuna_mixin import IsOptimizible

if OPTUNA_AVAILABLE:
    from .optuna_objective import ItemKNNObjective, ObjectiveWrapper

    __all__ = ["IsOptimizible", "ItemKNNObjective", "ObjectiveWrapper"]
else:
    __all__ = ["IsOptimizible"]
