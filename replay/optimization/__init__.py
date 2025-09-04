"""
Hyperparameter optimization of models
"""

from replay.utils.types import OPTUNA_AVAILABLE

if OPTUNA_AVAILABLE:
    from .optuna_mixin import IsOptimizible
    from .optuna_objective import ItemKNNObjective, ObjectiveWrapper

    __all__ = ["IsOptimizible", "ItemKNNObjective", "ObjectiveWrapper", "suggest_params"]
else:
    __all__ = ["IsOptimizible"]
