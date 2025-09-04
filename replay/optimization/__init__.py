"""
Hyperparameter optimization of models
"""
from replay.utils.types import OPTUNA_AVAILABLE

if OPTUNA_AVAILABLE:
    from .optuna_objective import ItemKNNObjective, ObjectiveWrapper
    from .optuna_mixin import IsOptimizible


    __all__ = ["ItemKNNObjective", "ObjectiveWrapper", "suggest_params", "IsOptimizible"]
else:
    __all__ = ["IsOptimizible"]
