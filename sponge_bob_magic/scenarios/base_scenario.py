"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

import optuna
from pyspark.sql import DataFrame, SparkSession


class Scenario(ABC):
    """ Базовый класс сценария. """
    optuna_study: Optional[optuna.Study]
    optuna_max_n_trials: Optional[int] = 100
    optuna_n_jobs: int = 1
    filter_seen_items: bool = True

    def __init__(self, spark: SparkSession, **kwargs):
        self.spark = spark

    @staticmethod
    def suggest_param(
            trial: optuna.Trial,
            param_name: str,
            param_dict: Dict[str, Dict[str, Any]]
    ) -> Any:
        distribution_type = param_dict["type"]

        param = getattr(trial, f"suggest_{distribution_type}")(
            param_name, *param_dict["args"]
        )
        return param

    @staticmethod
    def suggest_all_params(
            trial: optuna.Trial,
            params_grid
    ):
        params = dict()
        for param_name, param_dict in params_grid.items():
            param = Scenario.suggest_param(trial, param_name, param_dict)
            params[param_name] = param
        return params

    @staticmethod
    def check_trial_on_duplicates(trial: optuna.Trial):
        for t in trial.study.trials:
            # проверяем, что засемлпенные значения не повторялись раньше
            if t.state != optuna.structs.TrialState.COMPLETE:
                continue

            if t.params == trial.params:
                raise optuna.exceptions.TrialPruned(
                    "Повторные значения параметров"
                )

    @abstractmethod
    def research(
            self,
            params_grid: Any,
            log: DataFrame,
            users: Optional[DataFrame],
            items: Optional[DataFrame],
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            k: int = 10,
            context: Optional[str] = None,
            n_trials: int = 10,
            path: Optional[str] = None
    ) -> Dict[str, Any]:
        """

        :param params_grid:
        :param log:
        :param users:
        :param items:
        :param user_features:
        :param item_features:
        :param k:
        :param context:
        :param n_trials:
        :param path:
        :return:
        """

    @abstractmethod
    def production(
            self,
            params: Dict[str, Any],
            log: DataFrame,
            users: Optional[DataFrame],
            items: Optional[DataFrame],
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            k: int = 10,
            context: Optional[str] = None
    ) -> DataFrame:
        """

        :param params:
        :param log:
        :param users:
        :param items:
        :param user_features:
        :param item_features:
        :param k:
        :param context:
        :return:
        """
