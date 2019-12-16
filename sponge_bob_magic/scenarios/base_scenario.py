"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any

import optuna
from pyspark.sql import DataFrame, SparkSession

from sponge_bob_magic.models.base_recommender import Recommender


class Scenario(ABC):
    """ Базовый класс сценария. """
    model: Optional[Recommender]
    study: Optional[optuna.Study]
    maximum_num_attempts: Optional[int] = 100
    n_jobs: int = 1
    to_filter_seen_items: bool = True

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
