"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import collections
import logging
from typing import Any, Dict, List, Optional

from optuna import Trial
from pyspark.sql import DataFrame

from replay.constants import IntOrList
from replay.experiment import Experiment
from replay.metrics.base_metric import Metric
from replay.models.base_rec import Recommender
from replay.utils import fallback

SplitData = collections.namedtuple(
    "SplitData", "train test users items user_features item_features"
)


# pylint: disable=too-few-public-methods
class MainObjective:
    """
    Данный класс реализован в соответствии с
    `инструкцией <https://optuna.readthedocs.io/en/stable/faq.html#how-to-define-objective-functions-that-have-own-arguments>`_
    по интеграции произвольной библиотеки машинного обучения с ``optuna``.
    По факту представляет собой обёртку вокруг некоторой целевой функции (процедуры обучения модели),
    параметры которой (гипер-параметры модели) ``optuna`` подбирает.

    Вызов подсчета критерия происходит через ``__call__``,
    а все остальные аргументы передаются через ``__init__``.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        split_data: SplitData,
        recommender: Recommender,
        criterion: Metric,
        metrics: Dict[Metric, IntOrList],
        fallback_recs: Optional[DataFrame],
        k: int,
    ):
        self.metrics = metrics
        self.criterion = criterion
        self.k = k
        self.split_data = split_data
        self.recommender = recommender
        self.search_space = search_space
        self.fallback_recs = fallback_recs
        self.logger = logging.getLogger("replay")
        self.experiment = Experiment(split_data.test, metrics)

    def __call__(self, trial: Trial) -> float:
        """
        Эта функция вызывается при вычислении критерия в переборе параметров с помощью ``optuna``.
        Сигнатура функции совапдает с той, что описана в документации ``optuna``.

        :param trial: текущее испытание
        :return: значение критерия, который оптимизируется
        """
        params = dict()
        for key in self.search_space:
            params[key] = self.search_space[key][
                trial.suggest_int(
                    key, low=0, high=len(self.search_space[key]) - 1,
                )
            ]
        self.recommender.set_params(**params)
        self.logger.debug("-- Второй фит модели в оптимизации")
        # pylint: disable=protected-access
        self.recommender._fit_wrap(
            self.split_data.train,
            self.split_data.user_features,
            self.split_data.item_features,
            False,
        )
        self.logger.debug("-- Предикт модели в оптимизации")
        # pylint: disable=protected-access
        recs = self.recommender._predict_wrap(
            log=self.split_data.train,
            k=self.k,
            users=self.split_data.users,
            items=self.split_data.items,
            user_features=self.split_data.user_features,
            item_features=self.split_data.item_features,
        ).cache()
        if self.fallback_recs is not None:
            recs = fallback(recs, self.fallback_recs, self.k)
        self.logger.debug("-- Подсчет метрики в оптимизации")
        criterion_value = self.criterion(recs, self.split_data.test, self.k)
        self.experiment.add_result(f"{str(self.recommender)}{params}", recs)
        self.logger.debug("%s=%.2f", self.criterion, criterion_value)
        return criterion_value  # type: ignore
