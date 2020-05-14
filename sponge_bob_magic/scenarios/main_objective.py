"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import collections
import logging
from typing import Any, Dict, List, Optional

from optuna import Study, Trial
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from sponge_bob_magic.constants import IntOrList, NumType
from sponge_bob_magic.experiment import Experiment
from sponge_bob_magic.metrics.base_metric import Metric
from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import get_top_k_recs

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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        study: Study,
        split_data: SplitData,
        recommender: Recommender,
        criterion: Metric,
        metrics: Dict[Metric, IntOrList],
        fallback_recs: DataFrame,
        k: int,
    ):
        self.metrics = metrics
        self.criterion = criterion
        self.k = k
        self.split_data = split_data
        self.recommender = recommender
        self.study = study
        self.search_space = search_space
        self.max_in_fallback_recs = (
            (fallback_recs.agg({"relevance": "max"}).collect()[0][0])
            if fallback_recs is not None
            else 0
        )
        self.fallback_recs = (
            (
                fallback_recs.withColumnRenamed(
                    "relevance", "relevance_fallback"
                )
            )
            if fallback_recs is not None
            else None
        )
        self.logger = logging.getLogger("sponge_bob_magic")
        self.experiment = Experiment(split_data.test, metrics)

    def __call__(self, trial: Trial,) -> float:
        """
        Эта функция вызывается при вычислении критерия в переборе параметров с помощью ``optuna``.
        Сигнатура функции совапдает с той, что описана в документации ``optuna``.

        :param trial: текущее испытание
        :return: значение критерия, который оптимизируется
        """
        params = dict()
        for key in self.search_space:
            params[key] = trial.suggest_uniform(
                key,
                low=min(self.search_space[key]) - 1,
                high=max(self.search_space[key]) + 1,
            )
        self.recommender.set_params(**params)
        self.logger.debug("-- Второй фит модели в оптимизации")
        self.recommender.fit(
            self.split_data.train,
            self.split_data.user_features,
            self.split_data.item_features,
            False,
        )
        self.logger.debug("-- Предикт модели в оптимизации")
        recs = self.recommender.predict(
            log=self.split_data.train,
            k=self.k,
            users=self.split_data.users,
            items=self.split_data.items,
            user_features=self.split_data.user_features,
            item_features=self.split_data.item_features,
        ).cache()
        recs = self._join_fallback_recs(
            recs, self.fallback_recs, self.k, self.max_in_fallback_recs
        )
        self.logger.debug("-- Подсчет метрики в оптимизации")
        criterion_value = self.criterion(recs, self.split_data.test, self.k)
        self.experiment.add_result(repr(self.recommender), recs)
        self.logger.debug("%s=%.2f", self.criterion, criterion_value)
        return criterion_value  # type: ignore

    def _join_fallback_recs(
        self,
        recs: DataFrame,
        fallback_recs: Optional[DataFrame],
        k: int,
        max_in_fallback_recs: float,
    ) -> DataFrame:
        """ Добавляет к рекомендациям fallback-рекомендации. """
        self.logger.debug("-- Длина рекомендаций: %d", recs.count())
        if fallback_recs is not None:
            self.logger.debug("-- Добавляем fallback рекомендации")
            # добавим максимум из fallback реков,
            # чтобы сохранить порядок при заборе топ-k
            recs = recs.withColumn(
                "relevance", sf.col("relevance") + 10 * max_in_fallback_recs
            )
            recs = recs.join(
                fallback_recs, on=["user_id", "item_id"], how="full_outer"
            )
            recs = recs.withColumn(
                "relevance", sf.coalesce("relevance", "relevance_fallback")
            ).select("user_id", "item_id", "relevance")
            recs = get_top_k_recs(recs, k)
            self.logger.debug(
                "-- Длина рекомендаций после добавления %s: %d",
                "fallback-рекомендаций",
                recs.count(),
            )
        return recs
