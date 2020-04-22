"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import collections
import logging
import os
from typing import Any, Dict, Optional

import joblib
import optuna
from optuna import Study, Trial
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from sponge_bob_magic.constants import IntOrList
from sponge_bob_magic.experiment import Experiment
from sponge_bob_magic.metrics.base_metric import Metric
from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import get_top_k_recs

SplitData = collections.namedtuple(
    "SplitData", "train test users items user_features item_features"
)


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

    def __init__(
        self,
        params_grid: Dict[str, Dict[str, Any]],
        study: Study,
        split_data: SplitData,
        recommender: Recommender,
        criterion: Metric,
        metrics: Dict[Metric, IntOrList],
        k: int = 10,
        fallback_recs: Optional[DataFrame] = None,
        filter_seen_items: bool = False,
        path: str = None,
    ):
        self.path = path
        self.metrics = metrics
        self.criterion = criterion
        self.k = k
        self.split_data = split_data
        self.recommender = recommender
        self.study = study
        self.params_grid = params_grid
        self.filter_seen_items = filter_seen_items
        self.max_in_fallback_recs = (
            (fallback_recs.agg({"relevance": "max"}).collect()[0][0])
            if fallback_recs is not None
            else 0
        )
        self.fallback_recs = (
            (fallback_recs.withColumnRenamed("relevance", "relevance_fallback"))
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
        params = self._suggest_all_params(trial, self.params_grid)
        self.recommender.set_params(**params)
        self._check_trial_on_duplicates(trial)
        self._save_study(self.study, self.path)
        self.logger.debug("-- Второй фит модели в оптимизации")
        self.recommender._fit(
            self.split_data.train,
            self.split_data.user_features,
            self.split_data.item_features,
        )
        self.logger.debug("-- Предикт модели в оптимизации")
        recs = self.recommender.predict(
            log=self.split_data.train,
            k=self.k,
            users=self.split_data.users,
            items=self.split_data.items,
            user_features=self.split_data.user_features,
            item_features=self.split_data.item_features,
            filter_seen_items=self.filter_seen_items,
        ).cache()
        self.logger.debug("-- Дополняем рекомендации fallback рекомендациями")
        recs = self._join_fallback_recs(
            recs, self.fallback_recs, self.k, self.max_in_fallback_recs
        )
        self.logger.debug("-- Подсчет метрики в оптимизации")
        criterion_value = self.criterion(recs, self.split_data.test, self.k)
        self.experiment.add_result(repr(self.recommender), recs)
        self.logger.debug("%s=%.2f", self.criterion, criterion_value)
        return criterion_value

    @staticmethod
    def _suggest_param(
        trial: Trial, param_name: str, param_dict: Dict[str, Dict[str, Any]]
    ) -> Any:
        """ Сэмплит заданный параметр в соответствии с сеткой. """
        distribution_type = param_dict["type"]
        param = getattr(trial, f"suggest_{distribution_type}")(
            param_name, *param_dict["args"]
        )
        return param

    def _suggest_all_params(
        self, trial: Trial, params_grid: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """ Сэмплит все параметры модели в соответствии с заданной сеткой. """
        params = dict()
        for param_name, param_dict in params_grid.items():
            param = MainObjective._suggest_param(trial, param_name, param_dict)
            params[param_name] = param
        self.logger.debug("-- Параметры: %s", params)
        return params

    @staticmethod
    def _check_trial_on_duplicates(trial: Trial):
        """ Проверяет, что испытание `trial` не повторяется с другими. """
        for another_trial in trial.study.trials:
            # проверяем, что засемлпенные значения не повторялись раньше
            if (
                another_trial.state == optuna.structs.TrialState.COMPLETE
                and another_trial.params == trial.params
            ):
                raise optuna.exceptions.TrialPruned("Повторные значения параметров")

    def _save_study(self, study: Study, path: Optional[str]):
        """ Сохраняет объект исследования `study` на диск. """
        if path is not None:
            self.logger.debug("-- Сохраняем optuna study на диск")
            joblib.dump(study, os.path.join(path, "optuna_study.joblib"))

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
            # добавим максимум из fallback реков,
            # чтобы сохранить порядок при заборе топ-k
            recs = recs.withColumn(
                "relevance", sf.col("relevance") + 10 * max_in_fallback_recs
            )
            recs = recs.join(fallback_recs, on=["user_id", "item_id"], how="full_outer")
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
