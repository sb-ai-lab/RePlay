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

from sponge_bob_magic.constants import IterOrList
from sponge_bob_magic.metrics import Metric
from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import get_top_k_recs

SplitData = collections.namedtuple(
    "SplitData",
    "train predict_input test users items user_features item_features"
)


class MainObjective:
    """
    Класс функции, которая оптимизируется при подборе параметров (критерий).
    Принимает на вход объект класса `optuna.Trial` и возвращает значение
    метрики, которая оптимизируется.

    Вынесена в отдельный класс, так как она должна принимать только
    один аргумент. Вызов подсчета критерия происходит через `__call__`,
    а все остальные аргументы передаются через `__init__`.
    """

    def __init__(
            self,
            params_grid: Dict[str, Dict[str, Any]],
            study: Study,
            split_data: SplitData,
            recommender: Recommender,
            criterion: Metric,
            metrics: Dict[Metric, IterOrList],
            k: int = 10,
            context: Optional[str] = None,
            fallback_recs: Optional[DataFrame] = None,
            filter_seen_items: bool = False,
            path: str = None
    ):
        self.path = path
        self.metrics = metrics
        self.criterion = criterion
        self.context = context
        self.k = k
        self.split_data = split_data
        self.recommender = recommender
        self.study = study
        self.params_grid = params_grid
        self.filter_seen_items = filter_seen_items

        self.max_in_fallback_recs = (
            fallback_recs
            .agg({"relevance": "max"})
            .collect()[0][0]
        ) if fallback_recs is not None else 0

        self.fallback_recs = (
            fallback_recs
            .withColumnRenamed("context", "context_fallback")
            .withColumnRenamed("relevance", "relevance_fallback")
        ) if fallback_recs is not None else None

    def __call__(
            self,
            trial: Trial,
    ) -> float:
        """
        Основная функция, которую должны реализовать классы-наследники.
        Именно она вызывается при вычилении критерия в переборе параметров
        optuna. Сигнатура функции совапдает с той, что описана в документации
        к optuna.

        :param trial: текущее испытание
        :return: значение критерия, который оптимизируется
        """
        params = self._suggest_all_params(trial, self.params_grid)
        self.recommender.set_params(**params)

        self._check_trial_on_duplicates(trial)
        self._save_study(self.study, self.path)

        logging.debug("-- Второй фит модели в оптимизации")
        self.recommender._fit_partial(self.split_data.train,
                                      self.split_data.user_features,
                                      self.split_data.item_features)

        logging.debug("-- Предикт модели в оптимизации")
        recs = self.recommender.predict(log=self.split_data.predict_input, k=self.k, users=self.split_data.users,
                                        items=self.split_data.items, context=self.context,
                                        user_features=self.split_data.user_features,
                                        item_features=self.split_data.item_features,
                                        filter_seen_items=self.filter_seen_items)

        logging.debug("-- Дополняем рекомендации fallback рекомендациями")
        recs = self._join_fallback_recs(recs, self.fallback_recs, self.k,
                                        self.max_in_fallback_recs)

        logging.debug("-- Подсчет метрики в оптимизации")
        criterion_value = self._calculate_metrics(trial, recs,
                                                  self.split_data.test,
                                                  self.criterion, self.metrics,
                                                  self.k)

        return criterion_value

    @staticmethod
    def _suggest_param(
            trial: Trial,
            param_name: str,
            param_dict: Dict[str, Dict[str, Any]]
    ) -> Any:
        """ Сэмплит заданный параметр в соответствии с сеткой. """
        distribution_type = param_dict["type"]

        param = getattr(trial, f"suggest_{distribution_type}")(
                param_name, *param_dict["args"]
        )
        return param

    @staticmethod
    def _suggest_all_params(
            trial: Trial,
            params_grid: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """ Сэмплит все параметры модели в соответствии с заданной сеткой. """
        params = dict()
        for param_name, param_dict in params_grid.items():
            param = MainObjective._suggest_param(trial, param_name, param_dict)
            params[param_name] = param

        logging.debug(f"-- Параметры: {params}")
        return params

    @staticmethod
    def _check_trial_on_duplicates(trial: Trial):
        """ Проверяет, что испытание `trial` не повторяется с другими. """
        for another_trial in trial.study.trials:
            # проверяем, что засемлпенные значения не повторялись раньше
            if (another_trial.state == optuna.structs.TrialState.COMPLETE and
                    another_trial.params == trial.params):
                raise optuna.exceptions.TrialPruned(
                        "Повторные значения параметров"
                )

    @staticmethod
    def _save_study(
            study: Study,
            path: Optional[str]
    ):
        """ Сохраняет объект исследования `study` на диск. """
        if path is not None:
            logging.debug("-- Сохраняем optuna study на диск")
            joblib.dump(study,
                        os.path.join(path, "optuna_study.joblib"))

    @staticmethod
    def _calculate_metrics(
            trial: Trial,
            recommendations: DataFrame,
            ground_truth: DataFrame,
            criterion: Metric,
            metrics: Dict[Metric, IterOrList],
            k: int
    ) -> float:
        """ Подсчитывает все метрики и сохраняет их в `trial`. """
        result_string = "-- Метрики:"

        criterion_value = criterion(recommendations, ground_truth, k=k)[k]
        result_string += f" {criterion}={criterion_value:.4f}"

        for metric in metrics:
            values = metric(recommendations, ground_truth, k=metrics[metric])
            trial.set_user_attr(str(metric), values)
            values_str = ", ".join(f"{key}: {values[key]:.4f}" for key in values)
            result_string += f" {metric}={values_str}; "

        logging.debug(result_string)
        return criterion_value

    @staticmethod
    def _join_fallback_recs(
            recs: DataFrame,
            fallback_recs: Optional[DataFrame],
            k: int,
            max_in_fallback_recs: float
    ) -> DataFrame:
        """ Добавляет к рекомендациям fallback-рекомендации. """
        logging.debug(f"-- Длина рекомендаций: {recs.count()}")

        if fallback_recs is not None:
            # добавим максимум из fallback реков,
            # чтобы сохранить порядок при заборе топ-k
            recs = recs.withColumn(
                    "relevance",
                    sf.col("relevance") + 10 * max_in_fallback_recs
            )

            recs = recs.join(fallback_recs,
                             on=["user_id", "item_id"],
                             how="full_outer")
            recs = (recs
                    .withColumn("context",
                                sf.coalesce("context", "context_fallback"))
                    .withColumn("relevance",
                                sf.coalesce("relevance", "relevance_fallback"))
                    .select("user_id", "item_id", "context", "relevance"))

            recs = get_top_k_recs(recs, k)

            logging.debug(
                    "-- Длина рекомендаций после добавления fallback-рекомендаций:"
                    f" {recs.count()}"
            )

        return recs
