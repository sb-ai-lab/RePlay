"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Set, Tuple, TypeVar

import joblib
import optuna
from pyspark.sql import DataFrame, SparkSession

from sponge_bob_magic import constants
from sponge_bob_magic.metrics.metrics import Metric
from sponge_bob_magic.models.base_recommender import Recommender
from sponge_bob_magic.models.linear_recomennder import LinearRecommender
from sponge_bob_magic.splitters.log_splitter import (LogSplitByDateSplitter,
                                                     LogSplitRandomlySplitter)

TNum = TypeVar("TNum", int, float)


class LinearScenario:
    """ Сценарий с линейной моделью с эмбеддингами. """
    num_attempts: int = 10
    tested_params_set: Set[Tuple[TNum, TNum]] = set()
    seed: int = 1234
    model: Optional[Recommender]
    study: Optional[optuna.Study]
    maximum_num_attempts: Optional[int] = 100

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def research(
            self,
            params_grid: Dict[str, Tuple[TNum, TNum]],
            log: DataFrame,
            users: Optional[DataFrame],
            items: Optional[DataFrame],
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            test_start: Optional[datetime] = None,
            test_size: float = None,
            k: int = 10,
            context: Optional[str] = None,
            to_filter_seen_items: bool = True,
            n_trials: int = 10,
            n_jobs: int = 1,
            how_to_split: str = "by_date",
            path: Optional[str] = None
    ) -> Dict[str, Any]:
        if context is None:
            context = constants.DEFAULT_CONTEXT

        logging.debug("Деление на трейн и тест")
        if how_to_split == "by_date":
            splitter = LogSplitByDateSplitter(spark=self.spark,
                                              test_start=test_start)
        elif how_to_split == "randomly":
            splitter = LogSplitRandomlySplitter(spark=self.spark,
                                                test_size=test_size,
                                                seed=self.seed)
        else:
            raise ValueError(
                f"Значение how_to_split неверное ({how_to_split}), "
                "допустимые варианты: 'by_date' или 'randomly'")

        train, predict_input, test = splitter.split(log,
                                                    drop_cold_users=True,
                                                    drop_cold_items=True)

        # рассчитываем все выборки перед подбором параметров
        train.cache()
        predict_input.cache()
        test.cache()
        logging.debug(f"Длина трейна и теста: {train.count(), test.count()}")
        logging.debug("Количество юзеров в трейне и тесте: "
                      f"{train.select('user_id').distinct().count()}, "
                      f"{test.select('user_id').distinct().count()}")
        logging.debug("Количество айтемов в трейне и тесте: "
                      f"{train.select('item_id').distinct().count()}, "
                      f"{test.select('item_id').distinct().count()}")

        # если users или items нет, возьмем всех из теста,
        # чтобы не делать на каждый trial их заново
        if users is None:
            users = test.select("user_id").distinct().cache()

        if items is None:
            items = test.select("item_id").distinct().cache()

        logging.debug("Модель LinearRecommender")
        self.model = LinearRecommender(self.spark)

        logging.debug("Первый пре-фит модели")
        self.model._pre_fit(log=train,
                            user_features=user_features,
                            item_features=item_features,
                            path=None)

        def objective(trial: optuna.Trial):
            if path is not None:
                joblib.dump(self.study,
                            os.path.join(path, "optuna_study.joblib"))

            lambda_ = trial.suggest_loguniform(
                "lambda_param",
                params_grid["lambda_param"][0],
                params_grid["lambda_param"][1],
            )
            elastic_net = trial.suggest_uniform(
                "elastic_net_param",
                params_grid["elastic_net_param"][0],
                params_grid["elastic_net_param"][1],
            )

            for t in trial.study.trials:
                # проверяем, что засемлпенные значения не повторялись раньше
                if t.state != optuna.structs.TrialState.COMPLETE:
                    continue

                if t.params == trial.params:
                    raise optuna.exceptions.TrialPruned(
                        "Повторные значения параметров")

            params = {"lambda_param": lambda_,
                      "elastic_net_param": elastic_net,
                      "num_iter": 100}
            self.model.set_params(**params)
            logging.debug(f"-- Параметры: {params}")

            logging.debug("-- Второй фит модели в оптимизации")
            self.model._fit_partial(log=train,
                                    user_features=user_features,
                                    item_features=item_features,
                                    path=None)

            logging.debug("-- Предикт модели в оптимизации")
            recs = self.model.predict(
                k=k,
                users=users, items=items,
                user_features=user_features,
                item_features=item_features,
                context=context,
                log=train,
                filter_seen_items=to_filter_seen_items
            )
            logging.debug(f"-- Длина рекомендаций: {recs.count()}")

            logging.debug("-- Подсчет метрики в оптимизации")
            hit_rate = Metric.hit_rate_at_k(recs, test, k=k)
            ndcg = Metric.ndcg_at_k(recs, test, k=k)
            precision = Metric.precision_at_k(recs, test, k=k)
            map_metric = Metric.map_at_k(recs, test, k=k)

            trial.set_user_attr("nDCG@k", ndcg)
            trial.set_user_attr("precision@k", precision)
            trial.set_user_attr("MAP@k", map_metric)
            trial.set_user_attr("HitRate@k", hit_rate)

            logging.debug(f"-- Метрики: "
                          f"hit_rate={hit_rate:.4f}, "
                          f"ndcg={ndcg:.4f}, "
                          f"precision={precision:.4f}, "
                          f"map_metric={map_metric:.4f}"
                          )

            return hit_rate

        logging.debug("-------------")
        logging.debug("Начало оптимизации параметров")
        logging.debug(
            f"Максимальное количество попыток: {self.maximum_num_attempts} "
            "(чтобы поменять его, задайте параметр 'optuna_max_n_trials')")
        sampler = optuna.samplers.RandomSampler()
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        count = 1
        while n_trials > len(set(str(t.params) for t in self.study.trials)) \
                and count <= self.maximum_num_attempts:
            self.study.optimize(objective, n_trials=1, n_jobs=n_jobs)
            count += 1

        logging.debug(f"Лучшие значения метрики: {self.study.best_value}")
        logging.debug(f"Лучшие параметры: {self.study.best_params}")
        return self.study.best_params

    def production(
            self,
            params,
            log: DataFrame,
            users: Optional[DataFrame],
            items: Optional[DataFrame],
            user_features: Optional[DataFrame],
            item_features: Optional[DataFrame],
            k: int,
            context: Optional[str],
            to_filter_seen_items: bool
    ) -> DataFrame:
        self.model = LinearRecommender(self.spark)
        self.model.set_params(**params)

        return self.model.fit_predict(k, users, items, context, log,
                                      user_features, item_features,
                                      to_filter_seen_items)
