"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from datetime import datetime
from typing import Dict, Optional, List, Any

import joblib
import optuna
from pyspark.sql import DataFrame, SparkSession

from sponge_bob_magic import constants
from sponge_bob_magic.metrics.base_metrics import Metrics
from sponge_bob_magic.metrics.metrics import (HitRateMetric, NDCGMetric,
                                              PrecisionMetric)
from sponge_bob_magic.models.base_recommender import Recommender
from sponge_bob_magic.models.popular_recomennder import PopularRecommender
from sponge_bob_magic.scenarios.base_scenario import Scenario
from sponge_bob_magic.splitters.base_splitter import Splitter
from sponge_bob_magic.splitters.log_splitter import LogSplitByDateSplitter


class BasicScenario(Scenario):
    """ Сценарий для обучения модели популярных рекомендаций. """
    model: Optional[Recommender]
    study: Optional[optuna.Study]
    maximum_num_attempts: Optional[int] = 100

    def __init__(
            self,
            spark: SparkSession,
            splitter: Splitter,
            recommender: Recommender,
            criterion: Metrics,
            metrics: List[Metrics],
    ):
        super().__init__(spark)

        self.metrics = metrics
        self.criterion = criterion
        self.recommender = recommender
        self.splitter = splitter
        self.spark = spark

    def research(
            self,
            params_grid: Dict[str, Dict[str, Any]],
            log: DataFrame,
            users: Optional[DataFrame],
            items: Optional[DataFrame],
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            k: int = 10,
            context: Optional[str] = None,
            n_trials: int = 10,
            path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if context is None:
            context = constants.DEFAULT_CONTEXT

        train, predict_input, test = self.splitter.split(
            log,
            drop_cold_users=True,
            drop_cold_items=True
        )

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
        users = users if users else test.select("user_id").distinct().cache()
        items = items if items else test.select("item_id").distinct().cache()

        logging.debug("Пре-фит модели")
        self.recommender._pre_fit(train, user_features, item_features, path)

        def objective(trial: optuna.Trial):
            params = dict()
            for param_name, param_dict in params_grid.items():
                param = self.suggest_param(trial, param_name, param_dict)
                params[param_name] = param

            logging.debug(f"-- Параметры: {params}")
            self.recommender.set_params(**params)

            for t in trial.study.trials:
                # проверяем, что засемлпенные значения не повторялись раньше
                if t.state != optuna.structs.TrialState.COMPLETE:
                    continue

                if t.params == trial.params:
                    raise optuna.exceptions.TrialPruned(
                        "Повторные значения параметров"
                    )

            if path is not None:
                logging.debug("-- Сохраняем study на диск")
                joblib.dump(self.study,
                            os.path.join(path, "optuna_study.joblib"))

            logging.debug("-- Второй фит модели в оптимизации")
            self.recommender._fit_partial(log=train,
                                          user_features=user_features,
                                          item_features=item_features,
                                          path=None)

            logging.debug("-- Предикт модели в оптимизации")
            recs = self.recommender.predict(
                k=k,
                users=users, items=items,
                user_features=user_features,
                item_features=item_features,
                context=context,
                log=train,
                to_filter_seen_items=self.to_filter_seen_items
            )
            logging.debug(f"-- Длина рекомендаций: {recs.count()}")

            logging.debug("-- Подсчет метрики в оптимизации")
            result_string = "-- Метрики:"

            criterion_value = self.criterion(recs, test, k=k)
            result_string += f" {str(self.criterion)}={criterion_value:.4f}"

            for metric in self.metrics:
                value = metric(recs, test, k=k)
                trial.set_user_attr(str(metric), value)
                result_string += f" {str(metric)}={value:.4f}"

            logging.debug(result_string)
            return criterion_value

        logging.debug("-------------")
        logging.debug("Оптимизация параметров")
        logging.debug(
            f"Максимальное количество попыток: {self.maximum_num_attempts} "
            "(чтобы поменять его, задайте параметр 'maximum_num_attempts')")

        sampler = optuna.samplers.RandomSampler()
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        count = 1
        while (n_trials > len(set(str(t.params) for t in self.study.trials))
               and count <= self.maximum_num_attempts):
            self.study.optimize(objective, n_trials=1, n_jobs=self.n_jobs)
            count += 1

        logging.debug(f"Лучшие значения метрики: {self.study.best_value}")
        logging.debug(f"Лучшие параметры: {self.study.best_params}")
        return self.study.best_params

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
        self.recommender.set_params(**params)

        return self.recommender.fit_predict(
            k, users, items, context, log,
            user_features, item_features,
            self.to_filter_seen_items
        )


if __name__ == "__main__":
    spark_ = (SparkSession
              .builder
              .master("local[4]")
              .config("spark.driver.memory", "2g")
              .config("spark.sql.shuffle.partitions", "1")
              .appName("testing-pyspark")
              .enableHiveSupport()
              .getOrCreate())

    data = [
        ["user1", "item1", 1.0, "no_context", datetime(2019, 10, 8)],
        ["user2", "item2", 2.0, "no_context", datetime(2019, 10, 9)],
        ["user1", "item3", 1.0, "no_context", datetime(2019, 10, 10)],
        ["user1", "item1", 1.0, "no_context", datetime(2019, 10, 11)],
        ["user1", "item2", 1.0, "no_context", datetime(2019, 10, 12)],
        ["user3", "item3", 2.0, "no_context", datetime(2019, 10, 13)],
    ]
    schema = ["user_id", "item_id", "relevance", "context", "timestamp"]
    log_ = spark_.createDataFrame(data=data,
                                  schema=schema)

    popular_scenario = BasicScenario(
        spark_,
        LogSplitByDateSplitter(spark_, test_start=datetime(2019, 10, 11)),
        PopularRecommender(spark_),
        criterion=HitRateMetric(spark_),
        metrics=[NDCGMetric(spark_), PrecisionMetric(spark_)]
    )
    popular_params_grid = {"alpha": {"type": "int", "args": [0, 100]},
                           "beta": {"type": "int", "args": [0, 100]}, }

    best_params = popular_scenario.research(
        popular_params_grid,
        log_,
        users=None, items=None,
        user_features=None,
        item_features=None,
        k=2, context="no_context",
        n_trials=4
    )

    print(best_params)

    recs_ = popular_scenario.production(
        best_params,
        log_,
        users=None,
        items=None,
        k=3,
        context="no_context"
    )

    recs_.show()
