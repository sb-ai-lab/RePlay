"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import collections
import logging
import os
from datetime import datetime
from typing import Dict, Optional, List, Any

import joblib
import optuna
from pyspark.sql import DataFrame, SparkSession

from sponge_bob_magic import constants
from sponge_bob_magic.metrics.base_metrics import Metric
from sponge_bob_magic.metrics.metrics import (HitRateMetric, NDCGMetric,
                                              PrecisionMetric)
from sponge_bob_magic.models.base_recommender import Recommender
from sponge_bob_magic.models.popular_recomennder import PopularRecommender
from sponge_bob_magic.scenarios.base_scenario import Scenario
from sponge_bob_magic.splitters.base_splitter import Splitter
from sponge_bob_magic.splitters.log_splitter import LogSplitByDateSplitter

SplitData = collections.namedtuple(
    'SplitData',
    'train predict_input test users items user_features item_features'
)


class Objective:
    def __init__(
            self,
            params_grid,
            study,
            recommender,
            split_data: SplitData,
            k,
            context,
            criterion,
            metrics,
            path
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

    def __call__(
            self,
            trial,
    ):
        params = Scenario.suggest_all_params(trial, self.params_grid)
        logging.debug(f"-- Параметры: {params}")

        self.recommender.set_params(**params)

        Scenario.check_trial_on_duplicates(trial)

        if self.path is not None:
            logging.debug("-- Сохраняем optuna study на диск")
            joblib.dump(self.study,
                        os.path.join(self.path, "optuna_study.joblib"))

        logging.debug("-- Второй фит модели в оптимизации")
        self.recommender._fit_partial(self.split_data.train,
                                      self.split_data.user_features,
                                      self.split_data.item_features,
                                      path=None)

        logging.debug("-- Предикт модели в оптимизации")
        recs = self.recommender.predict(
            k=self.k,
            users=self.split_data.users, items=self.split_data.items,
            user_features=self.split_data.user_features,
            item_features=self.split_data.item_features,
            context=self.context,
            log=self.split_data.predict_input,
            filter_seen_items=Scenario.filter_seen_items
        )
        logging.debug(f"-- Длина рекомендаций: {recs.count()}")

        logging.debug("-- Подсчет метрики в оптимизации")
        result_string = "-- Метрики:"

        criterion_value = self.criterion(recs, self.split_data.test, k=self.k)
        result_string += f" {str(self.criterion)}={criterion_value:.4f}"

        for metric in self.metrics:
            value = metric(recs, self.split_data.test, k=self.k)
            trial.set_user_attr(str(metric), value)
            result_string += f" {str(metric)}={value:.4f}"

        logging.debug(result_string)
        return criterion_value


class MainScenario(Scenario):
    """ Сценарий для обучения модели популярных рекомендаций. """

    def __init__(
            self,
            spark: SparkSession,
            splitter: Splitter,
            recommender: Recommender,
            criterion: Metric,
            metrics: List[Metric],
    ):
        super().__init__(spark)

        self.metrics = metrics
        self.criterion = criterion
        self.recommender = recommender
        self.splitter = splitter
        self.spark = spark

    def prepare_data(self, log, users, items, user_features, item_features):
        train, predict_input, test = self.splitter.split(log)
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

        split_data = SplitData(train, predict_input, test,
                               users, items,
                               user_features, item_features)

        return split_data

    def run_optimization(self, n_trials, params_grid, split_data, k
                         , context, path=None):
        logging.debug("-------------")
        logging.debug("Оптимизация параметров")
        logging.debug(
            f"Максимальное количество попыток: {self.optuna_max_n_trials} "
            "(чтобы поменять его, задайте параметр 'optuna_max_n_trials')")

        sampler = optuna.samplers.RandomSampler()
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        count = 1
        n_unique_trials = 0
        while n_trials > n_unique_trials and count <= self.optuna_max_n_trials:
            self.study.optimize(
                Objective(params_grid, self.study,
                          self.recommender, split_data,
                          k, context,
                          self.criterion, self.metrics,
                          path),
                n_trials=1,
                n_jobs=self.optuna_n_jobs)
            count += 1
            n_unique_trials = len(
                set(str(t.params)
                    for t in self.study.trials)
            )

        logging.debug(f"Лучшие значения метрики: {self.study.best_value}")
        logging.debug(f"Лучшие параметры: {self.study.best_params}")

        return self.study.best_params

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

        split_data = self.prepare_data(log,
                                       users, items,
                                       user_features, item_features)

        logging.debug("Пре-фит модели")
        self.recommender._pre_fit(
            split_data.train,
            split_data.user_features,
            split_data.item_features,
            path
        )

        best_params = self.run_optimization(
            n_trials,
            params_grid,
            split_data,
            k,
            context,
            path=path
        )

        return best_params

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
            self.filter_seen_items
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
    spark_logger = logging.getLogger('py4j')
    spark_logger.setLevel(logging.WARN)

    logger = logging.getLogger()
    formatter = logging.Formatter(
        '%(asctime)s, %(name)s, %(levelname)s: %(message)s',
        datefmt='%d-%b-%y %H:%M:%S')
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)

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

    popular_scenario = MainScenario(
        spark_,
        LogSplitByDateSplitter(spark_, True, True, datetime(2019, 10, 11)),
        PopularRecommender(spark_),
        criterion=HitRateMetric(spark_),
        metrics=[NDCGMetric(spark_), PrecisionMetric(spark_)]
    )
    popular_params_grid = {"alpha": {"type": "int", "args": [0, 100]},
                           "beta": {"type": "int", "args": [0, 100]}, }

    best_params_ = popular_scenario.research(
        popular_params_grid,
        log_,
        users=None, items=None,
        user_features=None,
        item_features=None,
        k=2, context="no_context",
        n_trials=4
    )

    print(best_params_)

    recs_ = popular_scenario.production(
        best_params_,
        log_,
        users=None,
        items=None,
        k=3,
        context="no_context"
    )

    recs_.show()
