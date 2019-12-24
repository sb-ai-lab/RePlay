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

from sponge_bob_magic.constants import DEFAULT_CONTEXT
from sponge_bob_magic.metrics.base_metrics import Metric
from sponge_bob_magic.metrics.metrics import (HitRateMetric, NDCGMetric,
                                              PrecisionMetric)
from sponge_bob_magic.models.base_recommender import Recommender
from sponge_bob_magic.models.knn_recommender import KNNRecommender
from sponge_bob_magic.models.popular_recomennder import PopularRecommender
from sponge_bob_magic.scenarios.base_scenario import Scenario
from sponge_bob_magic.splitters.base_splitter import Splitter
from sponge_bob_magic.splitters.log_splitter import LogSplitByDateSplitter

SplitData = collections.namedtuple(
    "SplitData",
    "train predict_input test users items user_features item_features"
)


class Objective:
    """
    Функция, которая оптимизируется при подборе параметров.
    Принимает на вход объект класса `optuna.Trial` и возвращает значение
    метрики, которая оптимизируется.

    Вынесена в отдельный класс, так как она должна принимать только
    один аргумент, и все остальные аргументы передаются через callback.
    """

    def __init__(
            self,
            params_grid: Dict[str, Dict[str, Any]],
            study: optuna.Study,
            split_data: SplitData,
            recommender: Recommender,
            criterion: Metric,
            metrics: List[Metric],
            k: int = 10,
            context: Optional[str] = None,
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

    def __call__(
            self,
            trial,
    ):
        params = Scenario._suggest_all_params(trial, self.params_grid)
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
    """ Сценарий для простого обучения моделей рекомендаций. """

    spark: SparkSession
    splitter: Splitter
    recommender: Recommender
    criterion: Metric
    metrics: List[Metric]
    study: optuna.Study

    def _prepare_data(
            self,
            log: DataFrame,
            users: Optional[DataFrame] = None,
            items: Optional[DataFrame] = None,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
    ) -> SplitData:
        """ Делит лог и готовит объекти типа `SplitData`. """
        train, predict_input, test = self.splitter.split(log)

        train.cache()
        predict_input.cache()
        test.cache()

        logging.debug(f"Длина трейна и теста: {train.count(), test.count()}")
        logging.debug("Количество пользователей в трейне и тесте: "
                      f"{train.select('user_id').distinct().count()}, "
                      f"{test.select('user_id').distinct().count()}")
        logging.debug("Количество объектов в трейне и тесте: "
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

    def run_optimization(
            self,
            n_trials: int,
            params_grid: Dict[str, Dict[str, Any]],
            split_data: SplitData,
            k: int = 10,
            context: Optional[str] = None,
            path=None
    ) -> Dict[str, Any]:
        """ Запускает подбор параметров в optuna. """
        sampler = optuna.samplers.RandomSampler()
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        # делаем триалы до тех пор, пока не засемплим уникальных n_trials или
        # не используем максимально попыток
        count = 1
        n_unique_trials = 0
        while n_trials > n_unique_trials and count <= self.optuna_max_n_trials:
            self.study.optimize(
                Objective(params_grid, self.study, split_data,
                          self.recommender, self.criterion, self.metrics, k,
                          context, path),
                n_trials=1,
                n_jobs=self.optuna_n_jobs
            )

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
            users: Optional[DataFrame] = None,
            items: Optional[DataFrame] = None,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            k: int = 10,
            context: Optional[str] = None,
            n_trials: int = 10,
            path: Optional[str] = None,
    ) -> Dict[str, Any]:
        context = context if context else DEFAULT_CONTEXT

        logging.debug("Деление лога на обучающую и тестовую выборку")
        split_data = self._prepare_data(log,
                                        users, items,
                                        user_features, item_features)

        logging.debug("Пре-фит модели")
        self.recommender._pre_fit(split_data.train,
                                  split_data.user_features,
                                  split_data.item_features,
                                  path=path)

        logging.debug("-------------")
        logging.debug("Оптимизация параметров")
        logging.debug(
            f"Максимальное количество попыток: {self.optuna_max_n_trials} "
            "(чтобы поменять его, задайте параметр 'optuna_max_n_trials')")

        best_params = self.run_optimization(n_trials, params_grid,
                                            split_data,
                                            k, context,
                                            path=path)

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
    spark_logger = logging.getLogger("py4j")
    spark_logger.setLevel(logging.WARN)

    logger = logging.getLogger()
    formatter = logging.Formatter(
        "%(asctime)s, %(name)s, %(levelname)s: %(message)s",
        datefmt="%d-%b-%y %H:%M:%S"
    )
    hdlr = logging.StreamHandler()
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.DEBUG)

    data = [
        ["user1", "item1", 1.0, "no_context", datetime(2019, 10, 8)],
        ["user1", "item2", 2.0, "no_context", datetime(2019, 10, 9)],
        ["user1", "item3", 1.0, "no_context", datetime(2019, 10, 10)],
        ["user2", "item1", 1.0, "no_context", datetime(2019, 10, 11)],
        ["user2", "item3", 1.0, "no_context", datetime(2019, 10, 12)],
        ["user3", "item2", 1.0, "no_context", datetime(2019, 10, 13)],
        ["user3", "item1", 1.0, "no_context", datetime(2019, 10, 14)],

        ["user1", "item1", 1.0, "no_context", datetime(2019, 10, 15)],
        ["user1", "item2", 1.0, "no_context", datetime(2019, 10, 16)],
        ["user2", "item3", 2.0, "no_context", datetime(2019, 10, 17)],
        ["user3", "item2", 2.0, "no_context", datetime(2019, 10, 18)],
    ]
    schema = ["user_id", "item_id", "relevance", "context", "timestamp"]
    log_ = spark_.createDataFrame(data=data,
                                  schema=schema)

    scenario = MainScenario(spark_)
    scenario.splitter = LogSplitByDateSplitter(spark_, True, True,
                                               datetime(2019, 10, 14))
    scenario.criterion = HitRateMetric(spark_)
    scenario.metrics = [NDCGMetric(spark_), PrecisionMetric(spark_)]
    scenario.optuna_max_n_trials = 10

    flag = True
    if flag:
        scenario.recommender = PopularRecommender(spark_)
        grid = {"alpha": {"type": "int", "args": [0, 100]},
                "beta": {"type": "int", "args": [0, 100]}}
    else:
        scenario.recommender = KNNRecommender(spark_)
        grid = {"num_neighbours": {"type": "categorical",
                                   "args": [[1]]}}

    best_params_ = scenario.research(grid, log_,
                                     k=2, n_trials=4)

    recs_ = scenario.production(best_params_, log_,
                                users=None, items=None,
                                k=2)

    recs_.show()
