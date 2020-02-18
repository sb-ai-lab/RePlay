"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from optuna import Study, create_study, samplers
from pyspark.sql import DataFrame, SparkSession

from sponge_bob_magic.constants import DEFAULT_CONTEXT
from sponge_bob_magic.metrics import NDCG, HitRate, Metric, Precision
from sponge_bob_magic.models.base_recommender import Recommender
from sponge_bob_magic.models.knn_recommender import KNNRecommender
from sponge_bob_magic.models.popular_recomennder import PopularRecommender
from sponge_bob_magic.scenarios.base_scenario import Scenario
from sponge_bob_magic.scenarios.main_scenario.main_objective import (
    MainObjective, SplitData)
from sponge_bob_magic.splitters.base_splitter import Splitter
from sponge_bob_magic.splitters.log_splitter import DateSplitter
from sponge_bob_magic.utils import write_read_dataframe


class MainScenario(Scenario):
    """ Сценарий для простого обучения моделей рекомендаций с замесом. """
    splitter: Splitter
    recommender: Recommender
    fallback_recommender: Recommender
    criterion: Metric
    metrics: List[Metric]
    study: Study

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
        spark = SparkSession(log.rdd.context)
        train = write_read_dataframe(
            train,
            os.path.join(spark.conf.get("spark.local.dir"), "train")
        )
        predict_input = write_read_dataframe(
            predict_input,
            os.path.join(spark.conf.get("spark.local.dir"),
                         "predict_input")
        )
        test = write_read_dataframe(
            test,
            os.path.join(spark.conf.get("spark.local.dir"), "test")
        )

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
            fallback_recs: Optional[DataFrame] = None
    ) -> Dict[str, Any]:
        """ Запускает подбор параметров в optuna. """
        sampler = samplers.RandomSampler()
        self.study = create_study(direction="maximize", sampler=sampler)

        # делаем триалы до тех пор, пока не засемплим уникальных n_trials или
        # не используем максимально попыток
        count = 1
        n_unique_trials = 0

        while n_trials > n_unique_trials and count <= self.optuna_max_n_trials:
            spark = SparkSession(split_data.train.rdd.context)
            self.study.optimize(
                MainObjective(
                    params_grid, self.study, split_data,
                    self.recommender, self.criterion, self.metrics,
                    k, context,
                    fallback_recs,
                    self.filter_seen_items,
                    spark.conf.get("spark.local.dir")),
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

        logging.debug("Обучение и предсказание дополнительной модели")
        fallback_recs = self._predict_fallback_recs(self.fallback_recommender,
                                                    split_data, k, context)

        logging.debug("Пре-фит модели")
        self.recommender._pre_fit(split_data.train, split_data.user_features,
                                  split_data.item_features)

        logging.debug("-------------")
        logging.debug("Оптимизация параметров")
        logging.debug(
            f"Максимальное количество попыток: {self.optuna_max_n_trials} "
            "(чтобы поменять его, задайте параметр 'optuna_max_n_trials')")

        best_params = self.run_optimization(n_trials, params_grid, split_data,
                                            k, context, fallback_recs)
        return best_params

    def _predict_fallback_recs(
            self,
            fallback_recommender: Recommender,
            split_data: SplitData,
            k: int,
            context: Optional[str] = None
    ) -> Optional[DataFrame]:
        """ Обучает fallback модель и возвращает ее рекомендации. """
        fallback_recs = None

        if fallback_recommender is not None:
            fallback_recs = (
                fallback_recommender
                .fit_predict(k,
                             split_data.users, split_data.items,
                             context,
                             split_data.predict_input,
                             split_data.user_features,
                             split_data.item_features,
                             self.filter_seen_items)
            )
        return fallback_recs

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
              .config(
                  "spark.local.dir", os.path.join(os.environ["HOME"], "tmp"))
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

    scenario = MainScenario()
    scenario.splitter = DateSplitter(datetime(2019, 10, 14), True, True)
    scenario.criterion = HitRate()
    scenario.metrics = [NDCG(), Precision()]
    scenario.optuna_max_n_trials = 10
    scenario.fallback_recommender = None

    flag = False
    if flag:
        scenario.recommender = PopularRecommender()
        grid = {"alpha": {"type": "int", "args": [0, 100]},
                "beta": {"type": "int", "args": [0, 100]}}
    else:
        scenario.recommender = KNNRecommender()
        scenario.fallback_recommender = PopularRecommender()
        grid = {"num_neighbours": {"type": "categorical",
                                   "args": [[1]]}}

    best_params_ = scenario.research(grid, log_,
                                     k=2, n_trials=4)

    recs_ = scenario.production(best_params_, log_,
                                users=None, items=None,
                                k=2)

    recs_.show()

    print(scenario.study.trials_dataframe())
