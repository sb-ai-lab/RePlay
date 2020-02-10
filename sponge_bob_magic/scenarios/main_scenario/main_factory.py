"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from datetime import datetime
from typing import List, Optional

from pyspark.sql import SparkSession

from sponge_bob_magic.metrics import HitRate, Metric
from sponge_bob_magic.models.base_recommender import Recommender
from sponge_bob_magic.models.knn_recommender import KNNRecommender
from sponge_bob_magic.models.popular_recomennder import PopularRecommender
from sponge_bob_magic.scenarios.base_factory import ScenarioFactory
from sponge_bob_magic.scenarios.base_scenario import Scenario
from sponge_bob_magic.scenarios.main_scenario.main_scenario import MainScenario
from sponge_bob_magic.splitters.base_splitter import Splitter
from sponge_bob_magic.splitters.log_splitter import (LogSplitByDateSplitter,
                                                     LogSplitRandomlySplitter)


class MainScenarioFactory(ScenarioFactory):
    """ Класс фабрики для простого сценария с замесом. """

    def get(
        self,
        splitter: Optional[Splitter] = None,
        recommender: Optional[Recommender] = None,
        criterion: Optional[Metric] = None,
        metrics: Optional[List[Metric]] = None,
        fallback_recommender: Optional[Recommender] = None
    ) -> Scenario:
        main_scenario = MainScenario()
        main_scenario.splitter = (
            splitter if splitter
            else LogSplitRandomlySplitter(
                    drop_cold_users=True,
                    drop_cold_items=True,
                    test_size=0.3,
                    seed=1234)
        )
        main_scenario.recommender = (
            recommender if recommender
            else PopularRecommender(alpha=0, beta=0)
        )
        main_scenario.criterion = (
            criterion if criterion
            else HitRate()
        )
        main_scenario.metrics = metrics if metrics else []
        main_scenario.fallback_recommender = fallback_recommender
        return main_scenario


if __name__ == "__main__":
    spark_ = (
        SparkSession
        .builder
        .master("local[4]")
        .config("spark.local.dir", os.path.join(os.environ["HOME"], "tmp"))
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

    flag = True
    if flag:
        recommender_ = PopularRecommender()
        grid = {"alpha": {"type": "int", "args": [0, 100]},
                "beta": {"type": "int", "args": [0, 100]}}
    else:
        recommender_ = KNNRecommender()
        grid = {"num_neighbours": {"type": "categorical",
                                   "args": [[1]]}}

    factory = MainScenarioFactory()

    scenario = factory.get(
        splitter=LogSplitByDateSplitter(True, True,
                                        datetime(2019, 10, 14)),
        criterion=None,
        metrics=None,
        recommender=recommender_
    )
    best_params_ = scenario.research(grid, log_,
                                     k=2, n_trials=4)

    recs_ = scenario.production(best_params_, log_,
                                users=None, items=None,
                                k=2)

    recs_.show()
