"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

from tests.pyspark_testcase import PySparkTest

from replay.constants import REC_SCHEMA
from replay.metrics import NDCG, HitRate, Precision, Surprisal
from replay.models.knn import KNN
from replay.models.pop_rec import PopRec
from replay.scenarios.main_scenario import MainScenario
from replay.splitters.log_splitter import DateSplitter


class MainScenarioTestCase(PySparkTest):
    def setUp(self):
        data = [
            ["user1", "item1", 1.0, datetime(2019, 10, 8)],
            ["user1", "item2", 2.0, datetime(2019, 10, 9)],
            ["user1", "item3", 1.0, datetime(2019, 10, 10)],
            ["user2", "item1", 1.0, datetime(2019, 10, 11)],
            ["user2", "item3", 1.0, datetime(2019, 10, 12)],
            ["user3", "item2", 1.0, datetime(2019, 10, 13)],
            ["user3", "item1", 1.0, datetime(2019, 10, 14)],
            ["user1", "item1", 1.0, datetime(2019, 10, 15)],
            ["user1", "item2", 1.0, datetime(2019, 10, 16)],
            ["user2", "item3", 2.0, datetime(2019, 10, 17)],
            ["user3", "item2", 2.0, datetime(2019, 10, 18)],
        ]
        schema = ["user_id", "item_id", "relevance", "timestamp"]
        self.log = self.spark.createDataFrame(data=data, schema=schema)
        splitter = DateSplitter(
            test_start=datetime(2019, 10, 14),
            drop_cold_users=True,
            drop_cold_items=True,
        )
        self.scenario = MainScenario(
            splitter, metrics={NDCG: [2], Precision: [2], Surprisal: 2},
        )

    def test_research_and_production(self):
        grid = {"rank": [1]}
        best_params = self.scenario.research(grid, self.log, k=2, n_trials=1)
        self.assertEqual(best_params, {"rank": 1})
        recs = self.scenario.production(best_params, self.log, k=2)
        self.assertEqual(recs.count(), 6)
