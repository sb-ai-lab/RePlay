"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import REC_SCHEMA
from sponge_bob_magic.metrics import NDCG, HitRate, Precision, Surprisal
from sponge_bob_magic.models.knn import KNN
from sponge_bob_magic.models.pop_rec import PopRec
from sponge_bob_magic.scenarios.main_scenario import MainScenario
from sponge_bob_magic.splitters.log_splitter import DateSplitter


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
        self.scenario = MainScenario()
        self.scenario.splitter = DateSplitter(
            test_start=datetime(2019, 10, 14),
            drop_cold_users=True,
            drop_cold_items=True,
        )
        self.scenario.criterion = HitRate
        self.scenario.metrics = {NDCG: [2], Precision: [2], Surprisal: [2]}
        self.scenario.optuna_max_n_trials = 10
        self.scenario.fallback_rec = None
        self.scenario._optuna_seed = 42

    def test_research_and_production(self):
        self.scenario.recommender = KNN()
        self.scenario.fallback_rec = PopRec()
        grid = {"num_neighbours": {"type": "categorical", "args": [[1, 2, 3, 4, 5]]}}
        best_params = self.scenario.research(grid, self.log, k=2, n_trials=2)
        self.assertEqual(best_params, {"num_neighbours": 4})
        recs = self.scenario.production(
            best_params, self.log, users=None, items=None, k=2
        )
        data = self.spark.createDataFrame(
            [
                ["user2", "item2", 0.3452994616207483],
                ["user3", "item3", 0.6905989232414966],
                ["user3", "item1", 0.0],
                ["user2", "item1", 0.0],
                ["user1", "item1", 0.0],
                ["user1", "item1", 0.0],
            ],
            REC_SCHEMA,
        )
        self.assertSparkDataFrameEqual(recs, data)
