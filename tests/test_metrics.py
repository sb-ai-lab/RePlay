from datetime import datetime

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.metrics.metrics import Metrics

from pyspark_testcase import PySparkTest


class TestMetrics(PySparkTest):
    def test_hit_rate_at_k(self):
        metrics = Metrics()
        recommendations = self.spark.createDataFrame(
            data=[
                ["user1", "item1", "day", 1.0],
                ["user1", "item2", "night", 2.0],
                ["user2", "item1", "night", 4.0],
                ["user2", "item2", "night", 3.0],
                ["user3", "item1", "day", 5.0],
                ["user1", "item3", "night", 1.0]
            ],
            schema=REC_SCHEMA
        )
        ground_truth = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item6", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item7", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertEqual(
            metrics.hit_rate_at_k(recommendations, ground_truth, 10),
            1 / 3
        )
        self.assertEqual(
            metrics.hit_rate_at_k(recommendations, ground_truth, 1),
            0.0
        )

    def test_ndcg_at_k(self):
        metrics = Metrics()
        recommendations = self.spark.createDataFrame(
            data=[
                ["user1", "item1", "day", 1.0],
                ["user1", "item2", "night", 2.0],
                ["user2", "item1", "night", 4.0],
                ["user2", "item2", "night", 3.0],
                ["user3", "item1", "day", 5.0],
                ["user1", "item3", "night", 1.0]
            ],
            schema=REC_SCHEMA
        )
        ground_truth = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item6", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item7", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertEqual(
            metrics.ndcg_at_k(recommendations, ground_truth, 1),
            0.0
        )
        self.assertEqual(
            metrics.ndcg_at_k(recommendations, ground_truth, 3),
            0.19342640361727076
        )

    def test_precision_at_k(self):
        metrics = Metrics()
        recommendations = self.spark.createDataFrame(
            data=[
                ["user1", "item1", "day", 1.0],
                ["user1", "item2", "night", 2.0],
                ["user2", "item1", "night", 4.0],
                ["user2", "item2", "night", 3.0],
                ["user3", "item1", "day", 5.0],
                ["user1", "item3", "night", 1.0]
            ],
            schema=REC_SCHEMA
        )
        ground_truth = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item6", datetime(2019, 9, 14), "day", 3.0],
                ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                ["user1", "item7", datetime(2019, 9, 17), "night", 1.0]
            ],
            schema=LOG_SCHEMA
        )
        self.assertAlmostEqual(
            metrics.precision_at_k(recommendations, ground_truth, 3),
            1 / 9
        )
        self.assertEqual(
            metrics.precision_at_k(recommendations, ground_truth, 1),
            0.0
        )
        self.assertAlmostEqual(
            metrics.precision_at_k(recommendations, ground_truth, 2),
            1 / 6
        )