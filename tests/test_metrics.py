"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime
from math import log

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.metrics.metrics import (HitRateMetric, NDCGMetric,
                                              PrecisionMetric,
                                              MAPMetric,
                                              RecallMetric)
from tests.pyspark_testcase import PySparkTest


class TestMetrics(PySparkTest):
    def setUp(self) -> None:
        self.recs = self.spark.createDataFrame(
            data=[["user1", "item1", "day  ", 3.0],
                  ["user1", "item2", "night", 2.0],
                  ["user1", "item3", "night", 1.0],
                  ["user2", "item1", "night", 3.0],
                  ["user2", "item2", "night", 4.0],
                  ["user3", "item1", "day  ", 5.0]],
            schema=REC_SCHEMA)

        self.ground_truth_recs = self.spark.createDataFrame(
            data=[["user1", "item1", datetime(2019, 9, 12), "day  ", 3.0],
                  ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                  ["user1", "item2", datetime(2019, 9, 17), "night", 1.0],
                  ["user2", "item6", datetime(2019, 9, 14), "day  ", 4.0],
                  ["user2", "item1", datetime(2019, 9, 15), "night", 3.0]],
            schema=LOG_SCHEMA)

    def test_hit_rate_at_k(self):
        self.assertEqual(
            HitRateMetric(self.spark)(self.recs, self.ground_truth_recs, 10),
            2 / 3
        )
        self.assertEqual(
            HitRateMetric(self.spark)(self.recs, self.ground_truth_recs, 1),
            1 / 3
        )

    def test_ndcg_at_k(self):
        self.assertEqual(
            NDCGMetric(self.spark)(self.recs, self.ground_truth_recs, 1),
            1 / 2
        )
        self.assertEqual(
            NDCGMetric(self.spark)(self.recs, self.ground_truth_recs, 3),
            1 / 2 * (
                    1 / (1 / log(2) + 1 / log(3) + 1 / log(4)) *
                    (1 / log(2) + 1 / log(3)) +
                    1 / (1 / log(2) + 1 / log(3)) *
                    (1 / log(3))
            )
        )

    def test_precision_at_k(self):
        self.assertAlmostEqual(
            PrecisionMetric(self.spark)(self.recs, self.ground_truth_recs, 3),
            1 / 2
        )
        self.assertEqual(
            PrecisionMetric(self.spark)(self.recs, self.ground_truth_recs, 1),
            1 / 2
        )
        self.assertAlmostEqual(
            PrecisionMetric(self.spark)(self.recs, self.ground_truth_recs, 2),
            3 / 4
        )

    def test_map_at_k(self):
        self.assertAlmostEqual(
            MAPMetric(self.spark)(self.recs, self.ground_truth_recs, 3),
            11 / 24
        )

        self.assertAlmostEqual(
            MAPMetric(self.spark)(self.recs, self.ground_truth_recs, 1),
            1 / 2
        )

    def test_recall_at_k(self):
        self.assertEqual(
            RecallMetric(self.spark)(self.recs, self.ground_truth_recs, 10),
            (1 / 2 + 2 / 3) / 2
        )
        self.assertEqual(
            RecallMetric(self.spark)(self.recs, self.ground_truth_recs, 1),
            1 / 6
        )
