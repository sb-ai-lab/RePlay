"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime
from math import log, log2

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.metrics.metrics import (HitRateMetric, NDCGMetric,
                                              PrecisionMetric,
                                              MAPMetric,
                                              RecallMetric, Surprisal)
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

        self.recs2 = self.spark.createDataFrame(data=[["user1", "item4", "day  ", 4.0],
                                                      ["user1", "item5", "night", 5.0]],
                                                schema=REC_SCHEMA)

        self.ground_truth_recs = self.spark.createDataFrame(
            data=[["user1", "item1", datetime(2019, 9, 12), "day  ", 3.0],
                  ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                  ["user1", "item2", datetime(2019, 9, 17), "night", 1.0],
                  ["user2", "item6", datetime(2019, 9, 14), "day  ", 4.0],
                  ["user2", "item1", datetime(2019, 9, 15), "night", 3.0]],
            schema=LOG_SCHEMA)

        self.empty = self.spark.createDataFrame(data=[], schema=LOG_SCHEMA)

        self.history = self.spark.createDataFrame(
            data=[["user1", "item1", datetime(2019, 8, 22), "day  ", 4.0],
                  ["user1", "item3", datetime(2019, 8, 23), "night", 3.0],
                  ["user1", "item2", datetime(2019, 8, 27), "night", 2.0],
                  ["user2", "item4", datetime(2019, 8, 24), "day  ", 3.0],
                  ["user2", "item1", datetime(2019, 8, 25), "night", 4.0],
                  ["user3", "item2", datetime(2019, 8, 26), "day", 5.0],
                  ["user3", "item1", datetime(2019, 8, 26), "day", 5.0],
                  ["user3", "item3", datetime(2019, 8, 26), "day", 3.0],
                  ["user4", "item2", datetime(2019, 8, 26), "day", 5.0],
                  ["user4", "item1", datetime(2019, 8, 26), "day", 5.0],
                  ["user4", "item1", datetime(2019, 8, 26), "night", 1.0]],
            schema=LOG_SCHEMA)

        self.items = self.spark.createDataFrame(data=[["item1"],
                                                      ["item2"],
                                                      ["item3"],
                                                      ["item4"],
                                                      ["item5"]],
                                                schema=["item_id"])

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

    def test_surprisal_at_k(self):
        surprisal = Surprisal(self.spark, self.history, self.items)

        self.assertAlmostEqual(
            surprisal(self.recs, self.empty, 1),
            (-log2(0.75)) / 3
        )

        self.assertAlmostEqual(
            surprisal(self.recs, self.empty, 2),
            (-log2(0.75)) / 3
        )

        self.assertAlmostEqual(
            surprisal(self.recs, self.empty, 3),
            ((1 - log2(0.75)) / 3 - log2(0.75) / 2) / 3
        )

        self.assertAlmostEqual(
            surprisal(self.recs2, self.empty, 2),
            2.0
        )

    def test_normalized_surprisal_at_k(self):
        surprisal = Surprisal(self.spark, self.history, self.items, normalize=True)

        self.assertAlmostEqual(
            surprisal(self.recs2, self.empty, 1),
            1.0
        )
