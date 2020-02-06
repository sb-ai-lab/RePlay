"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime
from math import log, log2

from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.metrics import (MAP, NDCG, HitRate, Metric, Precision,
                                      Recall, Surprisal)


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
        self.recs2 = self.spark.createDataFrame(
            data=[["user1", "item4", "day  ", 4.0],
                  ["user1", "item5", "night", 5.0]],
            schema=REC_SCHEMA)
        self.ground_truth_recs = self.spark.createDataFrame(
            data=[
                ["user1", "item1", datetime(2019, 9, 12), "day  ", 3.0],
                ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user1", "item2", datetime(2019, 9, 17), "night", 1.0],
                ["user2", "item6", datetime(2019, 9, 14), "day  ", 4.0],
                ["user2", "item1", datetime(2019, 9, 15), "night", 3.0],
                ["user3", "item2", datetime(2019, 9, 15), "night", 3.0]
            ],
            schema=LOG_SCHEMA)
        self.empty_df = self.spark.createDataFrame(data=[], schema=LOG_SCHEMA)
        self.log = self.spark.createDataFrame(
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

    def test_hit_rate_at_k(self):
        self.assertEqual(
            HitRate()(self.recs, self.ground_truth_recs, 10),
            2 / 3
        )
        self.assertEqual(
            HitRate()(self.recs, self.ground_truth_recs, 1),
            1 / 3
        )

    def test_ndcg_at_k(self):
        self.assertAlmostEqual(
            NDCG()(self.recs, self.ground_truth_recs, 1),
            1 / 3
        )
        self.assertAlmostEqual(
            NDCG()(self.recs, self.ground_truth_recs, 3),
            1 / 3 * (
                    1 / (1 / log(2) + 1 / log(3) + 1 / log(4)) *
                    (1 / log(2) + 1 / log(3)) +
                    1 / (1 / log(2) + 1 / log(3)) *
                    (1 / log(3))
            )
        )

    def test_precision_at_k(self):
        self.assertAlmostEqual(
            Precision()(self.recs, self.ground_truth_recs, 3),
            1 / 3
        )
        self.assertAlmostEqual(
            Precision()(self.recs, self.ground_truth_recs, 1),
            1 / 3
        )
        self.assertAlmostEqual(
            Precision()(self.recs, self.ground_truth_recs, 2),
            1 / 2
        )

    def test_map_at_k(self):
        self.assertAlmostEqual(
            MAP()(self.recs, self.ground_truth_recs, 3),
            11 / 36
        )

        self.assertAlmostEqual(
            MAP()(self.recs, self.ground_truth_recs, 1),
            1 / 3
        )

    def test_recall_at_k(self):
        self.assertEqual(
            Recall()(self.recs, self.ground_truth_recs, 10),
            (1 / 2 + 2 / 3) / 3
        )
        self.assertEqual(
            Recall()(self.recs, self.ground_truth_recs, 1),
            1 / 9
        )

    def test_surprisal_at_k(self):
        surprisal = Surprisal(self.log)
        surprisal_norm = Surprisal(self.log, normalize=True)

        test_cases = [
            [1, (-log2(0.75)) / 3],
            [2, (-log2(0.75)) / 3],
            [3, ((1 - log2(0.75)) / 3 - log2(0.75) / 2) / 3]
        ]
        for k, correct_value in test_cases:
            with self.subTest(f"recs, k={k}"):
                self.assertAlmostEqual(surprisal(self.recs, self.empty_df, k),
                                       correct_value)

        with self.subTest("recs2, k=2"):
            self.assertAlmostEqual(
                surprisal(self.recs2, self.empty_df, 2),
                2.0
            )

        with self.subTest("Normalized, recs2, k=1"):
            self.assertAlmostEqual(
                surprisal_norm(self.recs2, self.empty_df, 1),
                1.0
            )

    def test_check_users(self):
        class NewMetric(Metric):
            def __str__(self):
                return ""

            def _get_metric_value(self, recommendations, ground_truth, k):
                return 1.0
        test_cases = [
            [True, self.recs, self.ground_truth_recs],
            [False, self.recs, self.log],
            [False, self.log, self.recs]
        ]
        new_metric = NewMetric()
        for correct_value, left, right in test_cases:
            with self.subTest():
                self.assertEqual(
                    new_metric._check_users(left, right),
                    correct_value
                )
