"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime
from math import log2

from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.metrics import (MAP, NDCG, Coverage, HitRate, Metric,
                                      Precision, Recall, Surprisal)


class TestMetrics(PySparkTest):
    def setUp(self) -> None:
        self.recs = self.spark.createDataFrame(
            data=[["user1", "item1", "day  ", 3.0],
                  ["user1", "item2", "night", 2.0],
                  ["user1", "item3", "night", 1.0],
                  ["user2", "item1", "night", 3.0],
                  ["user2", "item2", "night", 4.0],
                  ["user2", "item5", "night", 1.0],
                  ["user3", "item1", "day  ", 5.0],
                  ["user3", "item3", "day  ", 1.0],
                  ["user3", "item4", "night", 2.0]],
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
        self.log2 = self.spark.createDataFrame(
            data=[
                ["user1", "item1", datetime(2019, 9, 12), "day  ", 3.0],
                ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                ["user1", "item2", datetime(2019, 9, 17), "night", 1.0],
                ["user2", "item6", datetime(2019, 9, 14), "day  ", 4.0],
                ["user2", "item1", datetime(2019, 9, 15), "night", 3.0],
                ["user3", "item2", datetime(2019, 9, 15), "night", 3.0]
            ],
            schema=LOG_SCHEMA)
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
        self.assertDictAlmostEqual(
            HitRate()(self.recs, self.ground_truth_recs, [3, 1]),
            {3: 2 / 3, 1: 1 / 3}
        )

    def test_ndcg_at_k(self):
        self.assertDictAlmostEqual(
            NDCG()(self.recs, self.ground_truth_recs, [1, 3]),
            {1: 1 / 3,
             3: 1 / 3 * (
                 1 / (1 / log2(2) + 1 / log2(3) + 1 / log2(4)) *
                 (1 / log2(2) + 1 / log2(3)) +
                 1 / (1 / log2(2) + 1 / log2(3)) *
                 (1 / log2(3))
             )}
        )

    def test_precision_at_k(self):
        self.assertDictAlmostEqual(
            Precision()(self.recs, self.ground_truth_recs, [1, 2, 3]),
            {3: 1 / 3,
             1: 1 / 3,
             2: 1 / 2}
        )

    def test_map_at_k(self):
        self.assertDictAlmostEqual(
            MAP()(self.recs, self.ground_truth_recs, [1, 3]),
            {3: 7 / 12,
             1: 1 / 3}
        )

    def test_recall_at_k(self):
        self.assertDictAlmostEqual(
            Recall()(self.recs, self.ground_truth_recs, [1, 3]),
            {3: (1 / 2 + 2 / 3) / 3,
             1: 1 / 9}
        )

    def test_surprisal_at_k(self):
        self.assertDictAlmostEqual(
            Surprisal(self.log2)(self.recs2, [1, 2]),
            {1: 1.0,
             2: 1.0}
        )

        self.assertAlmostEqual(
            Surprisal(self.log2)(self.recs, 3),
            5 * (1 - 1 / log2(3)) / 9 + 4 / 9
        )

    def test_check_users(self):
        class NewMetric(Metric):
            def __str__(self):
                return ""

            def _get_metric_value(self, recommendations, ground_truth, k):
                return 1.0

            @staticmethod
            def _get_metric_value_by_user(pdf):
                return pdf

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

    def test_coverage(self):
        coverage = Coverage(self.recs.union(
            self.ground_truth_recs.drop("timestamp")
        ))
        self.assertDictAlmostEqual(
            coverage(self.recs, [1, 3]),
            {1: 0.3333333333333333, 3: 0.8333333333333334}
        )
        self.assertEqual(
            Coverage(self.ground_truth_recs)(self.recs, 3),
            1.25
        )
