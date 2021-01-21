"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: skip-file
from datetime import datetime
from math import log2

from tests.pyspark_testcase import PySparkTest

from replay.constants import LOG_SCHEMA, REC_SCHEMA
from replay.metrics import (
    Coverage,
    HitRate,
    MAP,
    MRR,
    NDCG,
    Precision,
    Recall,
    RocAuc,
    Surprisal,
)
from replay.distributions import item_distribution


class TestMetrics(PySparkTest):
    def setUp(self) -> None:
        self.quality_metrics = (
            HitRate,
            MAP,
            MRR,
            NDCG,
            Precision,
            Recall,
            RocAuc,
        )
        self.recs = self.spark.createDataFrame(
            data=[
                ["user1", "item1", 3.0],
                ["user1", "item2", 2.0],
                ["user1", "item3", 1.0],
                ["user2", "item1", 3.0],
                ["user2", "item2", 4.0],
                ["user2", "item5", 1.0],
                ["user3", "item1", 5.0],
                ["user3", "item3", 1.0],
                ["user3", "item4", 2.0],
            ],
            schema=REC_SCHEMA,
        )
        self.recs2 = self.spark.createDataFrame(
            data=[["user1", "item4", 4.0], ["user1", "item5", 5.0]],
            schema=REC_SCHEMA,
        )
        self.ground_truth_recs = self.spark.createDataFrame(
            data=[
                ["user1", "item1", datetime(2019, 9, 12), 3.0],
                ["user1", "item5", datetime(2019, 9, 13), 2.0],
                ["user1", "item2", datetime(2019, 9, 17), 1.0],
                ["user2", "item6", datetime(2019, 9, 14), 4.0],
                ["user2", "item1", datetime(2019, 9, 15), 3.0],
                ["user3", "item2", datetime(2019, 9, 15), 3.0],
            ],
            schema=LOG_SCHEMA,
        )
        self.log2 = self.spark.createDataFrame(
            data=[
                ["user1", "item1", datetime(2019, 9, 12), 3.0],
                ["user1", "item5", datetime(2019, 9, 13), 2.0],
                ["user1", "item2", datetime(2019, 9, 17), 1.0],
                ["user2", "item6", datetime(2019, 9, 14), 4.0],
                ["user2", "item1", datetime(2019, 9, 15), 3.0],
                ["user3", "item2", datetime(2019, 9, 15), 3.0],
            ],
            schema=LOG_SCHEMA,
        )
        self.log = self.spark.createDataFrame(
            data=[
                ["user1", "item1", datetime(2019, 8, 22), 4.0],
                ["user1", "item3", datetime(2019, 8, 23), 3.0],
                ["user1", "item2", datetime(2019, 8, 27), 2.0],
                ["user2", "item4", datetime(2019, 8, 24), 3.0],
                ["user2", "item1", datetime(2019, 8, 25), 4.0],
                ["user3", "item2", datetime(2019, 8, 26), 5.0],
                ["user3", "item1", datetime(2019, 8, 26), 5.0],
                ["user3", "item3", datetime(2019, 8, 26), 3.0],
                ["user4", "item2", datetime(2019, 8, 26), 5.0],
                ["user4", "item1", datetime(2019, 8, 26), 5.0],
                ["user4", "item1", datetime(2019, 8, 26), 1.0],
            ],
            schema=LOG_SCHEMA,
        )

    def test_hit_rate_at_k(self):
        self.assertDictAlmostEqual(
            HitRate()(self.recs, self.ground_truth_recs, [3, 1]),
            {3: 2 / 3, 1: 1 / 3},
        )

    def test_user_dist(self):
        self.assertListEqual(
            HitRate()
            .user_distribution(self.log, self.recs, self.ground_truth_recs, 1)[
                "value"
            ]
            .to_list(),
            [0.0, 0.5],
        )

    def test_item_dist(self):
        self.assertListEqual(
            item_distribution(self.log, self.recs, 1)["rec_count"].to_list(),
            [0, 0, 1, 2],
        )

    def test_ndcg_at_k(self):
        pred = [300, 200, 100]
        k_set = [1, 2, 3]
        user_id = 1
        ground_truth = [200, 400]
        ndcg_value = 1 / log2(3) / (1 / log2(2) + 1 / log2(3))
        self.assertEqual(
            NDCG()._get_metric_value_by_user_all_k(
                k_set, user_id, pred, ground_truth
            ),
            [(1, 0, 1), (1, ndcg_value, 2), (1, ndcg_value, 3)],
        )
        self.assertDictAlmostEqual(
            NDCG()(self.recs, self.ground_truth_recs, [1, 3]),
            {
                1: 1 / 3,
                3: 1
                / 3
                * (
                    1
                    / (1 / log2(2) + 1 / log2(3) + 1 / log2(4))
                    * (1 / log2(2) + 1 / log2(3))
                    + 1 / (1 / log2(2) + 1 / log2(3)) * (1 / log2(3))
                ),
            },
        )

    def test_precision_at_k(self):
        self.assertDictAlmostEqual(
            Precision()(self.recs, self.ground_truth_recs, [1, 2, 3]),
            {3: 1 / 3, 1: 1 / 3, 2: 1 / 2},
        )

    def test_map_at_k(self):
        print(MAP()(self.recs, self.ground_truth_recs, [1, 3]))
        self.assertDictAlmostEqual(
            MAP()(self.recs, self.ground_truth_recs, [1, 3]),
            {3: 11 / 36, 1: 1 / 3},
        )

    def test_recall_at_k(self):
        self.assertDictAlmostEqual(
            Recall()(self.recs, self.ground_truth_recs, [1, 3]),
            {3: (1 / 2 + 2 / 3) / 3, 1: 1 / 9},
        )

    def test_surprisal_at_k(self):
        self.assertDictAlmostEqual(
            Surprisal(self.log2)(self.recs2, [1, 2]), {1: 1.0, 2: 1.0}
        )

        self.assertAlmostEqual(
            Surprisal(self.log2)(self.recs, 3),
            5 * (1 - 1 / log2(3)) / 9 + 4 / 9,
        )

    def test_coverage(self):
        coverage = Coverage(
            self.recs.union(self.ground_truth_recs.drop("timestamp"))
        )
        self.assertDictAlmostEqual(
            coverage(self.recs, [1, 3]),
            {1: 0.3333333333333333, 3: 0.8333333333333334},
        )

    def test_bad_coverage(self):
        self.assertEqual(Coverage(self.ground_truth_recs)(self.recs, 3), 1.25)

    def test_empty_recs(self):
        for metric_class in self.quality_metrics:
            self.assertEqual(
                metric_class._get_metric_value_by_user(
                    k=4, pred=[], ground_truth=[2, 4]
                ),
                0,
                metric_class(),
            )

    def test_bad_recs(self):
        for metric_class in self.quality_metrics:
            self.assertEqual(
                metric_class._get_metric_value_by_user(
                    k=4, pred=[1, 3], ground_truth=[2, 4]
                ),
                0,
                metric_class(),
            )

    def test_not_full_recs(self):
        for metric_class in self.quality_metrics:
            self.assertEqual(
                metric_class._get_metric_value_by_user(
                    k=4, pred=[4, 1, 2], ground_truth=[1, 4]
                ),
                metric_class._get_metric_value_by_user(
                    k=3, pred=[4, 1, 2], ground_truth=[2, 4]
                ),
                metric_class(),
            )
