from datetime import datetime

import numpy
from pyspark.sql import functions as sf

from pyspark_testcase import PySparkTest
from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.splitters.user_log_splitter import (
    RandomUserLogSplitter,
    ByTimeUserLogSplitter
)


class TestRandomUserLogSplitter(PySparkTest):
    def setUp(self):
        self.true_train = self.spark.createDataFrame(
            data=[["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                  ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                  ["user2", "item6", datetime(2019, 9, 14), "day", 3.0],
                  ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                  ["user3", "item7", datetime(2019, 9, 17), "night", 1.0],
                  ["user3", "item4", datetime(2019, 9, 12), "day", 1.0],
                  ["user3", "item5", datetime(2019, 9, 13), "night", 2.0],
                  ["user3", "item6", datetime(2019, 9, 14), "day", 3.0],
                  ["user2", "item3", datetime(2019, 9, 15), "night", 4.0],
                  ["user1", "item7", datetime(2019, 9, 17), "night", 1.0],
                  ["user2", "item4", datetime(2019, 9, 12), "day", 1.0],
                  ["user2", "item5", datetime(2019, 9, 13), "night", 2.0],
                  ["user3", "item2", datetime(2019, 9, 14), "day", 3.0],
                  ["user4", "item2", datetime(2019, 9, 15), "night", 4.0],
                  ["user4", "item1", datetime(2019, 9, 15), "night", 4.0],
                  ["user4", "item4", datetime(2019, 9, 15), "night", 4.0],
                  ["user1", "item6", datetime(2019, 9, 17), "night", 1.0]],
            schema=LOG_SCHEMA)

    def test_split_quantity(self):
        train, predict_input, test = (
            RandomUserLogSplitter(self.spark, test_size=2, seed=1234)
                .split(log=self.true_train,
                       drop_cold_items=False,
                       drop_cold_users=False)
        )
        self.assertSparkDataFrameEqual(test.union(train), self.true_train)
        self.assertSparkDataFrameEqual(test.union(predict_input),
                                       self.true_train)

        self.assertEqual(test.intersect(train).count(), 0)
        self.assertEqual(test.intersect(predict_input).count(), 0)

        self.assertSetEqual(
            set(test.select("user_id").distinct().collect()),
            set(train.select("user_id").distinct().collect())
        )

        num_items_per_user = numpy.mean(
            test.groupBy("user_id").count().select("count").collect()
        )
        self.assertEqual(num_items_per_user, 2)

    def test_split_proportion(self):
        train, predict_input, test = (
            RandomUserLogSplitter(self.spark, test_size=0.5, seed=1234)
                .split(log=self.true_train,
                       drop_cold_items=False,
                       drop_cold_users=False)
        )
        self.assertSparkDataFrameEqual(train.union(test), self.true_train)
        self.assertSparkDataFrameEqual(predict_input.union(test),
                                       self.true_train)

        self.assertEqual(test.intersect(train).count(), 0)
        self.assertEqual(test.intersect(predict_input).count(), 0)

        self.assertSetEqual(
            set(test.select("user_id").distinct().collect()),
            set(train.select("user_id").distinct().collect())
        )


class TestByTimeUserLogSplitter(PySparkTest):
    def setUp(self):
        self.true_train = self.spark.createDataFrame(
            data=[["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                  ["user1", "item5", datetime(2019, 9, 13), "night", 2.0],
                  ["user2", "item6", datetime(2019, 9, 14), "day", 3.0],
                  ["user2", "item2", datetime(2019, 9, 15), "night", 4.0],
                  ["user3", "item7", datetime(2019, 9, 17), "night", 1.0],
                  ["user3", "item4", datetime(2019, 9, 12), "day", 1.0],
                  ["user3", "item5", datetime(2019, 9, 13), "night", 2.0],
                  ["user3", "item6", datetime(2019, 9, 14), "day", 3.0],
                  ["user2", "item3", datetime(2019, 9, 15), "night", 4.0],
                  ["user1", "item7", datetime(2019, 9, 17), "night", 1.0],
                  ["user2", "item4", datetime(2019, 9, 12), "day", 1.0],
                  ["user2", "item5", datetime(2019, 9, 13), "night", 2.0],
                  ["user3", "item2", datetime(2019, 9, 14), "day", 3.0],
                  ["user4", "item2", datetime(2019, 9, 15), "night", 4.0],
                  ["user4", "item1", datetime(2019, 9, 15), "night", 4.0],
                  ["user4", "item4", datetime(2019, 9, 15), "night", 4.0],
                  ["user1", "item6", datetime(2019, 9, 17), "night", 1.0]],
            schema=LOG_SCHEMA)

    def test_split_quantity(self):
        train, test_input, test = (
            ByTimeUserLogSplitter(self.spark, test_size=2)
            .split(log=self.true_train,
                   drop_cold_items=False,
                   drop_cold_users=False)
        )

        self.assertSparkDataFrameEqual(test.union(train), self.true_train)
        self.assertSparkDataFrameEqual(test.union(test_input), self.true_train)

        self.assertEqual(test.intersect(train).count(), 0)
        self.assertEqual(test.intersect(test_input).count(), 0)

        self.assertSetEqual(
            set(test.select("user_id").distinct().collect()),
            set(train.select("user_id").distinct().collect())
        )

        num_items_per_user = numpy.mean(
            test.groupBy("user_id").count().select("count").collect()
        )
        self.assertEqual(num_items_per_user, 2)

        self.assertEqual(all([x <= y for x, y in zip(
            train
            .orderBy("user_id")
            .groupBy("user_id")
            .agg(sf.max("timestamp"))
            .select("max(timestamp)")
            .collect(),
            test
            .orderBy("user_id")
            .groupBy("user_id")
            .agg(sf.min("timestamp"))
            .select("min(timestamp)")
            .collect(),
        )]), True)

    def test_split_proportion(self):
        train, test_input, test = (
            ByTimeUserLogSplitter(self.spark, test_size=0.5)
            .split(log=self.true_train,
                   drop_cold_items=False,
                   drop_cold_users=False)
        )

        self.assertSparkDataFrameEqual(test.union(train), self.true_train)
        self.assertSparkDataFrameEqual(test.union(test_input), self.true_train)

        self.assertEqual(test.intersect(train).count(), 0)
        self.assertEqual(test.intersect(test_input).count(), 0)

        self.assertSetEqual(
            set(test.select("user_id").distinct().collect()),
            set(train.select("user_id").distinct().collect())
        )

        self.assertEqual(all([x <= y for x, y in zip(
            train
            .orderBy("user_id")
            .groupBy("user_id")
            .agg(sf.max("timestamp"))
            .select("max(timestamp)")
            .collect(),
            test
            .orderBy("user_id")
            .groupBy("user_id")
            .agg(sf.min("timestamp"))
            .select("min(timestamp)")
            .collect(),
        )]), True)
