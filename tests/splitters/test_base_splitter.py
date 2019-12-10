"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy

from pyspark_testcase import PySparkTest
from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.splitters.base_splitter import Splitter


class TestSplitter(PySparkTest):
    def test__filter_zero_relevance(self):
        log = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item1", datetime(2019, 9, 12), "day", 0.0],
                ["user4", "item2", datetime(2019, 9, 13), "night", 2.0],
                ["user2", "item4", datetime(2019, 9, 14), "day", 0.0],
                ["user2", "item4", datetime(2019, 9, 14), "day", -10.0],
            ],
            schema=LOG_SCHEMA
        )

        true_filtered_log = self.spark.createDataFrame(
            data=[
                ["user1", "item4", datetime(2019, 9, 12), "day", 1.0],
                ["user4", "item2", datetime(2019, 9, 13), "night", 2.0],
            ],
            schema=LOG_SCHEMA
        )

        test_filtered_log = Splitter(self.spark)._filter_zero_relevance(log)

        self.assertSparkDataFrameEqual(true_filtered_log, test_filtered_log)

    def test__drop_cold_items_and_users(self):
        train = self.spark.createDataFrame(
            data=[
                ["u1", "i4", datetime(2019, 9, 12), "day", 1.0],
                ["u4", "i1", datetime(2019, 9, 12), "day", 0.0],
                ["u4", "i2", datetime(2019, 9, 13), "night", 2.0],
                ["u2", "i3", datetime(2019, 9, 14), "day", 0.0],
                ["u2", "i4", datetime(2019, 9, 14), "day", -10.0],
            ],
            schema=LOG_SCHEMA
        )  # u1, u2, u4 and  i1, i2, i3, i4

        test_data = numpy.array([
            ["u10", "i4", datetime(2019, 9, 12), "day", 1.0],  # 0
            ["u4", "i10", datetime(2019, 9, 12), "day", 0.0],  # 1
            ["u4", "i1", datetime(2019, 9, 13), "night", 2.0],  # 2
            ["u2", "i2", datetime(2019, 9, 13), "night", 2.0],  # 3
            ["u1", "i3", datetime(2019, 9, 13), "night", 2.0],  # 4
            ["u2", "i4", datetime(2019, 9, 14), "day", 0.0],  # 5
            ["u30", "i50", datetime(2019, 9, 14), "day", -10.0],  # 6
            ["u4", "i60", datetime(2019, 9, 14), "day", -10.0],  # 7
        ])

        test = self.spark.createDataFrame(data=test_data.tolist(),
                                          schema=LOG_SCHEMA)

        self.assertSparkDataFrameEqual(
            self.spark.createDataFrame(data=test_data[[2, 3, 4, 5]].tolist(),
                                       schema=LOG_SCHEMA),
            Splitter(self.spark)._drop_cold_items_and_users(
                train, test, drop_cold_items=True, drop_cold_users=True
            )
        )

        self.assertSparkDataFrameEqual(
            self.spark.createDataFrame(
                data=test_data[[1, 2, 3, 4, 5, 7]].tolist(),
                schema=LOG_SCHEMA),
            Splitter(self.spark)._drop_cold_items_and_users(
                train, test, drop_cold_items=False, drop_cold_users=True
            )
        )

        self.assertSparkDataFrameEqual(
            self.spark.createDataFrame(
                data=test_data[[0, 2, 3, 4, 5]].tolist(),
                schema=LOG_SCHEMA),
            Splitter(self.spark)._drop_cold_items_and_users(
                train, test, drop_cold_items=True, drop_cold_users=False
            )
        )

        self.assertSparkDataFrameEqual(
            test,
            Splitter(self.spark)._drop_cold_items_and_users(
                train, test, drop_cold_items=False, drop_cold_users=False
            )
        )
