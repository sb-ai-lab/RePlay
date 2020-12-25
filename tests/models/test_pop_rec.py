"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
from parameterized import parameterized
from tests.pyspark_testcase import PySparkTest

from replay.models.pop_rec import PopRec
from replay.constants import LOG_SCHEMA
from datetime import datetime

TEST_DATE = datetime(2019, 1, 1)


class PopRecTestCase(PySparkTest):
    def setUp(self):
        self.model = PopRec()

    @parameterized.expand(
        [
            # users, k, items_relevance
            # проверяем выделение айтемов
            (
                ["u1", "u2", "u3"],
                4,
                [["i1", 2 / 3], ["i2", 2 / 3], ["i3", 2 / 3], ["i4", 1.0]],
            ),
            (
                ["u1", "u2", "u3"],
                3,
                [["i1", 2 / 3], ["i3", 2 / 3], ["i4", 1.0]],
            ),
            (
                ["u1", "u2", "u3"],
                4,
                [["i1", 2 / 3], ["i2", 2 / 3], ["i3", 2 / 3], ["i4", 1.0]],
            ),
            # проверяем выделение юзеров
            (
                ["u1", "u2"],
                4,
                [["i1", 2 / 3], ["i2", 2 / 3], ["i3", 2 / 3], ["i4", 1.0]],
            ),
            # проверяем выделение топ-к
            (["u1", "u2"], 1, [["i4", 1.0]]),
            (["u1", "u3"], 2, [["i2", 2 / 3], ["i4", 1.0]],),
            (["u3", "u2"], 3, [["i1", 2 / 3], ["i3", 2 / 3], ["i4", 1.0]]),
        ]
    )
    def test_popularity_recs(self, users, k, items_relevance):
        self.some_date = datetime(2019, 1, 1)
        log_data = [
            ["u1", "i1", TEST_DATE, 1.0],
            ["u2", "i1", TEST_DATE, 1.0],
            ["u3", "i3", TEST_DATE, 2.0],
            ["u3", "i3", TEST_DATE, 2.0],
            ["u2", "i3", TEST_DATE, 2.0],
            ["u3", "i4", TEST_DATE, 2.0],
            ["u1", "i4", TEST_DATE, 2.0],
            ["u2", "i1", TEST_DATE, 3.0],
            ["u3", "i2", TEST_DATE, 1.0],
            ["u2", "i2", TEST_DATE, 1.0],
            ["u2", "i3", TEST_DATE, 2.0],
            ["u3", "i4", TEST_DATE, 3.0],
            ["u2", "i4", TEST_DATE, 2.0],
            ["u1", "i4", TEST_DATE, 1.0],
        ]
        log = self.spark.createDataFrame(data=log_data, schema=LOG_SCHEMA)
        items_relevance = self.spark.createDataFrame(
            items_relevance, schema=["item_id", "relevance"]
        )
        users = self.spark.createDataFrame(
            data=[[user] for user in users], schema=["user_id"]
        )

        true_recs = users.crossJoin(items_relevance)
        # два вызова нужны, чтобы проверить, что они возващают одно и то же
        test_recs_first = self.model.fit_predict(
            log=log,
            k=k,
            users=users,
            items=items_relevance.select("item_id"),
            filter_seen_items=False,
        )
        print(test_recs_first.schema)
        test_recs_second = self.model.fit_predict(
            log=log,
            k=k,
            users=users,
            items=items_relevance.select("item_id"),
            filter_seen_items=False,
        )
        self.assertSparkDataFrameEqual(true_recs, test_recs_second)
        self.assertSparkDataFrameEqual(true_recs, test_recs_first)
        self.assertSparkDataFrameEqual(test_recs_first, test_recs_second)

    def test_popularity_recs_filter_seen_items(self):
        log_data = [
            ["u1", "i1", TEST_DATE, 1.0],
            ["u1", "i4", TEST_DATE, 2.0],
            ["u2", "i1", TEST_DATE, 1.0],
            ["u2", "i3", TEST_DATE, 2.0],
            ["u3", "i3", TEST_DATE, 2.0],
            ["u3", "i4", TEST_DATE, 1.0],
            ["u3", "i3", TEST_DATE, 2.0],
        ]
        log = self.spark.createDataFrame(data=log_data, schema=LOG_SCHEMA)
        true_recs_data = [
            ["u1", "i3", 2 / 3],
            ["u2", "i4", 2 / 3],
            ["u3", "i1", 2 / 3],
        ]
        true_recs_schema = ["user_id", "item_id", "relevance"]
        true_recs = self.spark.createDataFrame(
            data=true_recs_data, schema=true_recs_schema
        )
        users = self.spark.createDataFrame(
            data=[[user] for user in ["u1", "u2", "u3"]], schema=["user_id"]
        )
        items = self.spark.createDataFrame(
            data=[[item] for item in ["i1", "i2", "i3", "i4"]],
            schema=["item_id"],
        )
        test_recs = self.model.fit_predict(
            log=log, k=1, users=users, items=items
        )
        self.assertSparkDataFrameEqual(true_recs, test_recs)

    def test_popularity_recs_default_items_users(self):
        log_data = [
            ["u2", "i1", TEST_DATE, 1.0],
            ["u3", "i3", TEST_DATE, 2.0],
            ["u1", "i4", TEST_DATE, 2.0],
            ["u1", "i1", TEST_DATE, 1.0],
            ["u3", "i1", TEST_DATE, 2.0],
            ["u2", "i2", TEST_DATE, 1.0],
            ["u2", "i3", TEST_DATE, 3.0],
            ["u3", "i4", TEST_DATE, 2.0],
            ["u1", "i4", TEST_DATE, 2.0],
            ["u3", "i4", TEST_DATE, 4.0],
        ]
        log = self.spark.createDataFrame(data=log_data, schema=LOG_SCHEMA)
        true_recs_data = [
            ["u1", "i1", 1.0],
            ["u1", "i3", 2 / 3],
            ["u1", "i4", 2 / 3],
            ["u1", "i2", 1 / 3],
            ["u2", "i1", 1.0],
            ["u2", "i3", 2 / 3],
            ["u2", "i4", 2 / 3],
            ["u2", "i2", 1 / 3],
            ["u3", "i1", 1.0],
            ["u3", "i3", 2 / 3],
            ["u3", "i4", 2 / 3],
            ["u3", "i2", 1 / 3],
        ]
        true_recs_schema = ["user_id", "item_id", "relevance"]
        true_recs = self.spark.createDataFrame(
            data=true_recs_data, schema=true_recs_schema
        )
        test_recs = self.model.fit_predict(
            log=log, k=4, users=None, items=None, filter_seen_items=False
        )
        self.assertSparkDataFrameEqual(true_recs, test_recs)
