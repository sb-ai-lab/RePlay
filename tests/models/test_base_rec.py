"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
from typing import Dict, Optional

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType
from tests.pyspark_testcase import PySparkTest

from replay.models.base_rec import Recommender


class RecTestCase(PySparkTest):
    class DerivedRec(Recommender):
        def _fit(
            self,
            log: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
        ) -> None:
            pass

        def _predict(
            self,
            log: DataFrame,
            k: int,
            users: DataFrame,
            items: DataFrame,
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            filter_seen_items: bool = True,
        ) -> DataFrame:
            pass

    def setUp(self):
        self.model = self.DerivedRec()
        self.empty_df = self.spark.createDataFrame(
            data=[], schema=StructType([])
        )
        self.log = self.spark.createDataFrame(
            data=[["1", "2", "3", "4"]],
            schema=["item_id", "user_id", "timestamp", "relevance"],
        )

    def test_extract_if_needed(self):
        log = self.spark.createDataFrame(data=[[1], [2], [3]], schema=["test"])

        for array in [log, None, [1, 2, 2, 3]]:
            with self.subTest():
                self.assertSparkDataFrameEqual(
                    log, self.model._extract_unique(log, array, "test")
                )

    def test_users_count(self):
        model = self.DerivedRec()
        with self.assertRaises(AttributeError):
            model.users_count
        model.fit(self.log)
        self.assertEqual(model.users_count, 1)

    def test_items_count(self):
        model = self.DerivedRec()
        with self.assertRaises(AttributeError):
            model.items_count
        model.fit(self.log)
        self.assertEqual(model.items_count, 1)

    def test_str(self):
        self.assertEqual(str(self.model), "DerivedRec")
