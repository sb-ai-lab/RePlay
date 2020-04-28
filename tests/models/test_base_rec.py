"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Iterable, Optional, Union

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.models.base_rec import Recommender


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

        def get_params(self) -> Dict[str, object]:
            return {"alpha": 1, "beta": 2}

    def setUp(self):
        self.model = self.DerivedRec()
        self.empty_df = self.spark.createDataFrame(data=[], schema=StructType([]))
        self.log = self.spark.createDataFrame(
            data=[["1", "2", "3", "4"]],
            schema=["item_id", "user_id", "timestamp", "relevance"],
        )

    def test_fit_predict_feature_exception(self):
        # user_features пустой | item_features пустой
        self.assertRaises(
            ValueError,
            self.model.fit_predict,
            log=self.log,
            user_features=self.empty_df,
            item_features=None,
            k=10,
            users=None,
            items=None,
        )
        self.assertRaises(
            ValueError,
            self.model.fit_predict,
            log=self.log,
            user_features=None,
            item_features=self.empty_df,
            k=10,
            users=None,
            items=None,
        )
        # в user_features | item_features не достает колонок
        self.assertRaises(
            ValueError,
            self.model.fit_predict,
            log=self.log,
            item_features=None,
            user_features=self.spark.createDataFrame(
                data=[["1", "2"]], schema=["user_id", "f"]
            ),
            k=10,
            users=None,
            items=None,
        )
        self.assertRaises(
            ValueError,
            self.model.fit_predict,
            log=self.log,
            user_features=None,
            item_features=self.spark.createDataFrame(
                data=[["1", "2"]], schema=["item_id", "f"]
            ),
            k=10,
            users=None,
            items=None,
        )

        # в user_features | item_features не достает колонок с фичами
        self.assertRaises(
            ValueError,
            self.model.fit_predict,
            log=self.log,
            item_features=None,
            k=10,
            users=None,
            items=None,
            user_features=self.spark.createDataFrame(
                data=[["1", "2"]], schema=["user_id", "timestamp"]
            ),
        )
        self.assertRaises(
            ValueError,
            self.model.fit_predict,
            log=self.log,
            item_features=None,
            k=10,
            users=None,
            items=None,
            user_features=self.spark.createDataFrame(
                data=[["1", "2"]], schema=["item_id", "timestamp"]
            ),
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
        model._pre_fit(self.log)
        self.assertEqual(model.users_count, 1)

    def test_items_count(self):
        model = self.DerivedRec()
        with self.assertRaises(AttributeError):
            model.items_count
        model._pre_fit(self.log)
        self.assertEqual(model.items_count, 1)

    def test_repr(self):
        self.assertEqual(repr(self.model), "DerivedRec(alpha=1, beta=2)")

    def test_str(self):
        self.assertEqual(str(self.model), "DerivedRec")
