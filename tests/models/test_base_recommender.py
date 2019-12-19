"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
import unittest
from typing import Dict, Iterable, Optional

from pyspark.sql import DataFrame
from pyspark.sql.types import StructType
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.models.base_recommender import BaseRecommender


class BaseRecommenderCase(PySparkTest):
    def setUp(self):
        class DerivedRecommender(BaseRecommender):
            predict_count: int

            def _pre_fit(self, log: DataFrame,
                         user_features: Optional[DataFrame],
                         item_features: Optional[DataFrame],
                         path: Optional[str] = None) -> None:
                pass

            def _fit_partial(self, log: DataFrame,
                             user_features: Optional[DataFrame],
                             item_features: Optional[DataFrame],
                             path: Optional[str] = None) -> None:
                pass

            def _predict(self,
                         k: int,
                         users: Iterable or DataFrame,
                         items: Iterable or DataFrame,
                         context: str,
                         log: DataFrame,
                         user_features: Optional[DataFrame],
                         item_features: Optional[DataFrame],
                         to_filter_seen_items: bool = True,
                         path: Optional[str] = None) -> DataFrame:
                if path:
                    (
                        self.spark.createDataFrame([(1,)])
                        .write.parquet(os.path.join(path, "recs.parquet"))
                    )
                    self.predict_count += 1

            def get_params(self) -> Dict[str, object]:
                pass

        self.model = DerivedRecommender(self.spark)
        self.empty_df = self.spark.createDataFrame(data=[],
                                                   schema=StructType([]))

    def test_fit_predict_log_exception(self):
        # log is None
        self.assertRaises(ValueError, self.model.fit_predict,
                          log=None, user_features=None, item_features=None,
                          k=10, users=None, items=None, context=None)

        # log пустой
        self.assertRaises(ValueError, self.model.fit_predict,
                          log=self.empty_df, user_features=None,
                          item_features=None,
                          k=10, users=None, items=None, context=None)

        # log с недостающими колонкнами
        log_required_columns = ["item_id", "user_id", "timestamp",
                                "relevance", "context"]
        for i in range(1, len(log_required_columns) - 1):
            log = self.spark.createDataFrame(
                data=[["1", "2", "3", "4", "5"]],
                schema=log_required_columns[:i] + log_required_columns[i + 1:]
            )
            self.assertRaises(ValueError, self.model.fit_predict,
                              log=log, user_features=None, item_features=None,
                              k=10, users=None, items=None, context=None)

    def test_fit_predict_feature_exception(self):
        log = self.spark.createDataFrame(
            data=[["1", "2", "3", "4", "5", "6"]],
            schema=["item_id", "user_id", "timestamp", "relevance", "context"])

        # user_features пустой | item_features пустой
        self.assertRaises(ValueError, self.model.fit_predict,
                          log=log, user_features=self.empty_df,
                          item_features=None,
                          k=10, users=None, items=None, context=None)
        self.assertRaises(ValueError, self.model.fit_predict,
                          log=log, user_features=None,
                          item_features=self.empty_df,
                          k=10, users=None, items=None, context=None)

        # в user_features | item_features не достает колонок
        self.assertRaises(
            ValueError, self.model.fit_predict,
            log=log, item_features=None,
            user_features=self.spark.createDataFrame(data=[["1", "2"]],
                                                     schema=["user_id", "f"]),
            k=10, users=None, items=None, context=None,
        )
        self.assertRaises(
            ValueError, self.model.fit_predict,
            log=log, user_features=None,
            item_features=self.spark.createDataFrame(data=[["1", "2"]],
                                                     schema=["item_id", "f"]),
            k=10, users=None, items=None, context=None
        )

        # в user_features | item_features не достает колонок с фичами
        self.assertRaises(
            ValueError, self.model.fit_predict,
            log=log, item_features=None,
            k=10, users=None, items=None, context=None,
            user_features=self.spark.createDataFrame(
                data=[["1", "2"]],
                schema=["user_id", "timestamp"])
        )
        self.assertRaises(
            ValueError, self.model.fit_predict,
            log=log, item_features=None,
            k=10, users=None, items=None, context=None,
            user_features=self.spark.createDataFrame(
                data=[["1", "2"]],
                schema=["item_id", "timestamp"])
        )

    def test_batch_predict(self):
        log = self.spark.createDataFrame(
            data=[["1", "2", "3", "4", "5"], ["a", "b", "3", "4", "5"]],
            schema=["item_id", "user_id", "timestamp", "relevance", "context"])
        with self.assertRaises(ValueError):
            self.model.predict(
                log=log, user_features=None, item_features=None,
                k=1, users=None, items=None, context=None, batch_size=1
            )
        self.model.predict_count = 0
        self.model.predict(
            log=log, user_features=None, item_features=None, k=1, users=None,
            items=None, context=None, batch_size=1,
            path=os.path.join(os.environ["HOME"], "models/test_model")
        )
        self.assertEqual(self.model.predict_count, 2)


if __name__ == "__main__":
    unittest.main()
