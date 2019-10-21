"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
from datetime import datetime

from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from sponge_bob_magic.constants import DEFAULT_CONTEXT, LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.models.knn_recommender import KNNRecommender
from tests.pyspark_testcase import PySparkTest


class KNNRecommenderTestCase(PySparkTest):
    def setUp(self):
        self.model = KNNRecommender(self.spark, 1)
        self.some_date = datetime(2019, 1, 1)
        self.log = self.spark.createDataFrame(
            [
                ["u1", "i1", self.some_date, "c1", 1.0],
                ["u2", "i1", self.some_date, "c1", 1.0],
                ["u3", "i3", self.some_date, "c1", 2.0],
                ["u3", "i3", self.some_date, "c1", 2.0],
                ["u2", "i3", self.some_date, "c1", 2.0],
                ["u3", "i4", self.some_date, "c1", 2.0],
                ["u1", "i4", self.some_date, "c1", 2.0]
            ],
            schema=LOG_SCHEMA
        )

    def test_fit(self):
        self.model._pre_fit(self.log, None, None)
        self.model._fit_partial(self.log, None, None)
        self.assertSparkDataFrameEqual(
            self.model.similarity,
            self.spark.createDataFrame([
                ("i1", "i3", 0.408248),
                ("i3", "i1", 0.408248),
                ("i4", "i1", 1 / 2)
            ], schema=StructType([
                StructField("item_id_one", StringType()),
                StructField("item_id_two", StringType()),
                StructField("similarity", DoubleType()),
            ]))
        )

    def test_predict(self):
        self.model.fit(self.log, None, None)
        recs = self.model._predict(
            k=1,
            log=self.log,
            user_features=None,
            item_features=None,
            context=DEFAULT_CONTEXT,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct()
        )
        self.assertSparkDataFrameEqual(
            recs,
            self.spark.createDataFrame(
                [
                    ["u1", "i3", DEFAULT_CONTEXT, 0.408248],
                    ["u3", "i1", DEFAULT_CONTEXT, 1.316497],
                ],
                schema=REC_SCHEMA
            )
        )

    def test_get_params(self):
        self.assertEqual(
            self.model.get_params(),
            {"shrink": 0.0, "k": 1}
        )
