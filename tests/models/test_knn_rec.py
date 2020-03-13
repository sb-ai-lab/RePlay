"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import DEFAULT_CONTEXT, LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.models.knn_rec import KNNRec


class KNNRecTestCase(PySparkTest):
    def setUp(self):
        self.model = KNNRec(1)
        self.some_date = datetime(2019, 1, 1)
        self.log = self.spark.createDataFrame(
            [
                ["u1", "i1", self.some_date, 1.0],
                ["u2", "i1", self.some_date, 1.0],
                ["u3", "i3", self.some_date, 2.0],
                ["u3", "i3", self.some_date, 2.0],
                ["u2", "i3", self.some_date, 2.0],
                ["u3", "i4", self.some_date, 2.0],
                ["u1", "i4", self.some_date, 2.0]
            ],
            schema=LOG_SCHEMA
        )

    def test_fit(self):
        self.model._pre_fit(self.log, None, None)
        self.model._fit(self.log, None, None)
        self.assertSparkDataFrameEqual(
            self.model.similarity,
            self.spark.createDataFrame([
                ("i1", "i4", 0.5),
                ("i3", "i4", 0.18350341907227408),
                ("i4", "i3", 0.18350341907227408)
            ], schema=StructType([
                StructField("item_id_one", StringType()),
                StructField("item_id_two", StringType()),
                StructField("similarity", DoubleType()),
            ]))
        )

    def test_predict(self):
        self.model.fit(self.log, None, None)
        recs = self.model._predict(
            log=self.log,
            k=1,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
            user_features=None, item_features=None
        )
        self.assertSparkDataFrameEqual(
            recs,
            self.spark.createDataFrame(
                [
                    ["u1", "i3", 0.18350341907227408],
                    ["u2", "i4", 0.6835034190722742],
                ],
                schema=REC_SCHEMA
            )
        )

    def test_get_params(self):
        self.assertEqual(
            self.model.get_params(),
            {"shrink": 0.0, "num_neighbours": 1}
        )
