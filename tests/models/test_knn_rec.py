"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
from datetime import datetime

from pyspark.sql.types import DoubleType, StructField, StructType
from tests.pyspark_testcase import PySparkTest

from replay.constants import LOG_SCHEMA, REC_SCHEMA
from replay.models.knn import KNN


class KNNRecTestCase(PySparkTest):
    def setUp(self):
        self.model = KNN(1)
        self.some_date = datetime(2019, 1, 1)
        self.log = self.spark.createDataFrame(
            [
                ["u1", "i1", self.some_date, 1.0],
                ["u2", "i1", self.some_date, 1.0],
                ["u3", "i3", self.some_date, 2.0],
                ["u3", "i3", self.some_date, 2.0],
                ["u2", "i3", self.some_date, 2.0],
                ["u3", "i4", self.some_date, 2.0],
                ["u1", "i4", self.some_date, 2.0],
            ],
            schema=LOG_SCHEMA,
        )

    def test_fit(self):
        self.model.fit(self.log)
        self.assertSparkDataFrameEqual(
            self.model.similarity,
            self.spark.createDataFrame(
                [
                    (1.0, 2.0, 0.5),
                    (0.0, 2.0, 0.18350341907227408),
                    (2.0, 0.0, 0.18350341907227408),
                ],
                schema=StructType(
                    [
                        StructField("item_id_one", DoubleType()),
                        StructField("item_id_two", DoubleType()),
                        StructField("similarity", DoubleType()),
                    ]
                ),
            ),
        )

    def test_predict(self):
        self.model.fit(self.log)
        recs = self.model.predict(
            log=self.log,
            k=1,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
        )
        self.assertSparkDataFrameEqual(
            recs,
            self.spark.createDataFrame(
                [
                    ["u1", "i3", 0.18350341907227408],
                    ["u2", "i4", 0.6835034190722742],
                    ["u3", "i3", 0.0],
                ],
                schema=REC_SCHEMA,
            ),
        )
