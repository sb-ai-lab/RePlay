"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
from datetime import datetime

from pyspark.sql.types import DoubleType, StructField, StructType
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.models.knn import KNN


class KNNRecTestCase(PySparkTest):
    def setUp(self):
        self.model = KNN(1)
        self.some_date = datetime(2019, 1, 1)
        self.log = self.spark.createDataFrame(
            [
                ["u1", "i1", self.some_date, 1.0],
                ["u2", "i2", self.some_date, 1.0],
                ["u3", "i1", self.some_date, 1.0],
                ["u3", "i2", self.some_date, 1.0],
            ],
            schema=LOG_SCHEMA,
        )

    def test_fit(self):
        self.model.fit(self.log)
        self.assertSparkDataFrameEqual(
            self.model.similarity,
            self.spark.createDataFrame(
                [(1.0, 0.0, 0.5), (0.0, 1.0, 0.5),],
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
        recs = self.model.predict(log=self.log, k=1, users=["u1", "u2"],)
        self.assertSparkDataFrameEqual(
            recs,
            self.spark.createDataFrame(
                [["u1", "i2", 0.5], ["u2", "i1", 0.5],], schema=REC_SCHEMA,
            ),
        )
