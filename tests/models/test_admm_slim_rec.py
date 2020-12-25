"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
from datetime import datetime

from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType
from tests.pyspark_testcase import PySparkTest

from replay.constants import LOG_SCHEMA, REC_SCHEMA
from replay.models.admm_slim import ADMMSLIM


class AdmmSlimRecTestCase(PySparkTest):
    def setUp(self):
        self.model = ADMMSLIM(1, 10, 42)
        self.some_date = datetime(2019, 1, 1)
        self.log = self.spark.createDataFrame(
            [
                ["u1", "i1", self.some_date, 1.0],
                ["u2", "i1", self.some_date, 1.0],
                ["u3", "i3", self.some_date, 2.0],
                ["u2", "i3", self.some_date, 2.0],
                ["u3", "i4", self.some_date, 2.0],
                ["u1", "i4", self.some_date, 2.0],
                ["u4", "i1", self.some_date, 2.0],
            ],
            schema=LOG_SCHEMA,
        )

    def test_fit(self):
        self.model.fit(self.log)
        self.assertSparkDataFrameEqual(
            self.model.similarity,
            self.spark.createDataFrame(
                [
                    (0, 1, 0.03095617860316846),
                    (0, 2, 0.030967752554031502),
                    (1, 0, 0.031891083964224354),
                    (1, 2, 0.1073860741574666),
                    (2, 0, 0.031883667509449376),
                    (2, 1, 0.10739028463512135),
                ],
                schema=StructType(
                    [
                        StructField("item_id_one", IntegerType()),
                        StructField("item_id_two", IntegerType()),
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
                    ["u3", "i1", 0.06377475147367373],
                    ["u1", "i3", 0.1383464632382898],
                    ["u4", "i4", 0.030967752554031502],
                    ["u2", "i4", 0.1383538267114981],
                ],
                schema=REC_SCHEMA,
            ),
        )

    def test_zeros_params_exception(self):
        self.assertRaises(ValueError, ADMMSLIM, lambda_1=0.0, lambda_2=0.0)

    def test_negative_beta_exception(self):
        self.assertRaises(ValueError, ADMMSLIM, lambda_1=-0.1, lambda_2=0.1)

    def test_negative_lambda_exception(self):
        self.assertRaises(ValueError, ADMMSLIM, lambda_1=0.1, lambda_2=-0.1)
