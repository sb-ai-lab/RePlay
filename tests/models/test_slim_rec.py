"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

from pyspark.sql.types import DoubleType, FloatType, StructField, StructType
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.models.slim_rec import SlimRec


class SlimRecTestCase(PySparkTest):
    def setUp(self):
        self.model = SlimRec(0.0, 0.01, seed=42)
        self.some_date = datetime(2019, 1, 1)
        self.log = self.spark.createDataFrame(
            [
                ["u1", "i1", self.some_date, 1.0],
                ["u2", "i1", self.some_date, 1.0],
                ["u3", "i3", self.some_date, 2.0],
                ["u2", "i3", self.some_date, 2.0],
                ["u3", "i4", self.some_date, 2.0],
                ["u1", "i4", self.some_date, 2.0],
                ["u4", "i1", self.some_date, 2.0]
            ],
            schema=LOG_SCHEMA
        )

    def test_fit(self):
        self.model._pre_fit(self.log, None, None)
        self.model._fit(self.log, None, None)
        self.assertSparkDataFrameEqual(
            self.model.similarity,
            self.spark.createDataFrame([
                (0., 1., 0.1633383178710937),
                (0., 2., 0.16333084106445306),
                (1., 0., 0.17636424039780527),
                (1., 2., 0.4509089399005487),
                (2., 0., 0.17636424039780527),
                (2., 1., 0.4509089399005487)
            ], schema=StructType([
                StructField("item_id_one", FloatType()),
                StructField("item_id_two", FloatType()),
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
                    ["u3", "i1", 0.35272848079561053],
                    ["u4", "i3", 0.1633383178710937],
                    ["u1", "i3", 0.6142472577716425],
                    ["u2", "i4", 0.6142397809650018],
                ],
                schema=REC_SCHEMA
            )
        )

    def test_get_params(self):
        self.assertDictEqual(
            self.model.get_params(),
            {"lambda": 0.01,
             "beta": 0.0}
        )

    def test_zeros_params_exception(self):
        self.assertRaises(
            ValueError,
            SlimRec,
            beta=0.0, lambda_=0.0
        )

    def test_negative_beta_exception(self):
        self.assertRaises(
            ValueError,
            SlimRec,
            beta=-0.1, lambda_=0.1
        )

    def test_negative_lambda_exception(self):
        self.assertRaises(
            ValueError,
            SlimRec,
            beta=0.1, lambda_=-0.1
        )
