"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

from pyspark.sql.types import DoubleType, FloatType, StructField, StructType
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.models.slim import SLIM


class SlimRecTestCase(PySparkTest):
    def setUp(self):
        self.model = SLIM(0.0, 0.01, seed=42)
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
        self.model._pre_fit(self.log, None, None)
        self.model._fit(self.log, None, None)
        self.assertSparkDataFrameEqual(
            self.model.similarity,
            self.spark.createDataFrame(
                [
                    (0.0, 1.0, 0.163338303565979),
                    (0.0, 2.0, 0.1633233278989792),
                    (1.0, 0.0, 0.17635512351989746),
                    (1.0, 2.0, 0.45091119408607483),
                    (2.0, 0.0, 0.17635512351989746),
                    (2.0, 1.0, 0.45091116428375244),
                ],
                schema=StructType(
                    [
                        StructField("item_id_one", FloatType()),
                        StructField("item_id_two", FloatType()),
                        StructField("similarity", DoubleType()),
                    ]
                ),
            ),
        )

    def test_predict(self):
        self.model.fit(self.log, None, None)
        recs = self.model.predict(
            log=self.log,
            k=1,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
            user_features=None,
            item_features=None,
        )
        self.assertSparkDataFrameEqual(
            recs,
            self.spark.createDataFrame(
                [
                    ["u3", "i1", 0.3527102470397949],
                    ["u4", "i3", 0.163338303565979],
                    ["u1", "i3", 0.6142494678497314],
                    ["u2", "i4", 0.614234521985054],
                ],
                schema=REC_SCHEMA,
            ),
        )

    def test_get_params(self):
        self.assertDictEqual(self.model.get_params(), {"lambda": 0.01, "beta": 0.0})

    def test_zeros_params_exception(self):
        self.assertRaises(ValueError, SLIM, beta=0.0, lambda_=0.0)

    def test_negative_beta_exception(self):
        self.assertRaises(ValueError, SLIM, beta=-0.1, lambda_=0.1)

    def test_negative_lambda_exception(self):
        self.assertRaises(ValueError, SLIM, beta=0.1, lambda_=-0.1)
