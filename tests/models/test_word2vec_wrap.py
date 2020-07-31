"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
from datetime import datetime

import numpy as np
from pyspark.sql import functions as sf
from tests.pyspark_testcase import PySparkTest

from replay.constants import LOG_SCHEMA, REC_SCHEMA
from replay.models.word2vec import Word2VecRec
from replay.utils import vector_dot


class Word2VecRecTestCase(PySparkTest):
    def setUp(self):
        self.word2vec = Word2VecRec(
            rank=1, window_size=1, use_idf=True, seed=42
        )
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
        self.word2vec.fit(self.log)
        vectors = (
            self.word2vec.vectors.select(
                "item",
                vector_dot(sf.col("vector"), sf.col("vector")).alias("norm"),
            )
            .toPandas()
            .to_numpy()
        )
        print(vectors)
        self.assertTrue(
            np.allclose(
                vectors,
                [
                    [0, 5.45887464e-04],
                    [2, 1.54838404e-01],
                    [1, 2.13055389e-01],
                ],
            )
        )

    def test_predict(self):
        recs = self.word2vec.fit_predict(
            log=self.log,
            k=1,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
        )
        true_recs = self.spark.createDataFrame(
            [
                ["u1", "i3", 1.000322493440465],
                ["u2", "i4", 0.9613139892286415],
                ["u3", "i1", 0.9783670469059589],
            ],
            schema=REC_SCHEMA,
        )
        self.assertSparkDataFrameEqual(recs, true_recs)
