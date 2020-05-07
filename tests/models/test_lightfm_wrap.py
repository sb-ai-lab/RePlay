"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
from datetime import datetime

import numpy as np
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.models.lightfm_wrap import LightFMWrap


class LightFMWrapTestCase(PySparkTest):
    def setUp(self):
        self.lightfm_wrap = LightFMWrap(
            no_components=1, random_state=42, loss="bpr"
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
        self.lightfm_wrap.fit(self.log)
        item_factors = self.lightfm_wrap.model.item_embeddings
        self.assertTrue(
            np.allclose(
                item_factors, [[-0.07829587], [0.3076668], [0.32417864]]
            )
        )

    def test_predict(self):
        recs = self.lightfm_wrap.fit_predict(
            log=self.log,
            k=1,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
        )
        true_recs = self.spark.createDataFrame(
            [["u3", "i1", 0.0], ["u1", "i3", 0.0], ["u2", "i4", 0.0]],
            schema=REC_SCHEMA,
        )
        self.assertSparkDataFrameEqual(recs, true_recs)

    def test_get_params(self):
        self.assertEqual(
            self.lightfm_wrap.get_params(),
            {"no_components": 1, "random_state": 42, "loss": "bpr"},
        )
