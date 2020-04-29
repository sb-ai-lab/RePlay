"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy as np
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.models.lightfm_wrap import LightFMWrap


class LightFMRecTestCase(PySparkTest):
    def setUp(self):
        self.lightfm_rec = LightFMWrap(no_components=1, random_state=42)
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
        self.lightfm_rec.fit(self.log, None, None)
        item_factors = self.lightfm_rec.model.item_embeddings
        self.assertEqual(item_factors.shape, (3, 1))

    def test_predict(self):
        recs = self.lightfm_rec.fit_predict(
            log=self.log,
            k=1,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
            user_features=None,
            item_features=None,
        )
        self.assertEqual(recs.schema, REC_SCHEMA)

    def test_get_params(self):
        self.assertEqual(self.lightfm_rec.get_params(),
                         {"no_components": 1, "random_state": 42})
