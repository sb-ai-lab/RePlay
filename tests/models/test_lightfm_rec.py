"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy as np
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import DEFAULT_CONTEXT, LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.models.lightfm_rec import LightFMRec


class LightFMRecTestCase(PySparkTest):
    def setUp(self):
        self.lightfm_rec = LightFMRec(1)
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
        self.lightfm_rec._seed = 42

    def test_fit(self):
        self.lightfm_rec.fit(self.log, None, None)
        item_factors = self.lightfm_rec.model.item_embeddings
        self.assertTrue(np.allclose(
            item_factors,
            [[-0.06065203], [0.5662015], [0.04397682]]
        ))

    def test_predict(self):
        recs = self.lightfm_rec.fit_predict(
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
                    ["u1", "i3", DEFAULT_CONTEXT, -0.25914710760116577],
                    ["u2", "i3", DEFAULT_CONTEXT, -0.2138521820306778],
                    ["u3", "i4", DEFAULT_CONTEXT, -0.3359125852584839]
                ],
                schema=REC_SCHEMA
            )
        )

    def test_get_params(self):
        self.assertEqual(self.lightfm_rec.get_params(), {"rank": 1})
