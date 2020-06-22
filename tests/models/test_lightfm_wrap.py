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
        self.lightfm_wrap.num_threads = 1
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
        self.user_features = self.spark.createDataFrame(
            [("u1", 2.0, 3.0)]
        ).toDF("user_id", "user_feature_1", "user_feature_2")
        self.item_features = self.spark.createDataFrame(
            [("i1", 4.0, 5.0)]
        ).toDF("item_id", "item_feature_1", "item_feature_2")

    def test_fit(self):
        self.lightfm_wrap.fit(self.log, self.user_features, self.item_features)
        item_factors = self.lightfm_wrap.model.item_embeddings
        self.assertTrue(
            np.allclose(
                item_factors,
                [
                    [0.01821348],
                    [0.55787325],
                    [-0.01927805],
                    [0.3785215],
                    [-0.04293519],
                ],
            )
        )

    def test_predict(self):
        recs = self.lightfm_wrap.fit_predict(
            log=self.log,
            k=1,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
            user_features=self.user_features,
            item_features=self.item_features,
        )
        true_recs = self.spark.createDataFrame(
            [["u3", "i1", 0.0], ["u1", "i3", 0.0], ["u2", "i4", 0.0]],
            schema=REC_SCHEMA,
        )
        self.assertSparkDataFrameEqual(recs, true_recs)
