"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy as np
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import DEFAULT_CONTEXT, LOG_SCHEMA, REC_SCHEMA
from sponge_bob_magic.models.als_recommender import ALSRecommender


class ALSRecommenderTestCase(PySparkTest):
    def setUp(self):
        self.als_recommender = ALSRecommender(1)
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
        self.als_recommender._seed = 42

    def test_fit(self):
        self.als_recommender.fit(self.log, None, None)
        item_factors = np.array(
            self.als_recommender.model.itemFactors
            .toPandas()["features"].tolist()
        )
        self.assertTrue(np.allclose(
            item_factors,
            [[0.94725847],  [0.82681108], [0.75606781]]
        ))

    def test_predict(self):
        recs = self.als_recommender.fit_predict(
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
                    ["u2", "i3", DEFAULT_CONTEXT, 0.8740121126174927],
                    ["u1", "i3", DEFAULT_CONTEXT, 0.8812910318374634],
                    ["u3", "i3", DEFAULT_CONTEXT, 1.0437875986099243]
                ],
                schema=REC_SCHEMA
            )
        )

    def test_get_params(self):
        self.assertEqual(self.als_recommender.get_params(), {"rank": 1})
