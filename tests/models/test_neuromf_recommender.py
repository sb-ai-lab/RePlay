"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy as np
import torch
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import DEFAULT_CONTEXT, LOG_SCHEMA
from sponge_bob_magic.models.neuromf_recommender import NeuroMFRecommender


class NeuroCFRecommenderTestCase(PySparkTest):
    def setUp(self):
        torch.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        params = {"learning_rate": 0.5,
                  "epochs": 1,
                  "embedding_dimension": 2}
        self.model = NeuroMFRecommender(**params)
        self.log = self.spark.createDataFrame(
            [
                ("0", "0", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("0", "1", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("0", "2", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("1", "0", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("1", "1", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("2", "0", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0)
            ],
            schema=LOG_SCHEMA
        )

    def test_fit(self):
        self.model.fit(log=self.log, user_features=None, item_features=None)

        true_parameters = [
            [[1.7252705, 1.5128007],
             [0.8895775, -0.45896536],
             [-0.22890945, -0.36223665]],
            [[0.13993913, -0.9970331],
             [1.4074863, 0.19857582],
             [0.8074074, 0.9650991]],
            [[-0.4999999],
             [0.4999999],
             [0]],
            [[0],
             [0],
             [0]]]

        for i, parameter in enumerate(self.model.model.parameters()):
            self.assertTrue(np.allclose(
                parameter.detach().cpu().numpy(), true_parameters[i],
                atol=1.e-3
            ))

    def test_predict(self):
        self.model.fit(log=self.log, user_features=None, item_features=None)
        predictions = self.model.predict(
            k=1,
            users=self.log.select('user_id').distinct(),
            items=self.log.select('item_id').distinct(),
            context='no_context',
            log=self.log,
            user_features=None,
            item_features=None,
            filter_seen_items=True
        )
        self.assertTrue(
            np.allclose(
                predictions.toPandas()
                [["user_id", "item_id"]].astype(int).values,
                [[0, 2], [1, 1], [2, 0]],
                atol=1.e-3
            )
        )
