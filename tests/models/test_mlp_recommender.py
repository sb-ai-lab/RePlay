"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy as np
import torch

from sponge_bob_magic.constants import DEFAULT_CONTEXT, LOG_SCHEMA
from sponge_bob_magic.models.mlp_rec import MLPRec
from tests.pyspark_testcase import PySparkTest


class MLPRecTestCase(PySparkTest):
    def setUp(self):
        torch.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        params = {"learning_rate": 0.5,
                  "epochs": 1,
                  "embedding_dimension": 2}
        self.model = MLPRec(**params)
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

        true_parameters = [[
            [1.7252257, 0.5128725],
            [1.3812021, -0.9589134],
            [-0.7288457, 0.13769689]]]

        for i, parameter in enumerate(self.model.model.parameters()):
            self.assertTrue(np.allclose(
                parameter.detach().cpu().numpy(), true_parameters[i],
                atol=1.e-3
            ))
            break

    def test_predict(self):
        self.model.fit(log=self.log, user_features=None, item_features=None)
        predictions = self.model.predict(log=self.log, k=1, users=self.log.select('user_id').distinct(),
                                         items=self.log.select('item_id').distinct(), context='no_context',
                                         user_features=None, item_features=None, filter_seen_items=True)
        self.assertTrue(
            np.allclose(
                predictions.toPandas()[["user_id", "item_id"]].astype(int).values,
                [[0, 2], [1, 1], [2, 0]],
                atol=1.e-3
            )
        )
