"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
from datetime import datetime

import numpy as np
import torch
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.models.neuromf import NMF, NeuroMF


class NeuroCFRecTestCase(PySparkTest):
    def setUp(self):
        torch.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        params = {"learning_rate": 0.5,
                  "epochs": 1,
                  "embedding_gmf_dim": 2}
        self.model = NeuroMF(**params)
        self.log = self.spark.createDataFrame(
            [
                ("0", "0", datetime(2019, 1, 1), 1.0),
                ("0", "1", datetime(2019, 1, 1), 1.0),
                ("0", "2", datetime(2019, 1, 1), 1.0),
                ("1", "0", datetime(2019, 1, 1), 1.0),
                ("1", "1", datetime(2019, 1, 1), 1.0),
                ("0", "0", datetime(2019, 1, 1), 1.0),
                ("0", "1", datetime(2019, 1, 1), 1.0),
                ("0", "2", datetime(2019, 1, 1), 1.0),
                ("1", "0", datetime(2019, 1, 1), 1.0),
                ("1", "1", datetime(2019, 1, 1), 1.0),
                ("0", "0", datetime(2019, 1, 1), 1.0),
                ("0", "1", datetime(2019, 1, 1), 1.0),
                ("0", "2", datetime(2019, 1, 1), 1.0),
                ("1", "0", datetime(2019, 1, 1), 1.0),
                ("1", "1", datetime(2019, 1, 1), 1.0),
            ],
            schema=LOG_SCHEMA
        )

    def test_fit(self):
        self.model.fit(log=self.log)

        true_parameters = [
            [[1.4524888, 1.2240735],
             [0.6433717, 1.2327943]],
            [[1.7811029, 0.62523645],
             [-1.0805497, 0.21044984],
             [-0.9581958, 1.3094656]],
            [[-0.49999997],
             [0.49999994],
             [0.49999958]],
            [[-0.49999997],
             [-0.4999999]],
            [[-1.3116516, 1.47019]],
            [-0.49802172]
            ]

        for i, parameter in enumerate(self.model.model.parameters()):
            self.assertTrue(np.allclose(
                parameter.detach().cpu().numpy(), true_parameters[i],
                atol=1.e-3
            ))

    def test_predict(self):
        self.model.fit(log=self.log, user_features=None, item_features=None)
        predictions = self.model.predict(
            log=self.log,
            k=1,
            users=self.log.select('user_id').distinct(),
            items=self.log.select('item_id').distinct(),
            user_features=None,
            item_features=None,
            filter_seen_items=True
        )
        self.assertTrue(
            np.allclose(
                predictions.toPandas()
                [["user_id", "item_id"]].astype(int).values,
                [[0, 0], [1, 2]],
                atol=1.e-3
            )
        )

    def test_save_load(self):
        path = os.path.join(
            self.spark.conf.get("spark.local.dir"),
            "best_nmf_1_loss=-1.3341923952102661.pth"
        )
        if os.path.exists(path):
            os.remove(path)
        self.model.fit(log=self.log, user_features=None, item_features=None)

        self.assertTrue(
            os.path.exists(path)
        )

        new_model = NeuroMF()
        new_model.model = NMF(2, 3, 2)
        new_model.load_model(path)

        true_parameters = [
            [[1.4524888, 1.2240735],
             [0.6433717, 1.2327943]],
            [[1.7811029, 0.62523645],
             [-1.0805497, 0.21044984],
             [-0.9581958, 1.3094656]],
            [[-0.49999997],
             [0.49999994],
             [0.49999958]],
            [[-0.49999997],
             [-0.4999999]],
            [[-1.3116516, 1.47019]],
            [-0.49802172]
        ]

        for i, parameter in enumerate(new_model.model.parameters()):
            self.assertTrue(np.allclose(
                parameter.detach().cpu().numpy(), true_parameters[i],
                atol=1.e-3
            ))
