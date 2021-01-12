"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
import os
from datetime import datetime

import numpy as np
import torch
from tests.pyspark_testcase import PySparkTest

from replay.constants import LOG_SCHEMA
from replay.models.neuromf import NMF, NeuroMF


class NeuroCFRecTestCase(PySparkTest):
    def setUp(self):
        torch.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        params = {
            "learning_rate": 0.5,
            "epochs": 1,
            "embedding_gmf_dim": 2,
            "embedding_mlp_dim": 2,
            "hidden_mlp_dims": [2],
        }
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
                ("3", "3", datetime(2019, 1, 1), 1.0),
            ],
            schema=LOG_SCHEMA,
        )
        self.true_params = [
            [
                [-0.08055201, 0.21044976],
                [0.04180124, 0.30946603],
                [-0.12870437, 0.6478908],
            ],
            [
                [0.15194291, 0.4323138],
                [0.61440194, 0.25606853],
                [0.451958, -0.06795466],
                [-0.22132027, -0.60869837],
            ],
            [[-0.49999988], [0.4999998], [0.49999988], [-0.49999973]],
            [[0.49999976], [-0.49999994], [-0.4999939]],
            [
                [-0.89578515, -0.7183033],
                [-0.88041973, -0.16961475],
                [-0.48463845, -0.08171481],
            ],
            [
                [-0.68096894, 0.18050548],
                [-0.2244729, 0.48060486],
                [-0.47243598, -0.04344084],
                [0.10215224, 0.47986147],
            ],
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [0.0]],
            [
                [0.5002869, -0.09522208, -0.56052274, -0.2334511],
                [-0.06946294, 1.3970579, -0.1873725, -0.9444881],
            ],
            [1.2796186e-04, -3.2406952e-06],
            [[0.4148497, -0.6594144, 0.13334092, 0.42699796]],
            [-0.49998853],
        ]

    def test_fit(self):
        self.model.fit(log=self.log)

        for i, parameter in enumerate(self.model.model.parameters()):
            self.assertTrue(
                np.allclose(
                    parameter.detach().cpu().numpy(),
                    self.true_params[i],
                    atol=1.0e-3,
                )
            )

    def test_predict(self):
        self.model.fit(log=self.log)
        predictions = self.model.predict(
            log=self.log,
            k=1,
            users=self.log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
            filter_seen_items=True,
        )
        self.assertTrue(
            np.allclose(
                predictions.toPandas()[["user_id", "item_id"]]
                .astype(int)
                .values,
                [[1, 3], [3, 0], [0, 3]],
                atol=1.0e-3,
            )
        )

    def test_check_gmf_only(self):
        params = {"learning_rate": 0.5, "epochs": 1, "embedding_gmf_dim": 2}
        raised = False
        self.model = NeuroMF(**params)
        try:
            self.model.fit(log=self.log)
        except RuntimeError:
            raised = True
        self.assertFalse(raised)

    def test_check_mlp_only(self):
        params = {
            "learning_rate": 0.5,
            "epochs": 1,
            "embedding_mlp_dim": 2,
            "hidden_mlp_dims": [2],
        }
        raised = False
        self.model = NeuroMF(**params)
        try:
            self.model.fit(log=self.log)
        except RuntimeError:
            raised = True
        self.assertFalse(raised)

    def test_check_simple_mlp_only(self):
        params = {"learning_rate": 0.5, "epochs": 1, "embedding_mlp_dim": 2}
        raised = False
        self.model = NeuroMF(**params)
        try:
            self.model.fit(log=self.log)
        except RuntimeError:
            raised = True
        self.assertFalse(raised)

    def test_save_load(self):
        path = os.path.join(
            self.spark.conf.get("spark.local.dir"),
            "best_neuromf_1_loss=-0.7371938228607178.pth",
        )
        if os.path.exists(path):
            os.remove(path)
        self.model.fit(log=self.log)
        self.assertTrue(os.path.exists(path))

        new_model = NeuroMF(embedding_mlp_dim=1)
        new_model.model = NMF(3, 4, 2, 2, [2])
        new_model.load_model(path)

        for i, parameter in enumerate(new_model.model.parameters()):
            self.assertTrue(
                np.allclose(
                    parameter.detach().cpu().numpy(),
                    self.true_params[i],
                    atol=1.0e-3,
                )
            )

    def test_empty_embeddings_exception(self):
        self.assertRaises(
            ValueError, NeuroMF,
        )

    def test_negative_dims_exception(self):
        self.assertRaises(
            ValueError, NeuroMF, embedding_gmf_dim=-2, embedding_mlp_dim=-1,
        )
