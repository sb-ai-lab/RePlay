"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
from datetime import datetime

import numpy as np
import torch
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.models.mult_vae import VAE, MultVAE


class VAERecTestCase(PySparkTest):
    def setUp(self):
        torch.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        params = {"learning_rate": 0.5,
                  "epochs": 1,
                  "latent_dim": 1,
                  "decoder_dims": [1],
                  "encoder_dims": [1]}
        self.model = MultVAE(**params)
        self.log = self.spark.createDataFrame(
            [
                ("0", "0", datetime(2019, 1, 1), 1.0),
                ("0", "2", datetime(2019, 1, 1), 1.0),
                ("1", "0", datetime(2019, 1, 1), 1.0),
                ("1", "1", datetime(2019, 1, 1), 1.0),
            ],
            schema=LOG_SCHEMA
        )

    def test_fit(self):
        self.model.fit(log=self.log, user_features=None, item_features=None)

        true_parameters = [
            [[0.09770283, -1.189494, 0.22463249]],
            [0.00013281],
            [[0.11212465], [0.1964051]],
            [-0.49788895, -0.49717355],
            [[2.438208]],
            [0.00020276],
            [[1.7327943], [1.4323168], [1.2580526]],
            [0.49908203, 0.49954212, -0.5007245]
        ]

        for i, parameter in enumerate(self.model.model.parameters()):
            self.assertTrue(np.allclose(
                parameter.detach().cpu().numpy(), true_parameters[i],
                atol=1.e-3
            ))

    def test_get_params(self):
        self.assertDictEqual(
            self.model.get_params(),
            {"anneal": 0.005,
             "decoder_dims": [1],
             "dropout": 0.3,
             "encoder_dims": [1],
             "epochs": 1,
             "gamma": 0.99,
             "l2_reg": 0,
             "latent_dim": 1,
             "learning_rate": 0.5}
        )

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
                [[0, 1], [1, 2]],
                atol=1.e-3
            )
        )

    def test_save_load(self):
        path = os.path.join(
            self.spark.conf.get("spark.local.dir"),
            "best_vae_1_loss=-2.7250239849090576.pth"
        )
        if os.path.exists(path):
            os.remove(path)
        self.model.fit(log=self.log, user_features=None, item_features=None)

        self.assertTrue(
            os.path.exists(path)
        )

        new_model = MultVAE()
        new_model.model = VAE(3, 1, [1])
        new_model.load_model(path)

        true_parameters = [
            [[0.09770283, - 1.189494, 0.22463249]],
            [0.00013281],
            [[0.11212465], [0.1964051]],
            [-0.49788895, -0.49717355],
            [[2.438208]],
            [0.00020276],
            [[1.7327943], [1.4323168], [1.2580526]],
            [0.49908203, 0.49954212, -0.5007245]
        ]

        for i, parameter in enumerate(new_model.model.parameters()):
            self.assertTrue(np.allclose(
                parameter.detach().cpu().numpy(), true_parameters[i],
                atol=1.e-3
            ))
