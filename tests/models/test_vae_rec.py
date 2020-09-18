"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
# pylint: disable-all
import os
import re
from datetime import datetime

import numpy as np
import torch
from tests.pyspark_testcase import PySparkTest

from replay.constants import LOG_SCHEMA
from replay.models.mult_vae import VAE, MultVAE


class VAERecTestCase(PySparkTest):
    def setUp(self):
        torch.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        params = {
            "learning_rate": 0.5,
            "epochs": 1,
            "latent_dim": 1,
            "decoder_dims": [1],
            "encoder_dims": [1],
        }
        self.parameter_stubs = [
            [[0.0, 0.0, 0.0]],
            [0.0],
            [[0.0], [0.0]],
            [0.0, 0.0],
            [[0.0]],
            [0.0],
            [[0.0], [0.0], [0.0]],
            [0.0, 0.0, 0.0],
        ]
        self.model = MultVAE(**params)
        self.log = self.spark.createDataFrame(
            [
                ("0", "0", datetime(2019, 1, 1), 1.0),
                ("0", "2", datetime(2019, 1, 1), 1.0),
                ("1", "0", datetime(2019, 1, 1), 1.0),
                ("1", "1", datetime(2019, 1, 1), 1.0),
            ],
            schema=LOG_SCHEMA,
        )
        self.other_log = self.spark.createDataFrame(
            [
                ("2", "0", datetime(2019, 1, 1), 1.0),
                ("2", "1", datetime(2019, 1, 1), 1.0),
                ("0", "0", datetime(2019, 1, 1), 1.0),
                ("0", "2", datetime(2019, 1, 1), 1.0),
            ],
            schema=LOG_SCHEMA,
        )

    def test_fit(self):
        self.model.fit(log=self.log)
        for i, parameter in enumerate(self.model.model.parameters()):
            self.assertEqual(
                parameter.shape, torch.tensor(self.parameter_stubs[i]).shape
            )

    def test_predict(self):
        self.model.fit(log=self.log)
        predictions = self.model.predict(
            log=self.other_log,
            k=1,
            users=self.other_log.select("user_id").distinct(),
            items=self.log.select("item_id").distinct(),
            filter_seen_items=True,
        )
        self.assertTrue(
            np.allclose(
                predictions.toPandas()[["user_id", "item_id"]]
                .astype(int)
                .values,
                [[0, 1], [2, 2]],
                atol=1.0e-3,
            )
        )

    def test_save_load(self):
        spark_local_dir = self.spark.conf.get("spark.local.dir")
        pattern = "best_multvae_1_loss=-\\d\\.\\d+.pth"
        for filename in os.listdir(spark_local_dir):
            if re.match(pattern, filename):
                os.remove(os.path.join(spark_local_dir, filename))
        self.model.fit(log=self.log)
        matched = False
        for filename in os.listdir(spark_local_dir):
            if re.match(pattern, filename):
                path = os.path.join(spark_local_dir, filename)
                matched = True
                break
        self.assertTrue(matched)
        new_model = MultVAE()
        new_model.model = VAE(3, 1, [1])
        new_model.load_model(path)
        for i, parameter in enumerate(new_model.model.parameters()):
            self.assertEqual(
                parameter.shape, torch.tensor(self.parameter_stubs[i]).shape
            )