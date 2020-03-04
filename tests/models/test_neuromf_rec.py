"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
from datetime import datetime

import numpy as np
import torch
from pyspark.sql import SparkSession

from sponge_bob_magic.constants import DEFAULT_CONTEXT, LOG_SCHEMA
from sponge_bob_magic.models.neuromf_rec import NeuroMFRec, NMF
from tests.pyspark_testcase import PySparkTest


class NeuroCFRecTestCase(PySparkTest):
    def setUp(self):
        torch.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        params = {"learning_rate": 0.5,
                  "epochs": 1,
                  "embedding_dimension": 2}
        self.model = NeuroMFRec(**params)
        self.log = self.spark.createDataFrame(
            [
                ("0", "0", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("0", "1", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("0", "2", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("1", "0", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("1", "1", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("0", "0", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("0", "1", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("0", "2", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("1", "0", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("1", "1", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("0", "0", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("0", "1", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("0", "2", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("1", "0", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
                ("1", "1", datetime(2019, 1, 1), DEFAULT_CONTEXT, 1.0),
            ],
            schema=LOG_SCHEMA
        )

    def test_fit(self):
        self.model.fit(log=self.log, user_features=None, item_features=None)

        true_parameters = [
            [[0.6735113, 1.219104],
             [0.10137914, 1.2252706]],
            [[1.0128009, 0.8895775],
             [-0.45896536, -0.22890945],
             [-0.36223665, 0.63993907]],
            [[0.], [0.], [0.]],
            [[0.], [0.]]
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
            context='no_context',
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
        spark = SparkSession(self.log.rdd.context)
        path = os.path.join(
            spark.conf.get("spark.local.dir"),
            "best_nmf.pth"
        )
        if os.path.exists(path):
            os.remove(path)
        self.model.fit(log=self.log, user_features=None, item_features=None)

        self.assertTrue(
            os.path.exists(path)
        )

        new_model = NeuroMFRec()
        new_model.model = NMF(2, 3, 2)
        new_model.load_model(path)

        true_parameters = [
            [[0.6735113, 1.219104],
             [0.10137914, 1.2252706]],
            [[1.0128009, 0.8895775],
             [-0.45896536, -0.22890945],
             [-0.36223665, 0.63993907]],
            [[0.], [0.], [0.]],
            [[0.], [0.]]
        ]

        for i, parameter in enumerate(new_model.model.parameters()):
            self.assertTrue(np.allclose(
                parameter.detach().cpu().numpy(), true_parameters[i],
                atol=1.e-3
            ))
