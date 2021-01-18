# pylint: disable-all
import os
import re
from datetime import datetime

import numpy as np
import torch
from tests.pyspark_testcase import PySparkTest
from tests.test_utils import del_files_by_pattern, find_file_by_pattern

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
        self.param_shapes = [
            (3, 2),
            (4, 2),
            (4, 1),
            (3, 1),
            (3, 2),
            (4, 2),
            (4, 1),
            (3, 1),
            (2, 4),
            (2,),
            (1, 4),
            (1,),
        ]

    def test_fit(self):
        self.model.fit(log=self.log)
        self.assertEqual(
            len(self.param_shapes), len(list(self.model.model.parameters()))
        )
        for i, parameter in enumerate(self.model.model.parameters()):
            self.assertEqual(
                self.param_shapes[i], tuple(parameter.shape),
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
                .sort_values("user_id")
                .astype(int)
                .values,
                [[0, 3], [1, 3], [3, 0]],
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
        spark_local_dir = self.spark.conf.get("spark.local.dir")
        pattern = "best_neuromf_1_loss=-\\d\\.\\d+.pth"
        del_files_by_pattern(spark_local_dir, pattern)

        self.model.fit(log=self.log)
        old_params = [
            param.detach().cpu().numpy()
            for param in self.model.model.parameters()
        ]
        path = find_file_by_pattern(spark_local_dir, pattern)
        self.assertIsNotNone(path)

        new_model = NeuroMF(embedding_mlp_dim=1)
        new_model.model = NMF(3, 4, 2, 2, [2])
        self.assertEqual(
            len(old_params), len(list(new_model.model.parameters()))
        )

        new_model.load_model(path)
        for i, parameter in enumerate(new_model.model.parameters()):
            self.assertTrue(
                np.allclose(
                    parameter.detach().cpu().numpy(),
                    old_params[i],
                    atol=1.0e-3,
                )
            )

    def test_embeddings_size(self):
        # параметры по умолчанию
        model = NeuroMF()
        self.assertTrue(
            (model.embedding_gmf_dim == 128) & (model.embedding_mlp_dim == 128)
        )
        # одна базовая модель, определенная пользователем
        model = NeuroMF(embedding_gmf_dim=16)
        self.assertTrue(
            (model.embedding_gmf_dim == 16) & (model.embedding_mlp_dim is None)
        )

        model = NeuroMF(embedding_gmf_dim=16, embedding_mlp_dim=32)
        self.assertTrue(
            (model.embedding_gmf_dim == 16) & (model.embedding_mlp_dim == 32)
        )

    def test_negative_dims_exception(self):
        self.assertRaises(
            ValueError, NeuroMF, embedding_gmf_dim=-2, embedding_mlp_dim=-1,
        )
