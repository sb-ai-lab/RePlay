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
                  "embedding_gmf_dim": 2,
                  "embedding_mlp_dim": 2,
                  "hidden_mlp_dims": [2]}
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
            [[1.4524879455566406, 1.2240769863128662],
             [0.6433713436126709, 1.2327945232391357]],
            [[1.7811020612716675, 0.6252365112304688],
             [-0.08055222034454346, 0.2104490101337433],
             [-0.9581958651542664, 1.3094656467437744]],
            [[-0.49999985098838806], [0.49999940395355225],
             [0.49999961256980896]],
            [[0.4999987483024597], [-0.4999995827674866]],
            [[0.07680675387382507, 1.6461725234985352],
             [-0.5584385395050049, 0.7213757634162903]],
            [[-0.44971248507499695, 0.37632137537002563],
             [-0.31071579456329346, 0.3033224046230316],
             [1.3194324970245361, -1.0064270496368408]],
            [[0.49999862909317017], [0.0], [-0.4999985694885254]],
            [[-0.49998268485069275], [0.0]],
            [[0.7620220184326172, -1.1707366704940796, 1.1114522218704224,
              -1.4673712253570557], [-0.0868292972445488, -1.0307914018630981,
                                     -0.8177362680435181, -0.655718207359314]],
            [-0.5013818144798279, -0.000268184463493526],
            [[-0.6089141964912415, -0.21500834822654724, 0.7593991756439209,
              0.5665111541748047]],
            [-0.5002881288528442]
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
            "best_nmf_1_loss=-0.9791234135627747.pth"
        )
        if os.path.exists(path):
            os.remove(path)
        self.model.fit(log=self.log, user_features=None, item_features=None)

        self.assertTrue(
            os.path.exists(path)
        )

        new_model = NeuroMF()
        new_model.model = NMF(2, 3, 2, 2, [2])
        new_model.load_model(path)

        true_parameters = [
            [[1.4524879455566406, 1.2240769863128662],
             [0.6433713436126709, 1.2327945232391357]],
            [[1.7811020612716675, 0.6252365112304688],
             [-0.08055222034454346, 0.2104490101337433],
             [-0.9581958651542664, 1.3094656467437744]],
            [[-0.49999985098838806], [0.49999940395355225],
             [0.49999961256980896]],
            [[0.4999987483024597], [-0.4999995827674866]],
            [[0.07680675387382507, 1.6461725234985352],
             [-0.5584385395050049, 0.7213757634162903]],
            [[-0.44971248507499695, 0.37632137537002563],
             [-0.31071579456329346, 0.3033224046230316],
             [1.3194324970245361, -1.0064270496368408]],
            [[0.49999862909317017], [0.0], [-0.4999985694885254]],
            [[-0.49998268485069275], [0.0]],
            [[0.7620220184326172, -1.1707366704940796, 1.1114522218704224,
              -1.4673712253570557], [-0.0868292972445488, -1.0307914018630981,
                                     -0.8177362680435181, -0.655718207359314]],
            [-0.5013818144798279, -0.000268184463493526],
            [[-0.6089141964912415, -0.21500834822654724, 0.7593991756439209,
              0.5665111541748047]],
            [-0.5002881288528442]
        ]

        for i, parameter in enumerate(new_model.model.parameters()):
            self.assertTrue(np.allclose(
                parameter.detach().cpu().numpy(), true_parameters[i],
                atol=1.e-3
            ))
