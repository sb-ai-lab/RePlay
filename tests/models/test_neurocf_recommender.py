"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime

import numpy as np
import torch

from sponge_bob_magic.constants import DEFAULT_CONTEXT, LOG_SCHEMA
from sponge_bob_magic.models.neurocf_recommender import NeuroCFRecommender
from tests.pyspark_testcase import PySparkTest


class NeuroCFRecommenderTestCase(PySparkTest):
    def setUp(self):
        torch.manual_seed(7)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)

        self.model = NeuroCFRecommender(self.spark)
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

    def test_build_model(self):
        params = {"learning_rate": 0.5,
                  "epochs": 1,
                  "embedding_dimension": 10}
        self.model.set_params(**params)

        self.model.fit(
            log=self.log,
            user_features=None, item_features=None,
            path=self.tmp_path
        )
        true_parameters = [
            [[-0.574079, -0.5637433, -0.65786004, -0.43539688, -0.492432,
              0.48474327, -0.48040557, -0.6568693, -0.56893927, -0.50494426],
             [-0.12239242, 0.06402726, 0.06839443, 0.1613234, -0.21339302,
              -0.07492699, 0.06339749, 0.10036361, -0.2188052, -0.09927215],
             [0.5888339, 0.35131282, -0.5912658, -0.37283406, -0.485432,
              0.63704497, -0.5375623, -0.48305985, -0.554148, -0.51603514]],
            [[0.58036685, -0.49260595, -0.51941055, 0.34859565, 0.55277073,
              0.6134183, -0.5063636, -0.37282014, -0.5530155, -0.48669678],
             [0.52109045, 0.45903623, 0.73525757, -0.4736089, -0.42985758,
              -0.56908673, -0.4701805, 0.6461237, 0.600655, 0.5977748],
             [-0.54631996, -0.670653, -0.715097, 0.37202054, 0.40095997,
              -0.5003761, 0.4812769, -0.48920715, -0.5403248, -0.4527604]],
            [[0.49999952], [-0.49999967], [0.49999905]],
            [[0.], [0.], [0.]]
        ]
        for i, parameter in enumerate(self.model.model.parameters()):
            self.assertTrue(np.allclose(
                parameter.detach().cpu().numpy(), true_parameters[i],
                atol=1.e-3
            ))
