import pandas as pd
from tests.pyspark_testcase import PySparkTest

from replay import experiment


class TestConverter(PySparkTest):
    def setUp(self):
        self.data_frame = pd.DataFrame(
            {"user_id": [1, 1, 1], "item_id": [1, 2, 3]}
        )

    def runTest(self):
        with self.assertRaises(TypeError):
            experiment.Experiment(self.data_frame, "ᕕ( ᐛ )ᕗ")

    def test_separate_k(self):
        e = experiment.Experiment(self.data_frame, [1, 2, 3], 3)
        self.assertEquals(list(e.metrics.values()), [3, 3, 3])
