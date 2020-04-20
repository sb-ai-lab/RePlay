import pandas as pd
from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic import experiment


df = pd.DataFrame({"user_id": [1, 1, 1],
                   "item_id": [1, 2, 3]})


class TestConverter(PySparkTest):
    def runTest(self):
        with self.assertRaises(TypeError):
            experiment.Experiment(df, "ᕕ( ᐛ )ᕗ")

    def test_separate_k(self):
        e = experiment.Experiment(df, [1, 2, 3], 3)
        self.assertEquals(list(e.metrics.values()), [3, 3, 3])
