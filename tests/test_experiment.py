import unittest
import doctest
from sponge_bob_magic import experiment
import pandas as pd


suite = unittest.TestSuite()
suite.addTests(doctest.DocTestSuite(experiment))


class TestConverter(unittest.TestCase):
    def runTest(self):
        with self.assertRaises(TypeError):
            experiment.Experiment(
                pd.DataFrame({"user_id": [1, 1, 1],
                              "item_id": [1, 2, 3],
                              "relevance": [5, 3, 4]}),
                "ᕕ( ᐛ )ᕗ"
            )


suite.addTest(TestConverter())
runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
