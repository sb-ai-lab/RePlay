import pandas as pd

from sponge_bob_magic.models.wilson import Wilson
from tests.pyspark_testcase import PySparkTest


class Test(PySparkTest):
    def setUp(self):
        self.model = Wilson()

    def test_wilson(self):
        df = pd.DataFrame({"user_id": [1, 2], "item_id": [1, 2], "relevance": [1, 1]})
        self.model.fit_predict(df, k=1)
        self.assertTrue(True)
