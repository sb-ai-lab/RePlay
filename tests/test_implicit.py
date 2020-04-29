import pandas as pd
import numpy as np

from sponge_bob_magic.converter import convert
from sponge_bob_magic.models.implicit import to_numpy
from tests.pyspark_testcase import PySparkTest

df = pd.DataFrame(
    {
        "user_idx": [1, 1, 1, 3, 3, 3, 5, 5, 5],
        "item_idx": [1, 2, 3, 3, 4, 5, 4, 5, 7],
        "relevance": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    }
)
sd = convert(df)


class TestImplicit(PySparkTest):
    def test_to_numpy(self):
        self.assertTrue((to_numpy(sd, "relevance") == np.array(df["relevance"])).all())
