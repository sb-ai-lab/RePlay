import unittest

import pandas as pd

from sponge_bob_magic.converter import convert
from sponge_bob_magic.session_handler import State


class TestConverter(unittest.TestCase):
    def setUp(self):
        self.spark = State().session

    def test_pandas_inversion(self):
        df = pd.DataFrame([[1,2,3],
                           [3,4,5]])
        self.assertTrue((df.values == convert(convert(df), 'pandas').values).all())