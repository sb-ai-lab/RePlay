import unittest

import pandas as pd

from sponge_bob_magic.converter import Converter
from sponge_bob_magic.session_handler import State


class TestConverter(unittest.TestCase):
    def setUp(self):
        self.spark = State().session

    def test_pandas_inversion(self):
        df = pd.DataFrame([[1,2,3],
                           [3,4,5]])
        c = Converter(df)
        self.assertTrue((df.values == c(c(df)).values).all())