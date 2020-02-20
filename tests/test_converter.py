import unittest

import pandas as pd

from sponge_bob_magic.converter import PANDAS, convert
from sponge_bob_magic.session_handler import State


class TestConverter(unittest.TestCase):
    def setUp(self):
        self.spark = State().session
        self.df = pd.DataFrame([[1, 2, 3],
                                [3, 4, 5]])

    def test_pandas_inversion(self):
        self.assertTrue(
            (self.df.values == convert(convert(self.df), PANDAS).values).all()
        )

    def test_unknown_type(self):
        with self.assertRaises(NotImplementedError):
            convert(1, "unknown_type")

    def test_spark_is_unchanged(self):
        spark = convert(self.df)
        self.assertEqual(spark, convert(spark))
