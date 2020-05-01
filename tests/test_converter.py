import pandas as pd

from tests.pyspark_testcase import PySparkTest

from sponge_bob_magic.converter import PANDAS, convert


class TestConverter(PySparkTest):
    def setUp(self):
        self.pandas_data_frame = pd.DataFrame([[1, 2, 3], [3, 4, 5]])

    def test_pandas_inversion(self):
        self.assertTrue(
            (
                self.pandas_data_frame.values
                == convert(convert(self.pandas_data_frame), to=PANDAS).values
            ).all()
        )

    def test_unknown_type(self):
        with self.assertRaises(NotImplementedError):
            convert(1, to="unknown_type")

    def test_spark_is_unchanged(self):
        spark_data_frame = convert(self.pandas_data_frame)
        self.assertEqual(spark_data_frame, convert(spark_data_frame))
