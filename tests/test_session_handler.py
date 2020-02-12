import unittest

from sponge_bob_magic.session_handler import State, get_spark_session


class TestConverter(unittest.TestCase):
    @staticmethod
    def get_session():
        return State().session

    def test_pandas_inversion(self):
        s1 = self.get_session()
        s2 = self.get_session()
        self.assertTrue(s1 is s2)

    def test_initialization(self):
        s1 = get_spark_session()
        s2 = State(s1).session
        self.assertTrue(s1 is s2)
