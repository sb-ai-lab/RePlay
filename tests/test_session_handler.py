import unittest

from sponge_bob_magic.session_handler import State


class TestConverter(unittest.TestCase):
    def get_session(self):
        return State().session

    def test_pandas_inversion(self):
        s1 = self.get_session()
        s2 = self.get_session()
        self.assertTrue(s1 is s2)