"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from unittest import TestCase

import sponge_bob_magic


class TestPackage(TestCase):
    def test_version(self):
        self.assertIsInstance(sponge_bob_magic.__version__, str)
