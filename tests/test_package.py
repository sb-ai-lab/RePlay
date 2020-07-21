"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from unittest import TestCase

import replay


class TestPackage(TestCase):
    def test_version(self):
        self.assertIsInstance(replay.__version__, str)
