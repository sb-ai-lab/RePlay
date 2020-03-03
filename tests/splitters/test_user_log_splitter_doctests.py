import doctest
from unittest import TestSuite, TextTestRunner

from sponge_bob_magic.splitters import user_log_splitter

suite = TestSuite()
suite.addTests(doctest.DocTestSuite(user_log_splitter))

runner = TextTestRunner()
runner.run(suite)
