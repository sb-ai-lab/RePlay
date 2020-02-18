import unittest
import doctest
from sponge_bob_magic.splitters import user_log_splitter


suite = unittest.TestSuite()
suite.addTests(doctest.DocTestSuite(user_log_splitter))

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)