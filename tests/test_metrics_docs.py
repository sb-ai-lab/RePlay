import unittest
import doctest
from sponge_bob_magic import metrics

suite = unittest.TestSuite()
suite.addTests(doctest.DocTestSuite(metrics))

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
