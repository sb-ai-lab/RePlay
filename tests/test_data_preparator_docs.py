import unittest
import doctest
from sponge_bob_magic import data_preparator

suite = unittest.TestSuite()
suite.addTests(doctest.DocTestSuite(data_preparator))

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)
