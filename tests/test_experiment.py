import unittest
import doctest
from sponge_bob_magic import experiment


suite = unittest.TestSuite()
suite.addTests(doctest.DocTestSuite(experiment))

runner = unittest.TextTestRunner(verbosity=2)
runner.run(suite)