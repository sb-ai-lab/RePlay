"""RecSys library"""

# NOTE: This ensures distutils monkey-patching is performed before any
# functionality removed in Python 3.12 is used in downstream packages (like lightfm)
import setuptools as _

__version__ = "0.0.0"
