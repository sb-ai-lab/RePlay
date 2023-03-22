""" RecSys library """
import pkg_resources

try:
    __version__ = pkg_resources.get_distribution("replay-rec").version
except pkg_resources.DistributionNotFound:
    __version__ = 'non-package'
