import os
import sys
from os.path import join


ROOT = os.path.dirname(sys.modules['sponge_bob_magic'].__file__)
DATA_FOLDER = join(ROOT, 'datasets')

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

from sponge_bob_magic.dataset_handler.movielens import MovieLens
