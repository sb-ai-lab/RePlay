import os
from os.path import join


DATA_FOLDER = os.getenv("KRUSTY_KRABS", None)
if DATA_FOLDER is None:
    ROOT = os.path.expanduser("~")
    DATA_FOLDER = join(ROOT, 'sb_magic_data')

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

from sponge_bob_magic.dataset_handler.movielens import MovieLens
