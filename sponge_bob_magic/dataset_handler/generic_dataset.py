import os
from os.path import join

import pandas as pd
from IPython.core.display import display


class Dataset:
    def __init__(self):
        DATA_FOLDER = os.getenv("KRUSTY_KRABS", None)
        if DATA_FOLDER is None:
            ROOT = os.path.expanduser("~")
            DATA_FOLDER = join(ROOT, 'sb_magic_data')
        self.data_folder = DATA_FOLDER
        if not os.path.exists(DATA_FOLDER):
            os.mkdir(DATA_FOLDER)

    def info(self):
        with pd.option_context('display.max_columns', 10):
            for name, df in self.__dict__.items():
                print(name)
                display(df.head(3))
                print()