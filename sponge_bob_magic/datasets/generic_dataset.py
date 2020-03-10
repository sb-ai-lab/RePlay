"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import os
from os.path import join

import pandas as pd
from pandas import DataFrame


class Dataset:
    def __init__(self, path: str = None):
        data_folder = (path or os.getenv("KRUSTY_KRABS", None) or
                       self.default_folder)
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        self.data_folder = data_folder
        try:
            display = __import__('IPython.core.display', globals(), locals(), ['display'])
            self.display = display.display
        except:
            self.display = print

    @property
    def default_folder(self):
        root = os.path.expanduser("~")
        return join(root, 'sb_magic_data')

    def info(self):
        with pd.option_context('display.max_columns', 10):
            for name, df in self.__dict__.items():
                if isinstance(df, DataFrame):
                    print(name)
                    self.display(df.head(3))
                    print()
