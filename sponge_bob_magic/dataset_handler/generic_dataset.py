import pandas as pd
from IPython.core.display import display


class Dataset:
    def info(self):
        with pd.option_context('display.max_columns', 10):
            for name, df in self.__dict__.items():
                print(name)
                display(df.head(3))
                print()