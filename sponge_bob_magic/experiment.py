from collections import Iterable

import pandas as pd

from sponge_bob_magic.converter import convert


class Experiment:
    def __init__(self, test, metrics):
        self.test = convert(test)
        if not isinstance(metrics, Iterable):
            metrics = [metrics]
        self.metrics = metrics
        self.df = pd.DataFrame()

    def add_result(self, name, pred, k):
        res = pd.Series(name=name)
        for metric in self.metrics:
            res[f"{metric}@{k}"] = metric(pred, self.test, k)
        self.df = self.df.append(res)
