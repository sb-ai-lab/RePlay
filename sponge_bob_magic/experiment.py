from collections import Iterable
from typing import Any, List

import pandas as pd

from sponge_bob_magic.converter import convert
from sponge_bob_magic.metrics import Metric


class Experiment:
    """
    Обеспечивает подсчет и хранение значений метрик.

    Пример:
    >>> import pandas as pd
    >>> from sponge_bob_magic.metrics import NDCG
    >>> from sponge_bob_magic.experiment import Experiment
    >>> test = pd.DataFrame({"user_id": [1,1,1], "item_id": [1,2,3], "relevance": [5,3,4]})
    >>> pred = pd.DataFrame({"user_id": [1,1,1], "item_id": [1,2,4], "relevance": [5,4,5]})
    >>> ex = Experiment(test, NDCG())
    >>> ex.add_result('my_model', pred, 3)
    >>> ex.df
                nDCG@3
    my_model  0.703918

    """
    def __init__(self, test: Any, metrics: List[Metric]):
        """
        :param test: Данные для теста в формате pandas или pyspark DataFrame
        :param metrics: Список метрик, которые необходимо считать
        """
        self.test = convert(test)
        if not isinstance(metrics, Iterable):
            metrics = [metrics]
        self.metrics = metrics
        self.df = pd.DataFrame()

    def add_result(self, name: str, pred: Any, k: int):
        """
        Подсчитать метрики для переданного списка рекомендаций
        :param name: имя модели/эксперимента для сохранения результатов
        :param pred: список рекомендаций для подсчета метрик
        :param k: конкретное значение k для metric@k
        """
        res = pd.Series(name=name)
        recs = convert(pred)
        for metric in self.metrics:
            res[f"{metric}@{k}"] = metric(recs, self.test, k)
        self.df = self.df.append(res)
