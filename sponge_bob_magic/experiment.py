from collections.abc import Iterable
from typing import Any, List, Union, Dict

import pandas as pd

from sponge_bob_magic.converter import convert
from sponge_bob_magic.metrics import Metric


class Experiment:
    """
    Обеспечивает подсчет и хранение значений метрик.

    Инициализируется тестом, на котором нужно считать метрики и словарём метрики-значения k.

    Результаты доступны в атрибуте ``df`` в виде pandas.DataFrame.

    Пример:

    >>> import pandas as pd
    >>> from sponge_bob_magic.metrics import NDCG, Surprisal
    >>> from sponge_bob_magic.experiment import Experiment
    >>> log = pd.DataFrame({"user_id": [2,2,2,1], "item_id": [1,2,3,3], "relevance": [5,5,5,5]})
    >>> test = pd.DataFrame({"user_id": [1,1,1], "item_id": [1,2,3], "relevance": [5,3,4]})
    >>> pred = pd.DataFrame({"user_id": [1,1,1], "item_id": [1,3,4], "relevance": [5,4,5]})
    >>> ex = Experiment(test, {NDCG(): [2,3], Surprisal(log): 3})
    >>> ex.add_result('my_model', pred)
    >>> ex.df
              Surprisal@3    nDCG@2    nDCG@3
    my_model     0.666667  0.613147  0.703918

    """
    def __init__(self,
                 test: Any,
                 metrics: Dict[Metric, Union[int, List[int]]]):
        """
        :param test: Данные для теста в формате ``pandas`` или ``pyspark`` DataFrame
        :param metrics: Словарь метрик, которые необходимо считать.
            Ключ -- метрика, значение -- ``int`` или список интов, обозначающих ``k``,
            для которых необходимо посчитать метрику.
        """
        self.test = convert(test)
        self.metrics = self._verify(metrics)
        self.df = pd.DataFrame()

    @staticmethod
    def _verify(metrics: Dict[Metric, Union[int, List[int]]]):
        """Проверяет корректность аргумента, конвертит инт в лист"""
        if not isinstance(metrics, dict):
            raise TypeError(f"metrics argument must be a dictionary, got {type(metrics)}")
        return metrics

    def add_result(self, name: str, pred: Any):
        """
        Подсчитать метрики для переданного списка рекомендаций

        :param name: имя модели/эксперимента для сохранения результатов
        :param pred: список рекомендаций для подсчета метрик
        """
        res = pd.Series(name=name)
        recs = convert(pred)
        for metric, k_list in self.metrics.items():
            values = metric(recs, self.test, k_list)
            if isinstance(k_list, int):
                res[f"{metric}@{k_list}"] = values
            else:
                for k, val in values.items():
                    res[f"{metric}@{k}"] = val
        self.df = self.df.append(res)
