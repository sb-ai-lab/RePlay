"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from sponge_bob_magic.constants import IntOrList
from sponge_bob_magic.converter import convert
from sponge_bob_magic.metrics.base_metric import Metric, RecOnlyMetric


# pylint: disable=too-few-public-methods
class Experiment:
    """
    Обеспечивает подсчет и хранение значений метрик.

    Инициализируется тестом, на котором нужно считать метрики и словарём метрики-значения k.

    Результаты доступны в атрибуте ``pandas_df`` в виде pandas.DataFrame.

    Пример:

    >>> import pandas as pd
    >>> from sponge_bob_magic.metrics import NDCG, Surprisal
    >>> log = pd.DataFrame({"user_id": [2, 2, 2, 1], "item_id": [1, 2, 3, 3], "relevance": [5, 5, 5, 5]})
    >>> test = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [1, 2, 3], "relevance": [5, 3, 4]})
    >>> pred = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [4, 1, 3], "relevance": [5, 4, 5]})
    >>> recs = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [1, 4, 5], "relevance": [5, 4, 5]})
    >>> ex = Experiment(test, {NDCG(): [2, 3], Surprisal(log): 3}, calc_median=True, calc_sem=0.95)
    >>> ex.add_result("baseline", recs)
    >>> ex.add_result("model", pred)
    >>> ex.results
                NDCG@2  NDCG@2_median  NDCG@2_0.95_sem    NDCG@3  NDCG@3_median  NDCG@3_0.95_sem  Surprisal@3  Surprisal@3_median  Surprisal@3_0.95_sem
    baseline  0.613147       0.613147              0.0  0.469279       0.469279              0.0     1.000000            1.000000                   0.0
    model     0.386853       0.386853              0.0  0.530721       0.530721              0.0     0.666667            0.666667                   0.0
    >>> ex.compare("baseline")
               NDCG@2  NDCG@3 Surprisal@3
    baseline        –       –           –
    model     -36.91%  13.09%     -33.33%
    """

    def __init__(
        self,
        test: Any,
        metrics: Union[Dict[Metric, IntOrList], List[Metric]],
        k: Optional[IntOrList] = None,
        calc_median: bool = False,
        calc_sem: Optional[float] = None,
    ):
        """
        :param test: Данные для теста в формате ``pandas`` или ``pyspark`` DataFrame
        :param metrics: Словарь метрик, которые необходимо считать.
            Ключ -- метрика, значение -- ``int`` или список интов, обозначающих ``k``,
            для которых необходимо посчитать метрику.
        :param k: ``int`` или список интов, обозначающих ``k``,
            для которых необходимо посчитать метрику.
            Если указан, то параметр ``metrics`` должен быть списком.
        """
        self.test = convert(test)
        self.results = pd.DataFrame()
        if k is not None:
            if isinstance(k, int):
                k = [k] * len(metrics)
            self.metrics = dict(zip(metrics, k))
        else:
            self.metrics = metrics

        self.calc_median = calc_median
        self.calc_sem = calc_sem

    def add_result(self, name: str, pred: Any) -> None:
        """
        Подсчитать метрики для переданного списка рекомендаций

        :param name: имя модели/эксперимента для сохранения результатов
        :param pred: список рекомендаций для подсчета метрик
        """
        recs = convert(pred)
        for metric, k_list in sorted(
            self.metrics.items(), key=lambda x: str(x[0])
        ):

            if isinstance(metric, RecOnlyMetric):
                values = metric(recs, k_list)
            else:
                values = metric(recs, self.test, k_list)
            if self.calc_median:
                median = metric.median(recs, self.test, k_list)
            if self.calc_sem is not None:
                sem = metric.sem(recs, self.test, k_list, self.calc_sem)

            if isinstance(k_list, int):
                self.results.at[name, f"{metric}@{k_list}"] = values
                if self.calc_median:
                    self.results.at[name, f"{metric}@{k_list}_median"] = median
                if self.calc_sem is not None:
                    self.results.at[
                        name, f"{metric}@{k_list}_{self.calc_sem}_sem"
                    ] = sem
            else:
                for k, val in sorted(values.items(), key=lambda x: x[0]):
                    self.results.at[name, f"{metric}@{k}"] = val
                    if self.calc_median:
                        self.results.at[name, f"{metric}@{k}_median"] = median[
                            k
                        ]
                    if self.calc_sem is not None:
                        self.results.at[
                            name, f"{metric}@{k}_{self.calc_sem}_sem"
                        ] = sem[k]

    def compare(self, name: str) -> pd.DataFrame:
        """
        Показать процентный прирост относительно записи ``name``.

        :param name: имя модели/эксперимента, которая считается бейзлайном
        :return: таблица с приростом в процентах
        """
        if name not in self.results.index:
            raise ValueError(f"No results for model {name}")
        columns = [
            column for column in self.results.columns if column[-1].isdigit()
        ]
        data_frame = self.results[columns].copy()
        baseline = data_frame.loc[name]
        for idx in data_frame.index:
            if idx != name:
                diff = data_frame.loc[idx] / baseline - 1
                data_frame.loc[idx] = [
                    str(round(v * 100, 2)) + "%" for v in diff
                ]
            else:
                data_frame.loc[name] = ["–"] * len(baseline)
        return data_frame
