"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from sponge_bob_magic.constants import CommonDataFrame
from sponge_bob_magic.converter import convert
from sponge_bob_magic.metrics.base_metric import RecOnlyMetric
from sponge_bob_magic.models import PopRec
from sponge_bob_magic.models.base_rec import Recommender


class Unexpectedness(RecOnlyMetric):
    """
    Доля объектов в рекомендациях, которая не содержится в рекомендациях некоторого базового алгоритма.
    По умолчанию используется рекомендатель по популярности ``PopRec``.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"user_id": [1, 1, 2, 3], "item_id": [1, 2, 1, 3], "relevance": [5, 5, 5, 5], "timestamp": [1, 1, 1, 1]})
    >>> dd = pd.DataFrame({"user_id": [1, 2, 1, 2], "item_id": [1, 2, 3, 1], "relevance": [5, 5, 5, 5], "timestamp": [1, 1, 1, 1]})
    >>> m = Unexpectedness(df)
    >>> m(dd, [1, 2])
    {1: 0.5, 2: 0.25}


    Возможен также режим, в котором рекомендации базового алгоритма передаются сразу при инициализации и рекомендатель не обучается

    >>> de = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [1, 2, 3], "relevance": [5, 5, 5], "timestamp": [1, 1, 1]})
    >>> dr = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [0, 0, 1], "relevance": [5, 5, 5], "timestamp": [1, 1, 1]})
    >>> m = Unexpectedness(dr, None)
    >>> round(m(de, 3), 2)
    0.67
    """

    def __init__(self, log: CommonDataFrame, rec: Recommender = PopRec()):
        """
        Есть два варианта инициализации в зависимости от значения параметра ``rec``.
        Если ``rec`` -- рекомендатель, то ``log`` считается данными для обучения.
        Если ``rec is None``, то ``log`` считается готовыми предсказаниями какой-то внешней модели,
        с которой необходимо сравниться.

        :param log: пандас или спарк датафрейм
        :param rec: одна из проинициализированных моделей библиотеки, либо ``None``
        """
        self.log = convert(log)
        if rec is None:
            self.train_model = False
        else:
            self.train_model = True
            rec.fit(log=self.log)
            self.model = rec

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        recs = pandas_df["item_id"]
        pandas_df["cum_agg"] = pandas_df.apply(
            lambda row:
            (
                row["k"] -
                np.isin(
                    recs[:row["k"]],
                    row["items_id"][:row["k"]]
                ).sum()
            ) / row["k"],
            axis=1)
        return pandas_df

    def _get_enriched_recommendations(
            self,
            recommendations: DataFrame,
            ground_truth: DataFrame
    ) -> DataFrame:
        if self.train_model:
            pred = self.model.predict(log=self.log, k=self.max_k)
        else:
            pred = self.log
        items_by_users = (pred
                          .groupby("user_id").agg(
                              sf.collect_list("item_id").alias("items_id")))
        res = recommendations.join(
            items_by_users,
            how="inner",
            on=["user_id"]
        )
        return res
