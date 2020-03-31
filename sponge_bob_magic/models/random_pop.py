"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import get_top_k_recs


class RandomPop(Recommender):
    """
    Рандомизированный рекомендатель на основе популярности.

    Вероятность того, что заданный объект :math:`i` будет порекомендован этой моделью любому пользователю
    не зависит от пользователя и определяется как аддитивное сглаживание оценки вероятности того,
    что случайно выбранный пользователь взаимодействовал с объектом:

    .. math::
        P\\left(i\\right)\\propto N_i + \\alpha

    :math:`N_i` --- количество пользователей, у которых было взаимодействие с объектом :math:`i`

    :math:`\\alpha` --- параметр сглаживания (гипер-параметр модели). По умолчанию :math:`\\alpha = 0`.
    Чем больше :math:`\\alpha`, тем чаще будут рекомендоваться менее популярные объекты.
    Требуется, чтобы всегда было :math:`\\alpha > -1`.

    >>> import pandas as pd
    >>> from sponge_bob_magic.converter import convert
    >>>
    >>> log = convert(pd.DataFrame({
    ...     "user_id": ["1", "1", "2", "2", "3"],
    ...     "item_id": ["1", "2", "3", "3", "3"]
    ... }))
    >>> log.show()
    +-------+-------+
    |user_id|item_id|
    +-------+-------+
    |      1|      1|
    |      1|      2|
    |      2|      3|
    |      2|      3|
    |      3|      3|
    +-------+-------+

    >>> random_pop = RandomPop(alpha=-1)
    Traceback (most recent call last):
     ...
    ValueError: alpha должно быть строго больше -1

    >>> random_pop = RandomPop(alpha=1.0, seed=777)
    >>> random_pop.get_params()
    {'alpha': 1.0, 'seed': 777}
    >>> random_pop.fit(log)
    >>> print(random_pop.model_items, random_pop.model_proba)
    [0 2 1] [0.42857143 0.28571429 0.28571429]

    >>> recs = random_pop.predict(log, 2)
    >>> recs.show()
    +-------+----------+-------+
    |user_id| relevance|item_id|
    +-------+----------+-------+
    |      3|       0.5|      2|
    |      3|0.33333334|      1|
    |      1|       0.5|      3|
    |      1|      -1.0|      1|
    |      2|       0.5|      1|
    |      2|0.33333334|      2|
    +-------+----------+-------+

    """
    items_popularity: DataFrame
    model_items: np.ndarray
    model_proba: np.ndarray

    def __init__(self, alpha: float = 0.0, seed: Optional[int] = None):
        """
        :param alpha: параметр аддитивного сглаживания. Чем он больше, тем чаще рекомендуются непопулярные объекты.
        :param seed: инициализация генератора псевдослучайности
        """
        if alpha <= -1.0:
            raise ValueError("alpha должно быть строго больше -1")
        self.alpha = alpha
        self.seed = seed

    def get_params(self) -> Dict[str, object]:
        return {"alpha": self.alpha, "seed": self.seed}

    def _pre_fit(self,
                 log: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None) -> None:
        super()._pre_fit(log, user_features, item_features)
        self.items_popularity = (
            log
            .groupBy("item_id")
            .agg(sf.countDistinct("user_id").alias("user_count"))
        )

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        probabilities = (
            self.item_indexer.transform(self.items_popularity).selectExpr(
                "CAST(item_idx AS INT) AS item_idx",
                f"CAST(user_count + {self.alpha} AS FLOAT) AS probability"
            )
        ).collect()
        model_items, model_proba = zip(*probabilities)
        self.model_items = np.array(model_items)
        self.model_proba = np.array(model_proba) / sum(model_proba)

    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: DataFrame,
                 items: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        model_items = self.model_items
        model_proba = self.model_proba
        seed = self.seed

        @sf.pandas_udf(
            st.StructType([
                st.StructField("user_id", st.StringType(), True),
                st.StructField("user_idx", st.LongType(), True),
                st.StructField("item_idx", st.LongType(), True),
                st.StructField("relevance", st.FloatType(), True)
            ]),
            sf.PandasUDFType.GROUPED_MAP
        )
        def grouped_map(pandas_df):
            user_idx = pandas_df["user_idx"][0]
            user_id = pandas_df["user_id"][0]
            cnt = pandas_df["cnt"][0]
            if seed is not None:
                np.random.seed(user_idx + seed)
            items_idx = np.random.choice(
                model_items,
                size=cnt,
                p=model_proba,
                replace=False
            )
            relevance = 1 / np.arange(1, cnt + 1)
            return pd.DataFrame({
                "user_id": cnt * [user_id],
                "user_idx": cnt * [user_idx],
                "item_idx": items_idx,
                "relevance": relevance
            })
        model_len = len(model_items)
        recs = self.inv_item_indexer.transform(
            self.user_indexer.transform(
                users.join(log, how="left", on="user_id")
                .select("user_id", "item_id")
                .groupby("user_id")
                .agg(sf.countDistinct("item_id").alias("cnt"))
            )
            .selectExpr(
                "user_id",
                "CAST(user_idx AS INT) AS user_idx",
                f"CAST(LEAST(cnt + {k}, {model_len}) AS INT) AS cnt"
            )
            .groupby("user_id", "user_idx")
            .apply(grouped_map)
        ).drop("item_idx", "user_idx")
        if filter_seen_items:
            recs = self._filter_seen_recs(recs, log)
        recs = get_top_k_recs(recs, k)
        return recs
