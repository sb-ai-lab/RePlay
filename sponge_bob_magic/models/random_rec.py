"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Optional

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from sponge_bob_magic.models.base_rec import Recommender


class RandomRec(Recommender):
    """
    Рандомизированный рекомендатель на основе популярности.

    Вероятность того, что заданный объект :math:`i` будет порекомендован этой моделью любому пользователю
    не зависит от пользователя. В зависисимости от выбранного распределения
    вероятность выбора элемента будет либо одинаковой для всех элементов,
    либо определяться как аддитивное сглаживание оценки вероятности того,
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
    <BLANKLINE>
    >>> random_pop = RandomRec(distribution="popular_based", alpha=-1)
    Traceback (most recent call last):
     ...
    ValueError: alpha должно быть строго больше -1

    >>> random_pop = RandomRec(distribution="abracadabra")
    Traceback (most recent call last):
     ...
    ValueError: distribution должно быть popular_based или uniform

    >>> random_pop = RandomRec(distribution="popular_based", alpha=1.0, seed=777)
    >>> random_pop.fit(log)
    >>> random_pop.item_popularity.show()
    +--------+-----------+
    |item_idx|probability|
    +--------+-----------+
    |       2|        2.0|
    |       1|        2.0|
    |       0|        3.0|
    +--------+-----------+
    <BLANKLINE>
    >>> recs = random_pop.predict(log, 2)
    >>> recs.show()
    +-------+-------+----------+
    |user_id|item_id| relevance|
    +-------+-------+----------+
    |      1|      3|       1.0|
    |      1|      2|       0.0|
    |      2|      1|       1.0|
    |      2|      2|       0.5|
    |      3|      1|       1.0|
    |      3|      2|0.33333334|
    +-------+-------+----------+
    <BLANKLINE>
    >>> random_pop = RandomRec(seed=555)
    >>> random_pop.fit(log)
    >>> random_pop.item_popularity.show()
    +--------+-----------+
    |item_idx|probability|
    +--------+-----------+
    |       2|          1|
    |       1|          1|
    |       0|          1|
    +--------+-----------+
    <BLANKLINE>
    >>> recs = random_pop.predict(log, 2)
    >>> recs.show()
    +-------+-------+----------+
    |user_id|item_id| relevance|
    +-------+-------+----------+
    |      1|      3|       1.0|
    |      1|      2|       0.0|
    |      2|      1|       0.5|
    |      2|      2|0.33333334|
    |      3|      2|       0.5|
    |      3|      1|0.33333334|
    +-------+-------+----------+
    <BLANKLINE>
    """

    item_popularity: DataFrame
    can_predict_cold_users = True

    def __init__(
        self,
        distribution: str = "uniform",
        alpha: float = 0.0,
        seed: Optional[int] = None,
    ):
        """
        :param distribution: вероятностоное распределение выбора элементов.
            "uniform" - равномерное, все элементы выбираются с равной вероятнсотью
            "popular_based" - аддитивное сглаживание оценки вероятности того,
            что случайно выбранный пользователь взаимодействовал с объектом
        :param alpha: параметр аддитивного сглаживания. Чем он больше, тем чаще рекомендуются непопулярные объекты.
        :param seed: инициализация генератора псевдослучайности
        """
        if distribution not in ("popular_based", "uniform"):
            raise ValueError(
                "distribution должно быть popular_based " "или uniform"
            )
        if alpha <= -1.0 and distribution == "popular_based":
            raise ValueError("alpha должно быть строго больше -1")
        self.distribution = distribution
        self.alpha = alpha
        self.seed = seed

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        if self.distribution == "popular_based":
            probability = f"CAST(user_count + {self.alpha} AS FLOAT)"
        else:
            probability = "1"
        self.item_popularity = log.groupBy("item_idx").agg(
            sf.countDistinct("user_idx").alias("user_count")
        )
        self.item_popularity = self.item_popularity.selectExpr(
            "item_idx", f"{probability} AS probability"
        ).cache()

    # pylint: disable=too-many-arguments
    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:

        items_pd = (
            items.join(
                self.item_popularity.withColumnRenamed("item_idx", "item"),
                on=sf.col("item_idx") == sf.col("item"),
                how="inner",
            )
            .drop("item")
            .toPandas()
        )
        items_pd.loc[:, "probability"] = (
            items_pd["probability"] / items_pd["probability"].sum()
        )
        seed = self.seed

        @sf.pandas_udf(
            st.StructType(
                [
                    st.StructField("user_idx", st.IntegerType(), True),
                    st.StructField("item_idx", st.IntegerType(), True),
                    st.StructField("relevance", st.FloatType(), True),
                ]
            ),
            sf.PandasUDFType.GROUPED_MAP,
        )
        def grouped_map(pandas_df):
            user_idx = pandas_df["user_idx"][0]
            cnt = pandas_df["cnt"][0]
            if seed is not None:
                np.random.seed(user_idx + seed)
            items_idx = np.random.choice(
                items_pd["item_idx"].values,
                size=cnt,
                p=items_pd["probability"].values,
                replace=False,
            )
            relevance = 1 / np.arange(1, cnt + 1)
            return pd.DataFrame(
                {
                    "user_idx": cnt * [user_idx],
                    "item_idx": items_idx,
                    "relevance": relevance,
                }
            )

        model_len = len(items_pd)
        recs = (
            users.join(log, how="left", on="user_idx")
            .select("user_idx", "item_idx")
            .groupby("user_idx")
            .agg(sf.countDistinct("item_idx").alias("cnt"))
            .selectExpr(
                "user_idx AS user_idx",
                f"LEAST(cnt + {k}, {model_len}) AS cnt",
            )
            .groupby("user_idx")
            .apply(grouped_map)
        )

        return recs
