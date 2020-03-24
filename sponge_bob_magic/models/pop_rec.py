"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import get_top_k_recs


class PopRec(Recommender):
    """
    Базовый рекомендатель на основе популярности.

    Популярность объекта определяется как вероятность того,
    что случайно выбранный пользователь взаимодействовал с объектом:

    .. math::
        Popularity(i) = \\dfrac{N_i}{N}

    :math:`N_i` - количество пользователей, у которых было взаимодействие с
    объектом :math:`i`

    :math:`N` - общее количество пользователей,
    независимо от взаимодействия с объектом.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"user_id": [1, 1, 2, 2, 3], "item_id": [1, 2, 3, 3, 3]})
    >>> df
       user_id  item_id
    0        1        1
    1        1        2
    2        2        3
    3        2        3
    4        3        3

    >>> from sponge_bob_magic.converter import convert
    >>> res = PopRec().fit_predict(convert(df), 1)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
       user_id  item_id  relevance
    0        1        3   0.666667
    1        2        2   0.333333
    2        3        1   0.333333

    >>> res = PopRec().fit_predict(convert(df), 1, filter_seen_items=False)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
       user_id  item_id  relevance
    0        1        3        0.666667
    1        2        3        0.666667
    2        3        3        0.666667
    """
    items_popularity: DataFrame

    def get_params(self) -> Dict[str, object]:
        return {}

    def _pre_fit(self,
                 log: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None) -> None:
        super()._pre_fit(log, user_features, item_features)
        popularity = (
            log
            .groupBy("item_id")
            .agg(sf.countDistinct("user_id").alias("user_count"))
        )
        self.items_popularity = popularity.select(
            "item_id",
            (
                sf.col("user_count") / sf.lit(self.users_count)
            ).alias("relevance")
        ).cache()

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        pass

    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: DataFrame,
                 items: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        # удаляем ненужные items и добавляем нулевые
        items = items.join(
            self.items_popularity,
            on="item_id",
            how="left"
        )
        items = items.na.fill({"relevance": 0})
        # (user_id, item_id, relevance)
        recs = users.crossJoin(items)
        if filter_seen_items:
            recs = self._filter_seen_recs(recs, log)
        # берем топ-к
        recs = get_top_k_recs(recs, k)
        # заменяем отрицательные рейтинги на 0
        # (они помогали отобрать в топ-k невиденные айтемы)
        recs = (recs
                .withColumn("relevance",
                            sf.when(recs["relevance"] < 0, 0)
                            .otherwise(recs["relevance"]))).cache()
        return recs
