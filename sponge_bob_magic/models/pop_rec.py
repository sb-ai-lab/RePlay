"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from typing import Dict, Optional

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from sponge_bob_magic.constants import DEFAULT_CONTEXT
from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import get_top_k_recs, get_top_k_rows


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

    Умеет учитывать контекст оценок при подсчете значения популярности.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"user_id": [1, 1, 2, 2, 3], "item_id": [1, 2, 3, 3, 3], "context": [1, 1, 1, 2, 2]})
    >>> df
       user_id  item_id  context
    0        1        1        1
    1        1        2        1
    2        2        3        1
    3        2        3        2
    4        3        3        2

    >>> from sponge_bob_magic.converter import convert
    >>> res = PopRec().fit_predict(convert(df), 1, context=1)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
       user_id  item_id  context  relevance
    0        1        3        1   0.333333
    1        2        2        1   0.333333
    2        3        1        1   0.333333

    >>> res = PopRec().fit_predict(convert(df), 1, filter_seen_items=False)
    >>> res.toPandas().sort_values("user_id", ignore_index=True)
       user_id  item_id     context  relevance
    0        1        3  no_context        0.6
    1        2        3  no_context        0.6
    2        3        3  no_context        0.6
    """

    avg_num_items: int
    items_popularity: DataFrame

    def get_params(self) -> Dict[str, object]:
        return {}

    def _pre_fit(self,
                 log: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None) -> None:

        if "context" not in log.columns:
            log = log.withColumn("context", sf.lit(DEFAULT_CONTEXT))

        popularity = (log
                      .groupBy("item_id", "context")
                      .count())

        self.items_popularity = popularity.select(
            "item_id", "context", "count"
        ).cache()

        # считаем среднее кол-во просмотренных items у каждого user
        self.avg_num_items = np.ceil(
            log
            .select("user_id", "item_id")
            .groupBy("user_id")
            .count()
            .select(sf.mean(sf.col("count")).alias("mean"))
            .collect()[0]["mean"]
        )
        logging.debug(
            "Среднее количество items у каждого user: %d", self.avg_num_items)

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        pass

    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: DataFrame = None,
                 items: DataFrame = None,
                 context: str = None,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        items_to_rec = self.items_popularity

        if context is None or context == DEFAULT_CONTEXT:
            items_to_rec = (items_to_rec
                            .select("item_id", "count")
                            .groupBy("item_id")
                            .agg(sf.sum("count").alias("count")))
            items_to_rec = (items_to_rec
                            .withColumn("context",
                                        sf.lit(DEFAULT_CONTEXT)))
            context = DEFAULT_CONTEXT
        else:
            items_to_rec = (items_to_rec
                            .filter(items_to_rec["context"] == context))

        count_sum = (items_to_rec
                     .groupBy()
                     .agg(sf.sum("count"))
                     .collect()[0][0])

        items_to_rec = (items_to_rec
                        .withColumn("relevance", sf.col("count") / count_sum)
                        .drop("count"))

        # удаляем ненужные items и добавляем нулевые
        items = items.join(
            items_to_rec,
            on="item_id",
            how="left"
        )
        items = items.na.fill({"context": context,
                               "relevance": 0})
        items = get_top_k_rows(
            items, k + self.avg_num_items,
            "relevance"
        )
        logging.debug("Количество items после фильтрации: %d", items.count())

        # (user_id, item_id, context, relevance)
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
