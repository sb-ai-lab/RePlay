"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from typing import Dict, Optional

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf

from sponge_bob_magic.constants import DEFAULT_CONTEXT
from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import (get_top_k_recs, get_top_k_rows,
                                    write_read_dataframe)


class PopRec(Recommender):
    """
    Простейший рекомендатель на основе популярности.

    Популярность объекта определяется как:
    popularity(i) = \\dfrac{N_i}{N},

    где $ N_i $ - количество пользователей, у которых было взаимодействие с
    данным объектом $ i $, $ N $ - общее количество пользователей,
    которые как провзаимодействовали с объектом, так и нет.
    """

    avg_num_items: int
    items_popularity: DataFrame

    def get_params(self) -> Dict[str, object]:
        return {}

    def _pre_fit(self,
                 log: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None) -> None:
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
        spark = SparkSession(log.rdd.context)
        self.items_popularity = write_read_dataframe(
            self.items_popularity,
            os.path.join(spark.conf.get("spark.local.dir"),
                         "items_popularity.parquet")
        )

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

        if context == DEFAULT_CONTEXT:
            items_to_rec = (items_to_rec
                            .select("item_id", "count")
                            .groupBy("item_id")
                            .agg(sf.sum("count").alias("count")))
            items_to_rec = (items_to_rec
                            .withColumn("context",
                                        sf.lit(DEFAULT_CONTEXT)))
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
