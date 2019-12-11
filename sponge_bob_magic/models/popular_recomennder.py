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
from sponge_bob_magic.models.base_recommender import BaseRecommender
from sponge_bob_magic.utils import (get_top_k_recs, get_top_k_rows,
                                    write_read_dataframe)


class PopularRecommender(BaseRecommender):
    """ Простейший рекомендатель на основе сглаженной популярности. """
    avg_num_items: int
    items_popularity: DataFrame

    def __init__(self, spark: SparkSession,
                 alpha: float = 1000,
                 beta: float = 1000):
        super().__init__(spark)

        self.alpha = alpha
        self.beta = beta

    def get_params(self) -> Dict[str, object]:
        return {"alpha": self.alpha,
                "beta": self.beta}

    def _pre_fit(self, log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 path: Optional[str] = None) -> None:
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

        if path is not None:
            self.items_popularity = write_read_dataframe(
                self.spark, self.items_popularity,
                os.path.join(path, "items_popularity.parquet"),
                self.to_overwrite_files)

    def _fit_partial(self, log: DataFrame,
                     user_features: Optional[DataFrame],
                     item_features: Optional[DataFrame],
                     path: Optional[str] = None) -> None:
        pass

    def _predict(self,
                 k: int,
                 users: DataFrame,
                 items: DataFrame,
                 context: str,
                 log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 to_filter_seen_items: bool = True,
                 path: Optional[str] = None) -> DataFrame:
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
                        .withColumn("relevance",
                                    (sf.col("count") + self.alpha) /
                                    (count_sum + self.beta))
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

        if to_filter_seen_items:
            recs = self._filter_seen_recs(recs, log)

        # берем топ-к
        recs = get_top_k_recs(recs, k)

        # заменяем отрицательные рейтинги на 0
        # (они помогали отобрать в топ-k невиденные айтемы)
        recs = (recs
                .withColumn("relevance",
                            sf.when(recs["relevance"] < 0, 0)
                            .otherwise(recs["relevance"]))).cache()

        if path is not None:
            recs = write_read_dataframe(
                self.spark, recs,
                os.path.join(path, "recs.parquet"),
                self.to_overwrite_files)

        return recs
