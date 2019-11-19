"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf

from sponge_bob_magic import constants, utils
from sponge_bob_magic.models.base_recommender import BaseRecommender


class PopularRecommender(BaseRecommender):
    """ Простейший рекомендатель на основе сглаженной популярности. """
    avg_num_items: int
    items_popularity: Optional[DataFrame]

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
            f"Среднее количество items у каждого user: {self.avg_num_items}")

        if path is not None:
            self.items_popularity = utils.write_read_dataframe(
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

        if context == constants.DEFAULT_CONTEXT:
            items_to_rec = (items_to_rec
                            .select("item_id", "count")
                            .groupBy("item_id")
                            .agg(sf.sum("count").alias("count")))
            items_to_rec = (items_to_rec
                            .withColumn("context",
                                        sf.lit(constants.DEFAULT_CONTEXT)))
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

        items = utils.get_top_k_rows(items, k + self.avg_num_items,
                                     "relevance")

        logging.debug(f"Количество items после фильтрации: {items.count()}")

        # (user_id, item_id, context, relevance)
        recs = users.crossJoin(items)

        if to_filter_seen_items:
            recs = self._filter_seen_recs(recs, log)

        # берем топ-к
        recs = self._get_top_k_recs(recs, k)

        # заменяем отрицательные рейтинги на 0
        # (они помогали отобрать в топ-k невиденные айтемы)
        recs = (recs
                .withColumn("relevance",
                            sf.when(recs["relevance"] < 0, 0)
                            .otherwise(recs["relevance"]))).cache()

        if path is not None:
            recs = utils.write_read_dataframe(
                self.spark, recs,
                os.path.join(path, "recs.parquet"),
                self.to_overwrite_files)

        return recs


if __name__ == "__main__":
    spark_ = (SparkSession
              .builder
              .master("local[1]")
              .config("spark.driver.memory", "512m")
              .config("spark.sql.shuffle.partitions", "1")
              .appName("testing-pyspark")
              .enableHiveSupport()
              .getOrCreate())

    data = [
        ["user1", "item1", datetime(2019, 1, 1), "context1", 1.0],
        ["user2", "item3", datetime(2019, 1, 1), "context1", 2.0],
        ["user1", "item2", datetime(2019, 1, 1), "context2", 1.0],
        ["user3", "item3", datetime(2019, 1, 1), "context1", 2.0],
    ]
    log_ = spark_.createDataFrame(data=data,
                                  schema=constants.LOG_SCHEMA)

    users_ = ["user1", "user2", "user3"]
    items_ = ["item1", "item2", "item3"]

    users_ = spark_.createDataFrame(data=[[user] for user in users_],
                                    schema=["user_id"])
    items_ = spark_.createDataFrame(data=[[item] for item in items_],
                                    schema=["item_id"])

    pr = PopularRecommender(spark_, alpha=0, beta=0)
    recs_ = pr.fit_predict(k=3, users=users_, items=items_,
                           context="context1",
                           log=log_,
                           user_features=None, item_features=None,
                           to_filter_seen_items=False)

    recs_.show()
