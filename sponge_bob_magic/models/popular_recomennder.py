"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
import logging
import os
from typing import Dict, Optional

import numpy as np
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf
from sponge_bob_magic import constants, utils
from sponge_bob_magic.models.base_recommender import BaseRecommender


class PopularRecommender(BaseRecommender):
    """ простейший рекомендатель на основе сглаженной популярности """
    items_popularity: Optional[DataFrame]

    def __init__(self, spark: SparkSession, alpha: float = 0.001,
                 beta: float = 0.001):
        super().__init__(spark)

        self.alpha = alpha
        self.beta = beta

        self.items_popularity = None

    def get_params(self) -> Dict[str, object]:
        return {'alpha': self.alpha,
                'beta': self.beta}

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame],
             item_features: Optional[DataFrame],
             path: Optional[str] = None) -> None:
        popularity = (log
                      .groupBy('item_id', 'context')
                      .count())

        self.items_popularity = popularity.select(
            'item_id', 'context', 'count'
        )

        # считаем среднее кол-во просмотренных items у каждого user
        self.k_fake = np.ceil(log
                              .select('user_id', 'item_id')
                              .groupBy('user_id')
                              .count()
                              .select(sf.mean(sf.col('count')).alias('mean'))
                              .collect()[0]['mean'])
        logging.debug(
            f"Среднее количество items у каждого user: {self.k_fake}")

        if path is not None:
            path_parquet = os.path.join(path, 'items_popularity.parquet')
            self.items_popularity.write.parquet(path_parquet)
            self.items_popularity = self.spark.read.parquet(path_parquet)
        else:
            self.items_popularity.checkpoint()

    def _predict(self,
                 k: int,
                 users: DataFrame,
                 items: DataFrame,
                 context: Optional[str],
                 log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 to_filter_seen_items: bool = True,
                 path: Optional[str] = None) -> DataFrame:
        items_to_rec = self.items_popularity

        if context is None or context == constants.DEFAULT_CONTEXT:
            items_to_rec = (items_to_rec
                            .select('item_id', 'count')
                            .groupBy('item_id')
                            .agg(sf.sum('count').alias('count')))
            items_to_rec = (items_to_rec
                            .withColumn('context',
                                        sf.lit(constants.DEFAULT_CONTEXT)))
        else:
            items_to_rec = (items_to_rec
                            .filter(items_to_rec['context'] == context))

        count_sum = (items_to_rec
                     .groupBy()
                     .agg(sf.sum("count"))
                     .collect()[0][0])

        items_to_rec = (items_to_rec
                        .withColumn('relevance',
                                    (sf.col('count') + self.alpha) /
                                    (count_sum + self.beta))
                        .drop('count'))

        # удаляем ненужные items и добавляем нулевые
        items = (items
                 .join(items_to_rec, on='item_id', how='left'))
        items = items.na.fill({'context': context,
                               'relevance': 0})

        items = utils.get_top_k_rows(items, k + self.k_fake, 'relevance')

        logging.debug(f"Количество items после фильтрации: {items.count()}")

        # (user_id, item_id, context, relevance)
        recs = users.crossJoin(items)
        logging.debug(f"Длина recs: {recs.count()}")

        if to_filter_seen_items:
            recs = self._filter_seen_recs(recs, log)

        # берем топ-к
        recs = self._get_top_k_recs(recs, k)

        # заменяем отрицательные рейтинги на 0
        # (они помогали отобрать в топ-k невиденные айтемы)
        recs = (recs
                .withColumn('relevance',
                            sf.when(recs['relevance'] < 0, 0)
                            .otherwise(recs['relevance'])))

        if path is not None:
            path_parquet = os.path.join(path, 'recs.parquet')
            recs.write.parquet(path_parquet)
            recs = self.spark.read.parquet(path_parquet)
        else:
            recs.checkpoint()

        return recs


if __name__ == '__main__':
    spark_ = (SparkSession
              .builder
              .master('local[1]')
              .config('spark.driver.memory', '512m')
              .config("spark.sql.shuffle.partitions", "1")
              .appName('testing-pyspark')
              .enableHiveSupport()
              .getOrCreate())

    spark_.sparkContext.setCheckpointDir(os.environ['SPONGE_BOB_CHECKPOINTS'])

    data = [
        ["user1", "item1", 1.0, 'context1', "timestamp"],
        ["user2", "item3", 2.0, 'context1', "timestamp"],
        ["user1", "item2", 1.0, 'context2', "timestamp"],
        ["user3", "item3", 2.0, 'context1', "timestamp"],
    ]
    schema = ['user_id', 'item_id', 'relevance', 'context', 'timestamp']
    log_ = spark_.createDataFrame(data=data,
                                  schema=schema)

    users_ = ["user1", "user2", "user3"]
    items_ = ["item1", "item2", "item3"]

    pr = PopularRecommender(spark_, alpha=0, beta=0)
    recs_ = pr.fit_predict(k=3, users=users_, items=items_,
                           context='context1',
                           log=log_,
                           user_features=None, item_features=None,
                           to_filter_seen_items=False)

    recs_.show()
