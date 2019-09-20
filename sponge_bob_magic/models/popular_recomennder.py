from typing import Iterable, Dict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as sf

from sponge_bob_magic.models.base_recommender import BaseRecommender


class PopularRecommender(BaseRecommender):
    items_popularity: DataFrame or None

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
             user_features: DataFrame or None,
             item_features: DataFrame or None) -> None:
        popularity = log \
            .groupBy('item_id', 'context') \
            .count()

        self.items_popularity = popularity.select(
            'item_id', 'context', 'count'
        )

    def _predict(self,
                 k: int,
                 users: Iterable or DataFrame,
                 items: Iterable or DataFrame,
                 context: str or None,
                 log: DataFrame,
                 user_features: DataFrame or None,
                 item_features: DataFrame or None,
                 to_filter_seen_items: bool = True) -> DataFrame:
        items_to_rec = self.items_popularity \
            .filter(self.items_popularity['item_id'].isin(items))

        if context is None or context == 'no_context':
            items_to_rec = items_to_rec \
                .select('item_id', 'count') \
                .groupBy('item_id') \
                .agg(sf.sum('count').alias('count'))
            items_to_rec = items_to_rec \
                .withColumn('context', sf.lit('no_context'))
        else:
            items_to_rec = items_to_rec \
                .filter(items_to_rec['context'] == context)

        count_sum = items_to_rec \
            .select('count') \
            .rdd \
            .map(lambda x: (1, x[0])) \
            .reduceByKey(lambda x, y: x + y) \
            .collect()[0][1]

        items_to_rec = items_to_rec \
            .withColumn('relevance',
                        (sf.col('count') + self.alpha) /
                        (count_sum + self.beta)) \
            .drop('count')

        # (user_id, item_id, context, relevance)
        recs = users.crossJoin(items_to_rec)
        return recs


if __name__ == '__main__':
    spark_ = (SparkSession
              .builder
              .master('local[1]')
              .config('spark.driver.memory', '512m')
              .appName('testing-pyspark')
              .enableHiveSupport()
              .getOrCreate())

    data = [
        ["user1", "item1", 1.0, 'context1'],
        ["user2", "item3", 2.0, 'context1'],
        ["user1", "item2", 1.0, 'context2'],
        ["user3", "item3", 2.0, 'context1'],
    ]
    schema = ['user_id', 'item_id', 'relevance', 'context']
    log_ = spark_.createDataFrame(data=data,
                                  schema=schema)

    users_ = ["user1", "user2", "user3"]
    items_ = ["item1", "item2", "item3"]

    pr = PopularRecommender(spark_, alpha=0, beta=0)
    pr.fit(log_, user_features=None, item_features=None)
    recs = pr.predict(k=10, users=users_, items=items_, context='context1',
                      log=log_,
                      user_features=None, item_features=None,
                      to_filter_seen_items=False)

    recs.show()
