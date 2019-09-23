import logging
from datetime import datetime
from typing import Iterable, Dict, List

import joblib
import optuna
from pyspark.sql import SparkSession, DataFrame

from sponge_bob_magic.metrics.metrics import Metrics
from sponge_bob_magic.models.base_recommender import BaseRecommender
from sponge_bob_magic.models.popular_recomennder import PopularRecommender
from sponge_bob_magic.validation_schemes import ValidationSchemes


class BigRecommender(BaseRecommender):
    def __init__(self, spark: SparkSession,
                 date: datetime, path_tmp: str or None = None):
        super().__init__(spark)

        self.splitter = ValidationSchemes(spark)
        self.split_date = date

        self.path_optuna_study = path_tmp
        self.optuna_study = None

    @staticmethod
    def get_distinct_values_as_list(df, column) -> List:
        return df.select(column).distinct().rdd.flatMap(lambda x: x).collect()

    def _fit(self, log: DataFrame,
             user_features: DataFrame or None,
             item_features: DataFrame or None) -> None:
        # ToDO: куда передавать параметры для оптимизации гиперпараметров?
        train, _, test = self.splitter.log_split_by_date(
            log, self.split_date,
            drop_cold_items=True, drop_cold_users=True)

        self.model = PopularRecommender(self.spark)
        users = self.get_distinct_values_as_list(log, 'user_id')
        items = self.get_distinct_values_as_list(log, 'item_id')

        def objective(trial):
            if self.path_optuna_study is not None:
                joblib.dump(self.optuna_study, self.path_optuna_study)

            alpha = trial.suggest_uniform('alpha', 0.0, 1.0)
            beta = trial.suggest_uniform('beta', 0.0, 1.0)

            params = {'alpha': alpha,
                      'beta': beta}
            self.model.set_params(**params)

            recs = self.model.fit_predict(k=15,
                                          users=users, items=items,
                                          user_features=user_features,
                                          item_features=item_features,
                                          context='no_context',
                                          log=train,
                                          to_filter_seen_items=True)

            return Metrics.hit_rate_at_k(recs, train, k=10)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=2, n_jobs=2)

        logging.debug(f"Best value: {study.best_value}")

        self.model.set_params(**study.best_params)
        self.model.fit(log, user_features, item_features)

    def get_params(self) -> Dict[str, object]:
        """

        :return:
        """

    def _predict(self,
                 k: int,
                 users: Iterable or DataFrame,
                 items: Iterable or DataFrame,
                 context: str or None,
                 log: DataFrame,
                 user_features: DataFrame or None,
                 item_features: DataFrame or None,
                 to_filter_seen_items: bool = True) -> DataFrame:
        """

        :param k:
        :param users:
        :param items:
        :param context:
        :param log:
        :param user_features:
        :param item_features:
        :param to_filter_seen_items:
        :return:
        """
        return self.model.predict(k, users, items, log,
                                  user_features, item_features,
                                  to_filter_seen_items)


if __name__ == '__main__':
    spark_ = (SparkSession
              .builder
              .master('local[1]')
              .config('spark.driver.memory', '512m')
              .appName('testing-pyspark')
              .enableHiveSupport()
              .getOrCreate())

    data = [
        ["user1", "item1", 1.0, 'no_context', datetime(2019, 10, 10)],
        ["user2", "item3", 2.0, 'no_context', datetime(2019, 10, 11)],
        ["user1", "item2", 1.0, 'no_context', datetime(2019, 10, 12)],
        ["user3", "item3", 2.0, 'no_context', datetime(2019, 10, 13)],
    ]
    schema = ['user_id', 'item_id', 'relevance', 'context', 'timestamp']
    log_ = spark_.createDataFrame(data=data,
                                  schema=schema)

    br = BigRecommender(spark_, date=datetime(2019, 10, 12))

    br.fit(log_, user_features=None, item_features=None)
