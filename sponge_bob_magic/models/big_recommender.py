from abc import abstractmethod
from datetime import datetime
from typing import Iterable, Dict

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

    def fit(self, log: DataFrame,
            user_features: DataFrame or None,
            item_features: DataFrame or None) -> None:
        train, _, test = self.splitter.log_split_by_date(
            log, self.split_date,
            drop_cold_items=True, drop_cold_users=True)

        self.model = PopularRecommender(self.spark)
        users = log.select('user_id').distinct().rdd.flatMap(lambda x: x).collect()
        items = log.select('item_id').distinct().rdd.flatMap(lambda x: x).collect()

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
            recs.show()

            return - Metrics.hit_rate_at_k(recs, train, k=10)

        study = optuna.create_study()
        study.optimize(objective, n_trials=2, n_jobs=2)

        print('best_value:', study.best_value)

    def _filter_seen_recs(self, recs: DataFrame, log: DataFrame) -> DataFrame:
        """

        :param recs:
        :param log:
        :return:
        """

    def _leave_top_recs(self, k: int, recs: DataFrame) -> DataFrame:
        """

        :param k:
        :param recs:
        :return:
        """

    def _get_batch_recs(self, users: Iterable,
                        items: Iterable,
                        context: str or None,
                        log: DataFrame,
                        user_features: DataFrame or None,
                        item_features: DataFrame or None,
                        to_filter_seen_items: bool = True) -> DataFrame:
        """

        :param users:
        :param items:
        :param context:
        :param log:
        :param user_features:
        :param item_features:
        :return:
        """

    def _get_single_recs(self,
                         user: str,
                         items: Iterable,
                         context: str or None,
                         log: DataFrame,
                         user_feature: DataFrame or None,
                         item_features: DataFrame or None,
                         to_filter_seen_items: bool = True
                         ) -> DataFrame:
        """

        :param user:
        :param items:
        :param context:
        :param log:
        :param user_feature:
        :param item_features:
        :param to_filter_seen_items:
        :return:
        """

    def get_params(self) -> Dict[str, object]:
        """

        :return:
        """

    def predict(self,
                k: int,
                users: Iterable or None,
                items: Iterable or None,
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


if __name__ == '__main__':
    spark = (SparkSession
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
    log = spark.createDataFrame(data=data,
                                schema=schema)

    br = BigRecommender(spark, date=datetime(2019, 10, 12))

    br.fit(log, user_features=None, item_features=None)
