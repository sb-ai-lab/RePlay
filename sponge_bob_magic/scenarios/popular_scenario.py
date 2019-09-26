import logging
from datetime import datetime
from typing import Dict, Tuple, TypeVar, Any, Iterable

import joblib
import optuna
from pyspark.sql import SparkSession, DataFrame

from metrics.metrics import Metrics
from models.popular_recomennder import PopularRecommender
from validation_schemes import ValidationSchemes

TNum = TypeVar('TNum', int, float)


class PopularScenario:
    def __init__(self, spark: SparkSession):
        self.model = None
        self.spark = spark
        self.optuna_study = None
        self.model = None
        self.path_optuna_study = None

    @staticmethod
    def get_values_in_column(df, column):
        return (df
                .select(column)
                .rdd
                .flatMap(lambda x: x)
                .collect())

    def research(self,
                 params_grid: Dict[str, Tuple[TNum, TNum]],
                 log: DataFrame,
                 users: Iterable or DataFrame or None,
                 items: Iterable or DataFrame or None,
                 user_features: DataFrame or None,
                 item_features: DataFrame or None,
                 test_start: datetime,
                 k: int,
                 context: str or None,
                 n_trials: int,
                 n_jobs: int,
                 ) -> Dict[str, Any]:
        splitter = ValidationSchemes(self.spark)
        train, test_input, test = splitter.log_split_by_date(
            log, test_start=test_start,
            drop_cold_users=False, drop_cold_items=True)

        self.model = PopularRecommender(self.spark)

        if users is None:
            users = self.get_values_in_column(log.select('user_id').distinct(),
                                              'user_id')
        else:
            users = self.spark.createDataFrame(data=[[user] for user in users],
                                               schema=['user_id'])
        if items is None:
            items = self.get_values_in_column(log.select('item_id').distinct(),
                                              'item_id')
        else:
            items = set(items)

        def objective(trial: optuna.Trial):
            if self.path_optuna_study is not None:
                joblib.dump(self.optuna_study, self.path_optuna_study)

            alpha = trial.suggest_int(
                'alpha', params_grid['alpha'][0], params_grid['alpha'][1])
            beta = trial.suggest_int(
                'beta', params_grid['beta'][0], params_grid['beta'][1])

            params = {'alpha': alpha,
                      'beta': beta}
            self.model.set_params(**params)

            self.model.fit(log=train,
                           user_features=user_features,
                           item_features=item_features)

            recs = self.model.predict(k=k,
                                      users=users, items=items,
                                      user_features=user_features,
                                      item_features=item_features,
                                      context=context,
                                      log=test,
                                      to_filter_seen_items=True)

            return Metrics.hit_rate_at_k(recs, test, k=k)

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        logging.debug(f"Best value of metric: {study.best_value}")
        logging.debug(f"Best values of parameters: {study.best_params}")
        return study.best_params


if __name__ == '__main__':
    spark_ = (SparkSession
              .builder
              .master('local[1]')
              .config('spark.driver.memory', '2g')
              .appName('testing-pyspark')
              .enableHiveSupport()
              .getOrCreate())

    data = [
        ["user1", "item1", 1.0, 'no_context', datetime(2019, 10, 8)],
        ["user2", "item2", 2.0, 'no_context', datetime(2019, 10, 9)],
        ["user1", "item3", 1.0, 'no_context', datetime(2019, 10, 10)],
        ["user1", "item1", 1.0, 'no_context', datetime(2019, 10, 11)],
        ["user1", "item2", 1.0, 'no_context', datetime(2019, 10, 12)],
        ["user3", "item3", 2.0, 'no_context', datetime(2019, 10, 13)],
    ]
    schema = ['user_id', 'item_id', 'relevance', 'context', 'timestamp']
    log_ = spark_.createDataFrame(data=data,
                                  schema=schema)

    br = PopularScenario(spark_)
    popular_params_grid = {'alpha': (0, 100), 'beta': (0, 100)}

    best_params = br.research(
        popular_params_grid,
        log_,
        users=None, items=None,
        user_features=None,
        item_features=None,
        test_start=datetime(2019, 10, 11),
        k=3, context='no_context',
        n_trials=4, n_jobs=4
    )

    print(best_params)
