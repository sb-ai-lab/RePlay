"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, TypeVar, Set

import joblib
import optuna
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as sf

from sponge_bob_magic import constants
from sponge_bob_magic.metrics.metrics import Metrics
from sponge_bob_magic.models.base_recommender import BaseRecommender
from sponge_bob_magic.models.knn_recommender import KNNRecommender
from sponge_bob_magic.models.popular_recomennder import PopularRecommender
from sponge_bob_magic.validation_schemes import ValidationSchemes

TNum = TypeVar('TNum', int, float)


class KNNScenario:
    """ Сценарий с item-based KNN моделью. """
    num_attempts: int = 10
    tested_params_set: Set[Tuple[TNum, TNum]] = set()
    seed: int = 1234
    model: Optional[BaseRecommender]
    study: Optional[optuna.Study]

    def __init__(self, spark: SparkSession):
        self.spark = spark

    def research(
            self,
            params_grid: Dict[str, Tuple[TNum, TNum]],
            log: DataFrame,
            users: Optional[DataFrame],
            items: Optional[DataFrame],
            user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None,
            test_start: Optional[datetime] = None,
            test_size: float = None,
            k: int = 10,
            context: Optional[str] = None,
            to_filter_seen_items: bool = True,
            n_trials: int = 10,
            n_jobs: int = 1,
            how_to_split: str = 'by_date',
            path: Optional[str] = None
    ) -> Dict[str, Any]:
        if context is None:
            context = constants.DEFAULT_CONTEXT

        splitter = ValidationSchemes(self.spark)

        logging.debug("Деление на трейн и тест")
        if how_to_split == 'by_date':
            train, test_input, test = splitter.log_split_by_date(
                log, test_start=test_start,
                drop_cold_users=True, drop_cold_items=True
            )
        elif how_to_split == 'randomly':
            train, test_input, test = splitter.log_split_randomly(
                log,
                drop_cold_users=True, drop_cold_items=True,
                seed=self.seed, test_size=test_size
            )
        else:
            raise ValueError(
                f"Значение how_to_split неверное ({how_to_split}), "
                "допустимые варианты: 'by_date' или 'randomly'")

        # рассчитываем все выборки перед подбором параметров
        train.cache()
        test_input.cache()
        test.cache()
        logging.debug(f"Длина трейна и теста: {train.count(), test.count()}")
        logging.debug("Количество юзеров в трейне и тесте: "
                      f"{train.select('user_id').distinct().count()}, "
                      f"{test.select('user_id').distinct().count()}")
        logging.debug("Количество айтемов в трейне и тесте: "
                      f"{train.select('item_id').distinct().count()}, "
                      f"{test.select('item_id').distinct().count()}")

        # если users или items нет, возьмем всех из теста,
        # чтобы не делать на каждый trial их заново
        if users is None:
            users = test.select('user_id').distinct().cache()

        if items is None:
            items = test.select('item_id').distinct().cache()

        logging.debug("Популярная модель: фит_предикт")
        popular_recs = (
            PopularRecommender(self.spark,
                               alpha=params_grid.get("alpha", 0),
                               beta=params_grid.get("beta", 0))
            .fit_predict(k, users, items, context, train,
                         user_features, item_features,
                         to_filter_seen_items)
            .select(sf.col("user_id"),
                    sf.col("item_id"),
                    sf.col("context").alias("context_pop"),
                    sf.col("relevance").alias("relevance_pop"), )
        )
        max_in_popular_recs = (
            popular_recs
            .agg({"relevance_pop": "max"})
            .collect()[0][0]
        )

        logging.debug("Модель KNN")
        self.model = KNNRecommender(self.spark)

        logging.debug("Первый пре-фит модели")
        self.model._pre_fit(log=train,
                            user_features=user_features,
                            item_features=item_features,
                            path=None)

        def objective(trial: optuna.Trial):
            if path is not None:
                joblib.dump(self.study,
                            os.path.join(path, "optuna_study.joblib"))

            # num_attempts раз пытаемся засемплить параметры, которых еще
            # не было; если не получилось, берем с последней попытки
            num_neighbours, shrink = 0, 0
            for attempt in range(KNNScenario.num_attempts):
                num_neighbours = trial.suggest_discrete_uniform(
                    'num_neighbours',
                    params_grid['num_neighbours'][0],
                    params_grid['num_neighbours'][1],
                    params_grid['num_neighbours'][2],
                )
                shrink = trial.suggest_categorical(
                    'shrink',
                    params_grid['shrink']
                )

                if (num_neighbours, shrink) not in self.tested_params_set:
                    break

            self.tested_params_set.add((num_neighbours, shrink))
            params = {'num_neighbours': num_neighbours,
                      'shrink': shrink}
            self.model.set_params(**params)
            logging.debug(f"-- Параметры: {params}")

            logging.debug("-- Второй фит модели в оптимизации")
            self.model._fit_partial(log=train,
                                    user_features=user_features,
                                    item_features=item_features,
                                    path=None)

            logging.debug("-- Предикт модели в оптимизации")
            recs = self.model.predict(
                k=k,
                users=users, items=items,
                user_features=user_features,
                item_features=item_features,
                context=context,
                log=train,
                to_filter_seen_items=to_filter_seen_items
            )

            # добавим максимум из популярных реков,
            # чтобы сохранить порядок при заборе топ-k
            recs = recs.withColumn(
                "relevance",
                sf.col("relevance") + 10 * max_in_popular_recs
            )
            logging.debug(f"-- Длина рекомендаций: {recs.count()}")

            logging.debug("-- Дополняем рекомендации популярными")
            recs = recs.join(popular_recs, on=['user_id', 'item_id'],
                             how='full_outer')

            recs = (recs
                    .withColumn('context',
                                sf.coalesce('context', 'context_pop'))
                    .withColumn('relevance',
                                sf.coalesce('relevance', 'relevance_pop'))
                    )
            recs = recs.select("user_id", "item_id", "context", "relevance")

            recs = self.model._get_top_k_recs(recs, k)
            logging.debug(f"-- Длина рекомендаций: {recs.count()}")

            logging.debug("-- Подсчет метрики в оптимизации")
            hit_rate = Metrics.hit_rate_at_k(recs, test, k=k)
            ndcg = Metrics.ndcg_at_k(recs, test, k=k)
            precision = Metrics.precision_at_k(recs, test, k=k)
            map_metric = Metrics.map_at_k(recs, test, k=k)

            trial.set_user_attr('nDCG@k', ndcg)
            trial.set_user_attr('precision@k', precision)
            trial.set_user_attr('MAP@k', map_metric)
            trial.set_user_attr('HitRate@k', hit_rate)

            logging.debug(f"-- Метрики: "
                          f"hit_rate={hit_rate:.4f}, "
                          f"ndcg={ndcg:.4f}, "
                          f"precision={precision:.4f}, "
                          f"map_metric={map_metric:.4f}"
                          )

            return hit_rate

        sampler = optuna.samplers.RandomSampler()
        self.study = optuna.create_study(direction='maximize', sampler=sampler)

        logging.debug("-------------")
        logging.debug("Начало оптимизации параметров")
        self.study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

        logging.debug(f"Лучшие значения метрики: {self.study.best_value}")
        logging.debug(f"Лучшие параметры: {self.study.best_params}")
        return self.study.best_params

    def production(
            self,
            params,
            log: DataFrame,
            users: Optional[DataFrame],
            items: Optional[DataFrame],
            user_features: Optional[DataFrame],
            item_features: Optional[DataFrame],
            k: int,
            context: Optional[str],
            to_filter_seen_items: bool
    ) -> DataFrame:
        self.model = KNNRecommender(self.spark)
        self.model.set_params(**params)

        return self.model.fit_predict(k, users, items, context, log,
                                      user_features, item_features,
                                      to_filter_seen_items)


if __name__ == '__main__':
    spark_ = (SparkSession
              .builder
              .master('local[4]')
              .config('spark.driver.memory', '2g')
              .config("spark.sql.shuffle.partitions", "1")
              .appName('testing-pyspark')
              .enableHiveSupport()
              .getOrCreate())

    spark_.sparkContext.setCheckpointDir(os.environ['SPONGE_BOB_CHECKPOINTS'])

    data = [
        ["user1", "item4", 1.0, 'no_context', datetime(2019, 10, 8)],
        ["user2", "item2", 2.0, 'no_context', datetime(2019, 10, 9)],
        ["user1", "item3", 1.0, 'no_context', datetime(2019, 10, 10)],
        ["user1", "item1", 1.0, 'no_context', datetime(2019, 10, 11)],
        ["user1", "item4", 1.0, 'no_context', datetime(2019, 10, 12)],
        ["user1", "item1", 1.0, 'no_context', datetime(2019, 10, 13)],
        ["user1", "item1", 1.0, 'no_context', datetime(2019, 10, 14)],
        ["user1", "item4", 1.0, 'no_context', datetime(2019, 10, 15)],
        ["user1", "item2", 1.0, 'no_context', datetime(2019, 10, 16)],
        ["user1", "item1", 1.0, 'no_context', datetime(2019, 10, 17)],
        ["user3", "item3", 2.0, 'no_context', datetime(2019, 10, 18)],
        ["user3", "item2", 2.0, 'no_context', datetime(2019, 10, 19)],
        ["user3", "item3", 2.0, 'no_context', datetime(2019, 10, 20)],
    ]
    schema = ['user_id', 'item_id', 'relevance', 'context', 'timestamp']
    log_ = spark_.createDataFrame(data=data,
                                  schema=schema)

    knn_scenario = KNNScenario(spark_)
    knn_params_grid = {'num_neighbours': (1, 9, 2), 'shrink': (0, 10)}

    best_params = knn_scenario.research(
        knn_params_grid,
        log_,
        users=None, items=None,
        user_features=None,
        item_features=None,
        test_start=datetime(2019, 10, 11),
        k=2, context=None,
        to_filter_seen_items=True,
        n_trials=4, n_jobs=4
    )

    print(best_params)

    recs_ = knn_scenario.production(
        best_params,
        log_,
        users=None,
        items=None,
        user_features=None,
        item_features=None,
        k=2,
        context='no_context',
        to_filter_seen_items=True
    )

    recs_.show()

    best_params = knn_scenario.research(
        knn_params_grid,
        log_,
        users=None, items=None,
        user_features=None,
        item_features=None,
        test_start=None,
        test_size=0.4,
        k=2, context=None,
        to_filter_seen_items=True,
        n_trials=4, n_jobs=1,
        how_to_split='randomly'
    )

    print(best_params)

    recs_ = knn_scenario.production(
        best_params,
        log_,
        users=None,
        items=None,
        user_features=None,
        item_features=None,
        k=2,
        context=None,
        to_filter_seen_items=True
    )

    recs_.show()
