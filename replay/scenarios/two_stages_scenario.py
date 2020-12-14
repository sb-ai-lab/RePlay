"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional, Tuple

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnull, lit, when

from replay.constants import IntOrList
from replay.experiment import Experiment
from replay.metrics import HitRate, Metric
from replay.models.als import ALSWrap
from replay.models.base_rec import Recommender
from replay.models.classifier_rec import ClassifierRec
from replay.session_handler import State
from replay.splitters import Splitter, UserSplitter
from replay.utils import get_log_info, horizontal_explode, get_stats

DEFAULT_SECOND_STAGE_SPLITTER = UserSplitter(
    drop_cold_items=False, item_test_size=1, shuffle=True, seed=42
)
DEFAULT_FIRST_STAGE_SPLITTER = UserSplitter(
    drop_cold_items=False, item_test_size=0.5, shuffle=True, seed=42
)


# pylint: disable=too-many-instance-attributes
class TwoStagesScenario:
    """
    Двухуровневый сценарий:

    * получить входной ``log``
    * с помощью ``second_stage_splitter`` разбить ``log`` на ``second_stage_train`` и ``second_stage_test``
    * с помощью ``first_stage_splitter`` разбить ``second_stage_train`` на ``first_stage_train`` и ``first_stage_test``
    * на ``first_stage_train`` обучить ``first_stage_model`` (которая умеет генерировать эмбеддинги пользователей и объектов)
    * с помощью ``first_stage_model`` получить по ``first_stage_k``
        рекомендованных объектов для тестовых пользователей (``first_stage_k > second_stage_k``)
    * сравнивая ``first_stage_recs`` с ``first_stage_test`` получить таргет
        для обучения классификатора (угадали --- ``1``, не угадали --- ``0``)
    * обучить ``second_stage_model`` (на основе классификатора) на таргете из предыдущего пункта и с эмбеддингами пользователей и объектов в качестве фичей
    * получить ``second_stage_k`` рекомендаций с помощью ``second_stage_model``
    * посчитать метрику от ``second_stage_recs`` и ``second_stage_test``

    """

    _experiment: Optional[Experiment] = None

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        second_stage_splitter: Splitter = DEFAULT_SECOND_STAGE_SPLITTER,
        first_stage_splitter: Splitter = DEFAULT_FIRST_STAGE_SPLITTER,
        first_model: Recommender = ALSWrap(rank=100),
        second_model: Optional[ClassifierRec] = None,
        first_stage_k: int = 100,
        metrics: Optional[Dict[Metric, IntOrList]] = None,
        stat_features: bool = True,
    ) -> None:
        """
        собрать двухуровневую рекомендательную архитектуру из блоков

        :param second_stage_splitter: как разбивать входной лог на ``train`` и ``test``.
                                      По умолчанию у каждого пользователя откладывается 1 объект в ``train``, а остальное идёт в ``test``.
                                      ``test`` будет использоваться для финальной оценки качества модели, а ``train`` будет разбиваться повторно.
        :param first_stage_splitter: как разбивать ``train`` на новые ``first_stage_train`` и ``first_stage_test``.
                                     По умолчанию у каждого пользователя откладывается 40% объектов в ``first_stage_train``,
                                     а остальное идёт в ``first_stage_test``.
        :param first_model: модель какого класса будем обучать на ``first_stage_train``. По умолчанию :ref:`ALS<als-rec>`.
        :param first_stage_k: сколько объектов будем рекомендовать моделью первого уровня (``first_model``). По умолчанию 100
        :param second_model: какую модель будем обучать на результате сравнения предсказаний ``first_model`` и ``first_stage_test``
        :param metrics: какие метрики будем оценивать у ``second_model`` на ``test``. По умолчанию :ref:`HitRate@10<hit-rate>`.
        :param stat_features: выполнить подсчет статистических признаков по логу взаимодействия для пользователей
                                               и объектов для обучения модели второго уровня
        """

        self.second_stage_splitter = second_stage_splitter
        self.first_stage_splitter = first_stage_splitter
        self.first_model = first_model
        self.first_stage_k = first_stage_k
        if second_model is None:
            self.second_model = ClassifierRec()
        else:
            self.second_model = second_model
        self.metrics = {HitRate(): [10]} if metrics is None else metrics
        self.stat_features = stat_features

    @property
    def experiment(self) -> Experiment:
        """ история экспериментов """
        if self._experiment is None:
            raise ValueError(
                "нужно запустить метод get_recs, чтобы провести эксперимент"
            )
        return self._experiment

    def _split_data(
        self, log: DataFrame
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        mixed_train, test = self.second_stage_splitter.split(log)
        mixed_train.cache()
        State().logger.debug("mixed_train stat: %s", get_log_info(mixed_train))
        State().logger.debug("test stat: %s", get_log_info(test))
        first_train, first_test = self.first_stage_splitter.split(mixed_train)
        State().logger.debug("first_train stat: %s", get_log_info(first_train))
        State().logger.debug("first_test stat: %s", get_log_info(first_test))
        return first_train.cache(), first_test.cache(), test.cache()

    def _get_first_stage_recs(
        self, first_train: DataFrame, first_test: DataFrame
    ) -> DataFrame:
        return self.first_model.fit_predict(
            log=first_train,
            k=self.first_stage_k,
            users=first_test.select("user_id").distinct().cache(),
            items=first_train.select("item_id").distinct().cache(),
        ).cache()

    @staticmethod
    def _join_features(
        first_df: DataFrame,
        other_df: Optional[DataFrame] = None,
        on_col="user_id",
        how="inner",
    ) -> DataFrame:
        if other_df is None:
            return first_df
        return first_df.join(other_df, how=how, on=on_col).cache()

    def _second_stage_data(
        self,
        first_recs: DataFrame,
        first_train: DataFrame,
        first_test: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        user_features_factors = (
            self.first_model.inv_user_indexer.transform(
                horizontal_explode(
                    self.first_model.model.userFactors,
                    "features",
                    "user_feature",
                    [col("id").alias("user_idx")],
                )
            )
            .drop("user_idx")
            .cache()
        )
        if self.stat_features:
            user_statistics = get_stats(first_train)
            user_features = self._join_features(user_statistics, user_features)

            item_statistics = get_stats(first_train, group_by="item_id")
            item_features = self._join_features(
                item_statistics, item_features, on_col="item_id"
            )

        user_features = self._join_features(
            user_features_factors, user_features
        )

        item_features_factors = (
            self.first_model.inv_item_indexer.transform(
                horizontal_explode(
                    self.first_model.model.itemFactors,
                    "features",
                    "item_feature",
                    [col("id").alias("item_idx")],
                )
            )
            .drop("item_idx")
            .cache()
        )
        item_features = self._join_features(
            item_features_factors, item_features, "item_id"
        )

        second_train = (
            first_recs.withColumnRenamed("relevance", "recs")
            .join(
                first_test.select("user_id", "item_id", "relevance").toDF(
                    "uid", "iid", "relevance"
                ),
                how="left",
                on=[
                    col("user_id") == col("uid"),
                    col("item_id") == col("iid"),
                ],
            )
            .withColumn(
                "relevance",
                when(isnull("relevance"), lit(0)).otherwise(lit(1)),
            )
            .drop("uid", "iid")
        ).cache()
        State().logger.debug(
            "баланс классов: положительных %d из %d",
            second_train.filter("relevance = 1").count(),
            second_train.count(),
        )
        return user_features, item_features, second_train

    def get_recs(
        self,
        log: DataFrame,
        k: int,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        обучить двухуровневую модель и выдать рекомендации на тестовом множестве,
        полученном в соответствии с выбранной схемой валидации

        >>> from replay.session_handler import get_spark_session, State
        >>> spark = get_spark_session(1, 1)
        >>> state = State(spark)

        >>> import numpy as np
        >>> np.random.seed(47)
        >>> from logging import ERROR
        >>> State().logger.setLevel(ERROR)
        >>> from replay.splitters import UserSplitter
        >>> splitter = UserSplitter(
        ...    item_test_size=1,
        ...    shuffle=True,
        ...    drop_cold_items=False,
        ...    seed=147
        ... )
        >>> from replay.metrics import HitRate
        >>> from pyspark.ml.classification import RandomForestClassifier
        >>> two_stages = TwoStagesScenario(
        ...     first_stage_k=10,
        ...     first_stage_splitter=splitter,
        ...     second_stage_splitter=splitter,
        ...     metrics={HitRate(): 1},
        ...     second_model=ClassifierRec(RandomForestClassifier(seed=47)),
        ...     stat_features=False
        ... )
        >>> two_stages.experiment
        Traceback (most recent call last):
            ...
        ValueError: нужно запустить метод get_recs, чтобы провести эксперимент
        >>> log = spark.createDataFrame(
        ...     [(i, i + j, 1) for i in range(10) for j in range(10)]
        ... ).toDF("user_id", "item_id", "relevance")
        >>> two_stages.get_recs(log, 1).show()
        +-------+-------+----------+
        |user_id|item_id| relevance|
        +-------+-------+----------+
        |      0|     10|       0.1|
        |      1|     10|     0.215|
        |      2|      5|0.06317336|
        |      3|     13|0.07817336|
        |      4|     12|0.11317336|
        |      5|      3|0.11263158|
        |      6|     10|       0.3|
        |      7|      8|0.15413581|
        |      8|      6|0.21785714|
        |      9|     14| 0.2115067|
        +-------+-------+----------+
        <BLANKLINE>
        >>> two_stages.experiment.results
                             HitRate@1
        two_stages_scenario        0.6

        :param log: лог пользовательских предпочтений
        :param k: количество рекомендаций, которые нужно вернуть каждому пользователю
        :param user_features: признаки пользователей,
            спарк-датафрейм с колонками
            ``[user_id , timestamp]`` и колонки с признаками
        :param item_features: признаки объектов,
            спарк-датафрейм с колонками
            ``[item_id , timestamp]`` и колонки с признаками
        :return: DataFrame со списком рекомендаций
        """

        first_train, first_test, test = self._split_data(log)
        full_train = first_train.union(first_test).cache()

        first_recs = self._get_first_stage_recs(first_train, first_test)
        user_features, item_features, second_train = self._second_stage_data(
            first_recs, first_train, first_test, user_features, item_features
        )
        first_recs_for_test = self.first_model.predict(
            log=full_train,
            k=self.first_stage_k,
            users=test.select("user_id").distinct(),
            items=first_train.select("item_id").distinct(),
        ).cache()
        # pylint: disable=protected-access
        self.second_model._fit_wrap(
            log=second_train,
            user_features=user_features,
            item_features=item_features,
        )

        second_recs = self.second_model.rerank(  # type: ignore
            log=first_recs_for_test.withColumnRenamed("relevance", "recs"),
            k=k,
            user_features=user_features,
            item_features=item_features,
            users=test.select("user_id").distinct().cache(),
        ).cache()
        State().logger.debug(
            "ROC AUC модели второго уровня (как классификатора): %.4f",
            BinaryClassificationEvaluator().evaluate(
                self.second_model.model.transform(
                    self.second_model.augmented_data
                )
            ),
        )
        self._experiment = Experiment(test, self.metrics)  # type: ignore
        self._experiment.add_result("two_stages_scenario", second_recs)
        return second_recs
