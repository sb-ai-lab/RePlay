"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional, Tuple

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, isnull, lit, when

from sponge_bob_magic.constants import IntOrList
from sponge_bob_magic.experiment import Experiment
from sponge_bob_magic.metrics import HitRate, Metric
from sponge_bob_magic.models import ALSWrap, ClassifierRec, Recommender
from sponge_bob_magic.session_handler import State
from sponge_bob_magic.splitters import Splitter, UserSplitter
from sponge_bob_magic.utils import get_log_info, to_vector

DEFAULT_SECOND_STAGE_SPLITTER = UserSplitter(
    drop_cold_items=False,
    item_test_size=1,
    shuffle=True
)
DEFAULT_FIRST_STAGE_SPLITTER = UserSplitter(
    drop_cold_items=False,
    item_test_size=0.4,
    shuffle=True
)


class TwoStagesScenario:
    """
    Двухуровневый сценарий:

    * получить входной ``log``
    * с помощью ``second_stage_splitter`` разбить ``log`` на ``second_stage_train`` и ``second_stage_test``
    * с помощью ``first_stage_splitter`` разбить ``second_stage_train`` на ``first_stage_train`` и ``first_stage_test``
    * на ``first_stage_train`` обучить ``first_stage_model`` (которая умеет генерировать эмбеддинги пользователей и объектов)
    * с помощью ``frist_stage_model`` получить по ``first_stage_k`` рекомендованных объектов для тестовых пользователей (``first_stage_k > second_stage_k``)
    * сравнивая ``frist_stage_recs`` с ``first_stage_test`` получить таргет для обучения классификатора (угадали --- ``1``, не угадали --- ``0``)
    * обучить ``second_stage_model`` (на основе классификатора) на таргете из предыдущего пункта и с эмбеддингами пользователей и объектов в качестве фичей
    * получить ``second_stage_k`` рекомендаций с помощью ``second_stage_model``
    * посчитать метрику от ``second_stage_recs`` и ``second_stage_test``

    """
    _experiment: Optional[Experiment] = None

    def __init__(
            self,
            second_stage_splitter: Splitter = DEFAULT_SECOND_STAGE_SPLITTER,
            first_stage_splitter: Splitter = DEFAULT_FIRST_STAGE_SPLITTER,
            first_model: Recommender = ALSWrap(rank=100),
            second_model: ClassifierRec = ClassifierRec(),
            first_stage_k: int = 100,
            metrics: Dict[Metric, IntOrList] = {HitRate(): 10}
    ):
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

        """
        self.second_stage_splitter = second_stage_splitter
        self.first_stage_splitter = first_stage_splitter
        self.first_model = first_model
        self.first_stage_k = first_stage_k
        self.second_model = second_model
        self.metrics = metrics

    @property
    def experiment(self) -> Experiment:
        """ история экспериментов """
        if self._experiment is None:
            raise ValueError(
                "нужно запустить метод get_recs, чтобы провести эксперимент"
            )
        return self._experiment

    def _split_data(
            self,
            log: DataFrame
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        mixed_train, test = self.second_stage_splitter.split(log)
        mixed_train.cache()
        State().logger.debug("mixed_train stat: %s", get_log_info(mixed_train))
        State().logger.debug("test stat: %s", get_log_info(test))
        first_train, first_test = self.first_stage_splitter.split(mixed_train)
        State().logger.debug("first_train stat: %s", get_log_info(first_train))
        State().logger.debug("first_test stat: %s", get_log_info(first_test))
        return first_train.cache(), first_test.cache(), test.cache()

    def _get_first_stage_recs(self, first_train: DataFrame) -> DataFrame:
        return self.first_model.fit_predict(
            log=first_train,
            k=self.first_stage_k,
            users=first_train.select("user_id").distinct().cache(),
            items=first_train.select("item_id").distinct().cache()
        ).cache()

    def _second_stage_data(
            self,
            first_recs: DataFrame,
            first_test: DataFrame
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        user_features = self.first_model.inv_user_indexer.transform(
            self.first_model.model.userFactors.select(
                col("id").alias("user_idx"),
                to_vector("features").alias("user_features")
            )
        ).drop("user_idx").cache()
        item_features = self.first_model.inv_item_indexer.transform(
            self.first_model.model.itemFactors.select(
                col("id").alias("item_idx"),
                to_vector("features").alias("item_features")
            )
        ).drop("item_idx").cache()
        second_train = self.first_model.item_indexer.transform(
            self.first_model.user_indexer.transform(
                first_recs.drop("relevance").join(
                    first_test.select("user_id", "item_id", "relevance")
                    .toDF("uid", "iid", "relevance"),
                    how="left",
                    on=[
                        col("user_id") == col("uid"),
                        col("item_id") == col("iid")
                    ]
                ).withColumn(
                    "relevance",
                    when(isnull("relevance"), lit(0)).otherwise(lit(1))
                ).drop("uid", "iid").cache()
            )
        ).cache()
        State().logger.debug(
            "баланс классов: положительных %d из %d",
            second_train.filter("relevance = 1").count(),
            second_train.count()
        )
        return user_features, item_features, second_train

    def get_recs(self, log: DataFrame, k: int) -> DataFrame:
        """
        обучить двухуровневую модель и выдать рекомендации на тестовом множестве,
        полученном в соответствии с выбранной схемой валидации

        >>> spark = State().session
        >>> import numpy as np
        >>> np.random.seed(47)
        >>> from logging import ERROR
        >>> State().logger.setLevel(ERROR)
        >>> from sponge_bob_magic.splitters import UserSplitter
        >>> splitter = UserSplitter(
        ...    item_test_size=1,
        ...    shuffle=True,
        ...    drop_cold_items=False,
        ...    seed=147
        ... )
        >>> from sponge_bob_magic.metrics import HitRate
        >>> from sponge_bob_magic.models import ClassifierRec
        >>> two_stages = TwoStagesScenario(
        ...     first_stage_k=10,
        ...     first_stage_splitter=splitter,
        ...     second_stage_splitter=splitter,
        ...     metrics={HitRate(): 1},
        ...     second_model=ClassifierRec(seed=47)
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
        |      0|     17|       0.0|
        |      1|     16|       0.1|
        |      2|      9|0.14434524|
        |      3|     15|      0.15|
        |      4|     16|       0.1|
        |      5|      9|0.43434525|
        |      6|      3|       0.1|
        |      7|     16|      0.15|
        |      8|     15|      0.05|
        |      9|     17|       0.0|
        +-------+-------+----------+
        <BLANKLINE>
        >>> two_stages.experiment.pandas_df
                            HitRate@1
        two_stages_scenario        0.0

        :param log: лог пользовательских предпочтений
        :param k: количество рекомендаций, которые нужно вернуть каждому пользователю
        """
        first_train, first_test, test = self._split_data(log)
        first_recs = self._get_first_stage_recs(first_train)
        user_features, item_features, second_train = self._second_stage_data(
            first_recs, first_test
        )
        second_recs = self.second_model.fit_predict(
            log=second_train,
            k=k,
            users=test.select("user_id").distinct().cache(),
            items=test.select("item_id").distinct().cache(),
            user_features=user_features,
            item_features=item_features
        ).cache()
        State().logger.debug(
            "ROC AUC модели второго уровня (как классификатора): %.4f",
            BinaryClassificationEvaluator().evaluate(
                self.second_model.model.transform(
                    self.second_model.augmented_data
                )
            )
        )
        self._experiment = Experiment(test, self.metrics)
        self._experiment.add_result("two_stages_scenario", second_recs)
        return second_recs
