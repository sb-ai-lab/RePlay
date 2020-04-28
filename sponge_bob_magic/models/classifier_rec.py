"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional

from pyspark.ml.classification import (
    RandomForestClassificationModel,
    RandomForestClassifier,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, udf
from pyspark.sql.types import DoubleType, FloatType

from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import func_get, vector_dot, vector_mult


class ClassifierRec(Recommender):
    """
    Рекомендатель на основе классификатора.

    Получает на вход лог, в котором ``relevance`` принимает значения ``0`` и ``1``.
    Обучение строится следующим образом:

    * к логу присоединяются свойства пользователей и объектов (если есть)
    * свойства считаются фичами классификатора, а ``relevance`` --- таргетом
    * обучается случайный лес, который умеет предсказывать ``relevance``

    В выдачу рекомендаций попадает top K объектов с наивысшим предсказанным скором от классификатора.
    """

    model: RandomForestClassificationModel
    augmented_data: DataFrame

    def __init__(self, **kwargs):
        self.model_params = kwargs

    def get_params(self) -> Dict[str, object]:
        return self.model_params

    def _pre_fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        relevances = {row[0] for row in log.select("relevance").distinct().collect()}
        if relevances != {0, 1}:
            raise ValueError(
                "в логе должны быть relevance только 0 или 1"
                " и присутствовать значения обоих классов"
            )
        self.augmented_data = (
            self._augment_data(log, user_features, item_features)
            .withColumnRenamed("relevance", "label")
            .select("label", "features")
        ).cache()

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        self.model = RandomForestClassifier(**self.model_params).fit(
            self.augmented_data
        )

    @staticmethod
    def _augment_data(
        log: DataFrame, user_features: DataFrame, item_features: DataFrame
    ) -> DataFrame:
        """
        Обогащает лог фичами пользователей и объектов.

        :param log: лог в стандартном формате
        :param user_features: свойства пользователей в стандартном формате
        :param item_features: свойства объектов в стандартном формате
        :return: новый спарк-датайрейм, в котором к каждой строчке лога
            добавлены фичи пользователя и объекта, которые в ней встречаются
        """
        return (
            VectorAssembler(
                inputCols=["user_features", "item_features", "mult", "dot_product"],
                outputCol="features",
            )
            .transform(
                log.withColumnRenamed("user_id", "uid")
                .withColumnRenamed("item_id", "iid")
                .join(
                    user_features.select("user_id", "user_features"),
                    on=col("user_id") == col("uid"),
                    how="inner",
                )
                .join(
                    item_features.select("item_id", "item_features"),
                    on=col("item_id") == col("iid"),
                    how="inner",
                )
                .drop("iid", "uid")
                .withColumn("mult", vector_mult("user_features", "item_features"))
                .withColumn("dot_product", vector_dot("user_features", "item_features"))
            )
            .drop("mult", "dot_product")
        )

    def _predict(
        self,
        log: DataFrame,
        k: int,
        users: DataFrame,
        items: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
        filter_seen_items: bool = True,
    ) -> DataFrame:
        data = self._augment_data(
            users.crossJoin(items), user_features, item_features
        ).select("features", "item_id", "user_id")
        recs = self.model.transform(data).select(
            "user_id",
            "item_id",
            udf(func_get, DoubleType())("probability", lit(1))
            .alias("relevance")
            .cast(FloatType()),
        )
        return recs
