"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Dict, Optional

from pyspark.ml.classification import (RandomForestClassificationModel,
                                       RandomForestClassifier)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, lit, udf, when
from pyspark.sql.types import DoubleType, FloatType

from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import func_get, get_feature_cols, get_top_k_recs


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

    def _pre_fit(self,
                 log: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None) -> None:
        self.augmented_data = (
            self._augment_data(log, user_features, item_features)
            .withColumnRenamed("relevance", "label")
            .select("label", "features")
        ).cache()

    def _fit(self,
             log: DataFrame,
             user_features: Optional[DataFrame] = None,
             item_features: Optional[DataFrame] = None) -> None:
        self.model = RandomForestClassifier(
            **self.model_params
        ).fit(
            self.augmented_data
        )

    @staticmethod
    def _augment_data(
            log: DataFrame,
            user_features: DataFrame,
            item_features: DataFrame
    ) -> DataFrame:
        """
        Обогащает лог фичами пользователей и объектов.

        :param log: лог в стандартном формате
        :param user_features: свойства пользователей в стандартном формате
        :param item_features: свойства объектов в стандартном формате
        :return: новый спарк-датайрейм, в котором к каждой строчке лога
            добавлены фичи пользователя и объекта, которые в ней встречаются
        """
        user_feature_cols, item_feature_cols = get_feature_cols(
            user_features, item_features
        )
        return VectorAssembler(
            inputCols=user_feature_cols + item_feature_cols,
            outputCol="features"
        ).transform(
            log
            .withColumnRenamed("user_id", "uid")
            .withColumnRenamed("item_id", "iid")
            .join(
                user_features.drop("timestamp"),
                on=col("user_id") == col("uid"),
                how="inner"
            )
            .join(
                item_features.drop("timestamp"),
                on=col("item_id") == col("iid"),
                how="inner"
            ).drop("iid", "uid")
        )

    def _predict(self,
                 log: DataFrame,
                 k: int,
                 users: DataFrame,
                 items: DataFrame,
                 user_features: Optional[DataFrame] = None,
                 item_features: Optional[DataFrame] = None,
                 filter_seen_items: bool = True) -> DataFrame:
        data = (
            self._augment_data(
                users.crossJoin(items), user_features, item_features
            )
            .select("features", "item_id", "user_id")
        )
        if filter_seen_items:
            data = data.join(log, on=["user_id", "item_id"], how="left_anti")
        recs = (
            self.model
            .transform(data)
            .select(
                "user_id",
                "item_id",
                udf(func_get, DoubleType())("probability", lit(1))
                .alias("relevance")
                .cast(FloatType())
            )
        )
        recs = get_top_k_recs(recs, k)
        recs = recs.withColumn(
            "relevance",
            when(recs["relevance"] < 0, 0).otherwise(recs["relevance"])
        )
        return recs
