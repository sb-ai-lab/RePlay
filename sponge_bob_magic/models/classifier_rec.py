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
from pyspark.sql.types import FloatType

from sponge_bob_magic.models.base_rec import Recommender
from sponge_bob_magic.utils import func_get


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

    def __init__(self, use_recs_value: Optional[bool] = False, **kwargs):
        self.model_params: Dict[str, object] = kwargs
        self.use_recs_value = use_recs_value

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:
        relevances = {
            row[0] for row in log.select("relevance").distinct().collect()
        }
        if relevances != {0, 1}:
            raise ValueError(
                "в логе должны быть relevance только 0 или 1"
                " и присутствовать значения обоих классов"
            )
        self.augmented_data = (
            self._augment_data(log, user_features, item_features)
            .withColumnRenamed("relevance", "label")
            .select("label", "features", "user_idx", "item_idx")
        ).cache()
        self.model = RandomForestClassifier(**self.model_params).fit(
            self.augmented_data
        )

    def _augment_data(
        self,
        log: DataFrame,
        user_features: DataFrame,
        item_features: DataFrame,
    ) -> DataFrame:
        """
        Обогащает лог фичами пользователей и объектов.

        :param log: лог в стандартном формате
        :param user_features: свойства пользователей в стандартном формате
        :param item_features: свойства объектов в стандартном формате
        :return: новый спарк-датайрейм, в котором к каждой строчке лога
            добавлены фичи пользователя и объекта, которые в ней встречаются
        """
        user_vectors = (
            VectorAssembler(
                inputCols=user_features.drop("user_idx").columns,
                outputCol="user_features",
            )
            .transform(user_features)
            .cache()
        )
        item_vectors = (
            VectorAssembler(
                inputCols=item_features.drop("item_idx").columns,
                outputCol="item_features",
            )
            .transform(item_features)
            .cache()
        )
        return VectorAssembler(
            inputCols=["user_features", "item_features"]
            + (["recs"] if self.use_recs_value else []),
            outputCol="features",
        ).transform(
            log.withColumnRenamed("user_idx", "uid")
            .withColumnRenamed("item_idx", "iid")
            .join(
                user_vectors.select("user_idx", "user_features"),
                on=col("user_idx") == col("uid"),
                how="inner",
            )
            .join(
                item_vectors.select("item_idx", "item_features"),
                on=col("item_idx") == col("iid"),
                how="inner",
            )
            .drop("iid", "uid")
        )

    # pylint: disable=too-many-arguments
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
            log.join(
                users.withColumnRenamed("user_idx", "user"),
                how="inner",
                on=col("user_idx") == col("user"),
            ).select(
                *(
                    ["item_idx", "user_idx"]
                    + (["recs"] if self.use_recs_value else [])
                )
            ),
            user_features,
            item_features,
        ).select("features", "item_idx", "user_idx")
        recs = self.model.transform(data).select(
            "user_idx",
            "item_idx",
            udf(func_get, FloatType())("probability", lit(1)).alias(
                "relevance"
            ),
        )
        return recs
