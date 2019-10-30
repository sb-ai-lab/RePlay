"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
import os
from typing import Dict, Optional

from pyspark.ml.classification import (LogisticRegression,
                                       LogisticRegressionModel)
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import DataFrame
from pyspark.sql.functions import lit, when
from pyspark.sql.types import FloatType
from sponge_bob_magic.constants import DEFAULT_CONTEXT
from sponge_bob_magic.models.base_recommender import BaseRecommender
from sponge_bob_magic.utils import get_feature_cols, udf_get


class LinearRecommender(BaseRecommender):
    """ рекомендатель на основе линейной модели и эмбеддингов """
    _model: LogisticRegressionModel

    def get_params(self) -> Dict[str, object]:
        return dict()

    def _pre_fit(self, log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 path: Optional[str] = None) -> None:
        pass

    # TODO: добавить проверку, что в логе есть только нули и единицы
    def _fit_partial(
            self,
            log: DataFrame,
            user_features: DataFrame,
            item_features: DataFrame,
            path: Optional[str] = None
    ) -> None:
        data = (
            self._augment_data(log, user_features, item_features)
            .withColumnRenamed("relevance", "label")
            .select("label", "features")
        )
        self._model = LogisticRegression().fit(data)
        if path is not None:
            model_path = os.path.join(path, "linear.model")
            self._model.write().overwrite().save(model_path)
            self._model = self._model.read().load(model_path)

    @staticmethod
    def _augment_data(
            log: DataFrame,
            user_features: DataFrame,
            item_features: DataFrame
    ) -> DataFrame:
        """
        обогатить лог данными о свойствах пользователей и объектов

        :param log: лог в стандартном формате
        :param user_features: свойства пользователей в стандартном формате
        :param item_features: свойства объектов в стандартном формате
        :return: новый Spark DataFrame, в котором к каждой строчке лога
        добавлены свойства пользователя и объекта, которые в ней встречаются
        """
        user_feature_cols, item_feature_cols = get_feature_cols(
            user_features, item_features)
        return VectorAssembler(
            inputCols=user_feature_cols + item_feature_cols,
            outputCol="features"
        ).transform(
            log
            .join(user_features.drop("timestamp"), on="user_id", how="inner")
            .join(item_features.drop("timestamp"), on="item_id", how="inner")
        )

    def _predict(self,
                 k: int,
                 users: DataFrame,
                 items: DataFrame,
                 context: Optional[str],
                 log: DataFrame,
                 user_features: Optional[DataFrame],
                 item_features: Optional[DataFrame],
                 to_filter_seen_items: bool = True,
                 path: Optional[str] = None) -> DataFrame:
        data = (
            self._augment_data(
                users.crossJoin(items), user_features, item_features
            )
            .select("features", "item_id", "user_id")
        )
        if to_filter_seen_items:
            data = data.join(log, on=["user_id", "item_id"], how="left_anti")
        recs = (
            self._model
            .transform(data)
            .select(
                "user_id",
                "item_id",
                udf_get("probability", lit(1))
                .alias("relevance")
                .cast(FloatType())
            )
            .withColumn("context", lit(DEFAULT_CONTEXT))
        )
        recs = self._get_top_k_recs(recs, k)
        recs = recs.withColumn(
            "relevance",
            when(recs["relevance"] < 0, 0).otherwise(recs["relevance"])
        )
        if path is not None:
            path_parquet = os.path.join(path, "recs.parquet")
            recs.write.mode("overwrite").parquet(path_parquet)
            recs = self.spark.read.parquet(path_parquet)
        return recs
