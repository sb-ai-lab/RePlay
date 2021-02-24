# pylint: disable=invalid-name, too-many-arguments, attribute-defined-outside-init
from typing import Optional, Dict, Any

from pandas import DataFrame
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as sf

from replay.constants import AnyDataFrame
from replay.models.base_rec import BaseRecommender
from replay.session_handler import State


class ColdUser(BaseRecommender):
    """
    Модель для рекомендаций холодным пользователям популярных объектов из их кластера.
    Кластеры выделяются из пользователей с историей по их признакам с помощью k means.
    """

    can_predict_cold_users = True
    _search_space = {
        "n": {"type": "int", "args": [2, 20]},
    }

    def __init__(self, n: int = 10):
        """
        :param n: количество кластеров
        """
        State()  # инициализируем сессию, если не создана
        self.kmeans = KMeans().setK(n).setFeaturesCol("features")

    def set_params(self, **params: Dict[str, Any]) -> None:
        """
        Устанавливает параметры модели.

        :param params: словарь, ключ - название параметра,
            значение - значение параметра
        :return:
        """
        if len(params) != 1:
            raise ValueError("Wrong number of params passed")
        if "n" not in params:
            raise ValueError("Wrong parameter name")
        self.kmeans = self.kmeans.setK(params["n"])

    def fit(self, log: AnyDataFrame, user_features: AnyDataFrame) -> None:
        """
        Выделить кластеры и посчитать популярность объектов в них.

        :param log: логи пользователей с историей для подсчета популярности объектов
        :param user_features: датафрейм связывающий `user_id` пользователей и их числовые признаки
        """
        self._fit_wrap(log=log, user_features=user_features)

    def _fit(
        self,
        log: DataFrame,
        user_features: Optional[DataFrame] = None,
        item_features: Optional[DataFrame] = None,
    ) -> None:

        df = self._transform_features(user_features)
        self.model = self.kmeans.fit(df)
        df = (
            self.model.transform(df)
            .select("user_idx", "prediction")
            .withColumnRenamed("prediction", "cluster")
        )

        log = log.join(df, on="user_idx", how="left")
        log = log.groupBy(["cluster", "item_idx"]).agg(
            sf.count("item_idx").alias("count")
        )
        m = log.groupby("cluster").agg(sf.max("count").alias("max"))
        log = log.join(m, on="cluster", how="left")
        log = log.withColumn(
            "relevance", sf.col("count") / sf.col("max")
        ).drop("count", "max")
        self.recs = log

    @staticmethod
    def _transform_features(df):
        feature_columns = df.drop("user_idx").columns
        vec = VectorAssembler(inputCols=feature_columns, outputCol="features")
        return vec.transform(df).select("user_idx", "features")

    def predict(
        self,
        user_features: DataFrame,
        k: int,
        log: Optional[DataFrame] = None,
    ) -> DataFrame:
        """
        Получить предсказания для переданных пользователей

        :param user_features: айди пользователей с числовыми фичами
        :param k: длина рекомендаций
        :param log: опциональный датафрейм с логами пользователей.
            Если передан, объекты отсюда удаляются из рекомендаций для соответствующих пользователей.
        :return: датафрейм с рекомендациями
        """
        filter_seen = bool(log)
        return self._predict_wrap(
            log=log,
            user_features=user_features,
            k=k,
            filter_seen_items=filter_seen,
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

        df = self._transform_features(user_features)
        df = (
            self.model.transform(df)
            .select("user_idx", "prediction")
            .withColumnRenamed("prediction", "cluster")
        )
        pred = df.join(self.recs, on="cluster")
        return pred
