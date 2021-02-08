# pylint: disable=invalid-name, too-many-arguments, attribute-defined-outside-init
from typing import Optional, List

from optuna import create_study
from optuna.samplers import TPESampler
from pandas import DataFrame
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as sf

from replay.constants import AnyDataFrame
from replay.metrics import Metric, NDCG
from replay.optuna_objective import (
    ColdObjective,
    ColdSplitData,
)
from replay.utils import convert2spark, get_top_k_recs


class ColdUser:
    """
    Модель для рекомендаций холодным пользователям популярных объектов из их кластера.
    Кластеры выделяются из пользователей с историей по их признакам с помощью k means.
    """

    def fit(
        self, log: AnyDataFrame, user_features: AnyDataFrame, k: int = 10,
    ) -> None:
        """
        Выделить кластеры и посчитать популярность объектов в них.

        :param log: логи пользователей с историей для подсчета популярности объектов
        :param user_features: датафрейм связывающий `user_id` пользователей и их числовые признаки
        :param k: количество кластеров
        """
        log = convert2spark(log)
        user_features = convert2spark(user_features)
        df = self._transform_features(user_features)
        kmeans = KMeans().setK(k).setFeaturesCol("features")
        self.model = kmeans.fit(df)
        df = (
            self.model.transform(df)
            .select("user_id", "prediction")
            .withColumnRenamed("prediction", "cluster")
        )

        log = log.join(df, on="user_id", how="left")
        log = log.groupBy(["cluster", "item_id"]).agg(
            sf.count("item_id").alias("count")
        )
        m = log.groupby("cluster").agg(sf.max("count").alias("max"))
        log = log.join(m, on="cluster", how="left")
        log = log.withColumn(
            "relevance", sf.col("count") / sf.col("max")
        ).drop("count", "max")
        self.recs = log

    @staticmethod
    def _transform_features(df):
        feature_columns = df.drop("user_id").columns
        vec = VectorAssembler(inputCols=feature_columns, outputCol="features")
        return vec.transform(df).select("user_id", "features")

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
        user_features = convert2spark(user_features)
        df = self._transform_features(user_features)
        df = (
            self.model.transform(df)
            .select("user_id", "prediction")
            .withColumnRenamed("prediction", "cluster")
        )
        pred = df.join(self.recs, on="cluster")

        if log is not None:
            log = convert2spark(log)
            pred = pred.join(
                log.withColumnRenamed("item_id", "item")
                .withColumnRenamed("user_id", "user")
                .select("user", "item"),
                on=(sf.col("user_id") == sf.col("user"))
                & (sf.col("item_id") == sf.col("item")),
                how="anti",
            ).drop("user", "item")
        pred = get_top_k_recs(pred, k)
        return pred

    def optimize(
        self,
        train_log: AnyDataFrame,
        test_log: AnyDataFrame,
        user_features_train: AnyDataFrame,
        user_features_test: AnyDataFrame,
        param_grid: Optional[List] = (2, 20),
        criterion: Metric = NDCG(),
        k: int = 10,
        budget: int = 10,
    ):
        """
       Подбирает гиперпараметры с помощью optuna.

       :param train_log: датафрейм для обучения
       :param test_log: датафрейм для проверки качества
       :param user_features_train: датафрейм с признаками пользователей из трейна
       :param user_features_test: датафрейм с признаками пользователей из теста
       :param param_grid: границы перебора количества кластеров
       :param criterion: метрика, которая будет оптимизироваться
       :param k: количество рекомендаций для каждого пользователя
       :param budget: количество попыток при поиске лучших гиперпараметров
       :return: оптимальное количество кластеров
       """
        train = convert2spark(train_log)
        test = convert2spark(test_log)
        user_features_train = convert2spark(user_features_train)
        user_features_test = convert2spark(user_features_test)
        split_data = ColdSplitData(
            train, test, user_features_train, user_features_test,
        )
        study = create_study(direction="maximize", sampler=TPESampler())
        objective = ColdObjective(
            search_space=param_grid,
            split_data=split_data,
            recommender=self,
            criterion=criterion,
            k=k,
        )
        study.optimize(objective, budget)
        return study.best_params["n"]
