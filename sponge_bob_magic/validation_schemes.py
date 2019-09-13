"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
from datetime import datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import TimestampType


class ValidationSchemes:
    """ различные методы разбиения на обучающую и тестовую выборки """
    def __init__(self, spark: SparkSession):
        """
        :param spark: сессия Spark, в рамках которой будет происходить
        обработка данных
        """
        self.spark = spark

    @staticmethod
    def log_split_by_date(
            log: DataFrame,
            test_start: datetime,
            drop_cold_items: bool,
            drop_cold_users: bool
    ) -> (DataFrame, DataFrame, DataFrame):
        """
        разбить лог действий пользователей по дате на обучающую и тестовую
        выборки

        :param log: таблица с колонками
        (timestamp, user_id, item_id, context, relevance)
        :param test_start: дата в формате yyyy-mm-dd
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
        которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
        которых нет в обучающей
        :returns: тройка таблиц структуры, аналогичной входной
        (train, test_inpit, test)
        """
        train = log.filter(
            col("timestamp") < lit(test_start).cast(TimestampType())
        )
        test = log.filter(
            col("timestamp") >= lit(test_start).cast(TimestampType())
        )
        if drop_cold_items:
            test = test.join(
                train.select("item_id").distinct(),
                how="inner",
                on="item_id"
            )
        if drop_cold_users:
            test = test.join(
                train.select("user_id").distinct(),
                how="inner",
                on="user_id"
            )
        return (train, train, test)
