"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from datetime import datetime
from typing import Tuple, Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import col, row_number, rand
from pyspark.sql.types import TimestampType


class ValidationSchemes:
    """ Различные методы разбиения на обучающую и тестовую выборки. """

    def __init__(self, spark: SparkSession):
        """
        :param spark: инициализированная спарк-сессия,
            в рамках которой будет происходить обработка данных
        """
        self.spark = spark

    @staticmethod
    def _drop_cold_items_and_users(train: DataFrame,
                                   test: DataFrame,
                                   drop_cold_items: bool,
                                   drop_cold_users: bool) -> DataFrame:
        """
        Удаляет из тестовой выборки холодных users и холодные items.

        :param train: обучающая выборка, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, context, relevance]`
        :param test: тестовая выборка как train
        :param drop_cold_items: если True, удалить холодные items
        :param drop_cold_users: если True, удалить холодные users
        :return: тестовая выборка без холодных users / items
        """
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
        return test

    @staticmethod
    def log_split_by_date(
            log: DataFrame,
            test_start: datetime,
            drop_cold_items: bool,
            drop_cold_users: bool
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Разбить лог действий пользователей по дате на обучающую и тестовую
        выборки.

        :param log: лог взаимодействия, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, context, relevance]`
        :param test_start: дата в формате `yyyy-mm-dd`
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
            которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
            которых нет в обучающей
        :returns: тройка таблиц структуры, аналогичной входной
            `train, test_input, test`
        """
        train = log.filter(
            F.col("timestamp") < F.lit(test_start).cast(TimestampType())
        )
        test = log.filter(
            F.col("timestamp") >= F.lit(test_start).cast(TimestampType())
        )

        test = ValidationSchemes._drop_cold_items_and_users(
            train, test,
            drop_cold_items, drop_cold_users
        )
        return train, train, test

    @staticmethod
    def log_split_randomly(
            log: DataFrame,
            test_size: float,
            drop_cold_items: bool,
            drop_cold_users: bool,
            seed: int = 1234
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        """
        Разбить лог действий пользователей рандомно на обучающую и тестовую
        выборки.

        :param seed: рандомный сид, нужен для повторения разбиения
        :param test_size: размер тестовой выборки, от 0 до 1
        :param log: лог взаимодействия, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, context, relevance]`
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
            которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
            которых нет в обучающей
        :returns: тройка спарк-датафреймов структуры, аналогичной входной
            `train, test_input, test`
        """
        train, test = log.randomSplit([1 - test_size, test_size], seed)

        test = ValidationSchemes._drop_cold_items_and_users(
            train, test,
            drop_cold_items, drop_cold_users
        )
        return train, train, test

    @staticmethod
    def extract_cold_users(
            log: DataFrame,
            test_size: float
    ) -> Tuple[DataFrame, Optional[DataFrame], DataFrame]:
        """
        Выделить из лога действий пользователей случайную долю "холодных"
        пользователей.

        :param test_size: доля ото всех пользовтелей, которую должны составлять
            "холодные", от 0 до 1
        :param log: лог взаимодействия, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, context, relevance]`
        :returns: тройка спарк-датафреймов структуры, аналогичной входной
            `train, test_input, test`
        """
        start_date_by_user = (
            log
            .groupby("user_id")
            .agg(F.min("timestamp").alias("start_dt"))
            .cache()
        )
        test_start_date = (
            start_date_by_user
            .groupby("start_dt")
            .agg(F.count("user_id").alias("cnt"))
            .select(
                "start_dt",
                F.sum("cnt").over(Window.orderBy(
                    F.desc("start_dt")
                )).alias("cnt"),
                F.sum("cnt").over(Window.orderBy(F.lit(1))).alias("total")
            )
            .filter(F.col("cnt") >= F.col("total") * test_size)
            .agg(F.max("start_dt"))
            .head()[0]
        )
        train = log.filter(F.col("timestamp") < test_start_date).cache()
        test = (
            log
            .join(
                start_date_by_user.filter(
                    F.col("start_dt") >= test_start_date
                ),
                how="inner",
                on="user_id"
            )
            .drop("start_dt")
            .cache()
        )
        return train, None, test

    @staticmethod
    def log_split_randomly_by_user_num(
            log: DataFrame,
            test_size: int,
            drop_cold_items: bool,
            drop_cold_users: bool,
            seed: int = 1234
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        log = log.withColumn("rand", rand(seed))
        res = log \
            .withColumn("row_num", row_number()
                        .over(Window
                              .partitionBy("user_id")
                              .orderBy("rand")
                              )
                        ).cache()

        train = res.filter(res.row_num > test_size).drop("rand", "row_num")
        test = res.filter(res.row_num <= test_size).drop("rand", "row_num")

        test = ValidationSchemes._drop_cold_items_and_users(
            train, test,
            drop_cold_items, drop_cold_users
        )
        return train, train, test

    @staticmethod
    def log_split_randomly_by_user_frac(
            log: DataFrame,
            test_size: float,
            drop_cold_items: bool,
            drop_cold_users: bool,
            seed: int = 1234
    ) -> Tuple[DataFrame, DataFrame, DataFrame]:
        log = log.withColumn("rand", rand(seed))
        counts = log.groupBy("user_id").count()
        res = log \
            .withColumn("row_num", row_number()
                        .over(Window
                              .partitionBy("user_id")
                              .orderBy('rand')
                              )
                        )
        res = res.join(counts, on="user_id", how="left")
        res = res.withColumn("frac", col("row_num") / col("count")).cache()

        train = res.filter(res.row_num > test_size).drop("rand", "row_num", "count", "frac")
        test = res.filter(res.row_num <= test_size).drop("rand", "row_num", "count", "frac")

        test = ValidationSchemes._drop_cold_items_and_users(
            train, test,
            drop_cold_items, drop_cold_users
        )
        return train, train, test
