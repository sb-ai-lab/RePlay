"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.

В скрипте собраны классы по разбиению всего лога взаимодействий пользователей и
объектов на обучающую и тестовую выборки.
Способы разбиения - по времени и случайно, а также способ,
когда в тестовую выборку попадают только холодные пользователи.
"""
from datetime import datetime

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.types import TimestampType

from sponge_bob_magic.constants import LOG_SCHEMA
from sponge_bob_magic.splitters.base_splitter import (Splitter,
                                                      SplitterReturnType)


class LogSplitByDateSplitter(Splitter):
    """ Делит лог взаимодействия по времени на обучающую и тестовую выборки. """

    def __init__(self, spark: SparkSession,
                 test_start: datetime):
        """
        :param spark: инициализированная спарк-сессия
        :param test_start: дата в формате `yyyy-mm-dd`
        """
        super().__init__(spark)

        self.test_start = test_start

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        train = log.filter(
            sf.col("timestamp") < sf.lit(self.test_start).cast(TimestampType())
        )
        test = log.filter(
            sf.col("timestamp") >= sf.lit(self.test_start).cast(
                TimestampType())
        )
        return train, train, test


class LogSplitRandomlySplitter(Splitter):
    """ Делит лог взаимодействия случайно на обучающую и тестовую выборки. """

    def __init__(self, spark: SparkSession,
                 test_size: float,
                 seed: int):
        """
        :param seed: сид для разбиения
        :param spark: инициализированная спарк-сессия
        :param test_size: размер тестовой выборки, от 0 до 1
        """
        super().__init__(spark)

        self.seed = seed
        self.test_size = test_size

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        train, test = log.randomSplit(
            [1 - self.test_size, self.test_size],
            self.seed
        )
        return train, train, test


class ColdUsersExtractingSplitter(Splitter):
    """
    Делит лог взаимодействия на обучающую и тестовую выборки,
    таким образом, чтобы в тестовой выборке оказались холодные пользователи.
    На основе желаемого размера тестовой выборки подбирается дата,
    и пользователи, появившиеся после этой даты, считаются холодными, и
    их лог попадает в тестовую выборку.
    """

    def __init__(self, spark: SparkSession,
                 test_size: float):
        """
        :param spark: инициализированная спарк-сессия,
            в рамках которой будет происходить обработка данных
        :param test_size: размер тестовой выборки, от 0 до 1
        """
        super().__init__(spark)

        self.test_size = test_size

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        start_date_by_user = (
            log
            .groupby("user_id")
            .agg(sf.min("timestamp").alias("start_dt"))
            .cache()
        )
        test_start_date = (
            start_date_by_user
            .groupby("start_dt")
            .agg(sf.count("user_id").alias("cnt"))
            .select("start_dt",
                    sf.sum("cnt").over(Window
                                       .orderBy(sf.desc("start_dt")))
                    .alias("cnt"),
                    sf.sum("cnt").over(Window
                                       .orderBy(sf.lit(1)))
                    .alias("total")
                    )
            .filter(sf.col("cnt") >= sf.col("total") * self.test_size)
            .agg(sf.max("start_dt"))
            .head()[0]
        )

        train = log.filter(sf.col("timestamp") < test_start_date).cache()

        test = (
            log
            .join(start_date_by_user.filter(
                sf.col("start_dt") >= test_start_date
            ),
                how="inner",
                on="user_id"
            )
            .drop("start_dt")
            .cache()
        )

        predict_input = self.spark.createDataFrame(data=[], schema=LOG_SCHEMA)

        return train, predict_input, test
