"""
Данные сплиттеры делят лог оценок несколькими способами:

- по времени
- по размеру теста случайно
- так, чтобы в тестовую выборку попали только холодные пользователи

Каждый может по запросу удалять холодных пользователей и предметы.
"""

from datetime import datetime
from typing import Optional

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import TimestampType

from sponge_bob_magic.splitters.base_splitter import (Splitter,
                                                      SplitterReturnType)


class DateSplitter(Splitter):
    """
    Делит лог по дате, начиная с которой все записи будут отнесены к тесту.
    """
    def __init__(self,
                 test_start: datetime,
                 drop_cold_items: bool = False,
                 drop_cold_users: bool = False):
        """
        :param test_start: дата в формате ``yyyy-mm-dd``
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
           которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
           которых нет в обучающей
        """
        super().__init__(drop_cold_items, drop_cold_users)
        self.test_start = test_start

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        train = log.filter(
            sf.col("timestamp") < sf.lit(self.test_start).cast(TimestampType())
        )
        test = log.filter(
            sf.col("timestamp") >= sf.lit(self.test_start).cast(
                TimestampType())
        )
        return train, test


class RandomSplitter(Splitter):
    """ Случайным образом распределяет записи на трейн и тест по переданному значению размера теста. """
    def __init__(self,
                 test_size: float,
                 drop_cold_items: bool = False,
                 drop_cold_users: bool = False,
                 seed: Optional[int] = None):
        """
        :param test_size: размер тестовой выборки, от 0 до 1
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
           которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
           которых нет в обучающей
        :param seed: сид для разбиения
        """
        super().__init__(drop_cold_items, drop_cold_users)
        self.seed = seed
        self.test_size = test_size

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        train, test = log.randomSplit(
            [1 - self.test_size, self.test_size],
            self.seed
        )
        return train, test


class ColdUsersSplitter(Splitter):
    """
    На основе желаемого размера тестовой выборки подбирается дата,
    и пользователи, появившиеся после этой даты, считаются холодными, и
    их лог попадает в тестовую выборку.
    """

    def __init__(self,
                 test_size: float,
                 drop_cold_items: bool = False,
                 drop_cold_users: bool = False):
        """
        :param test_size: размер тестовой выборки, от 0 до 1
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
           которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
           которых нет в обучающей
        """
        super().__init__(drop_cold_items, drop_cold_users)
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
            .join(
                start_date_by_user.filter(
                    sf.col("start_dt") >= test_start_date
                ),
                how="inner",
                on="user_id"
            )
            .drop("start_dt")
            .cache()
        )

        return train, test
