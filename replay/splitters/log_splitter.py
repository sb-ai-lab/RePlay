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

from replay.splitters.base_splitter import (
    Splitter,
    SplitterReturnType,
)


# pylint: disable=too-few-public-methods
class DateSplitter(Splitter):
    """
    Делит лог по дате, начиная с которой все записи будут отнесены к тесту.
    """

    def __init__(
        self,
        test_start: datetime,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
    ):
        """
        :param test_start: дата в формате ``yyyy-mm-dd``
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
           которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
           которых нет в обучающей
        """
        super().__init__(
            drop_cold_items=drop_cold_items, drop_cold_users=drop_cold_users
        )
        self.test_start = test_start

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        train = log.filter(
            sf.col("timestamp") < sf.lit(self.test_start).cast(TimestampType())
        )
        test = log.filter(
            sf.col("timestamp")
            >= sf.lit(self.test_start).cast(TimestampType())
        )
        return train, test


# pylint: disable=too-few-public-methods
class RandomSplitter(Splitter):
    """ Случайным образом распределяет записи на трейн и тест по переданному значению размера теста. """

    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
    ):
        """
        :param test_size: размер тестовой выборки, от 0 до 1
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
           которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
           которых нет в обучающей
        :param seed: сид для разбиения
        """
        super().__init__(
            drop_cold_items=drop_cold_items, drop_cold_users=drop_cold_users
        )
        self.seed = seed
        self.test_size = test_size

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        train, test = log.randomSplit(
            [1 - self.test_size, self.test_size], self.seed
        )
        return train, test


# pylint: disable=too-few-public-methods
class NewUsersSplitter(Splitter):
    """
    Позволяет оценить работу модели на новых/не идентифицированных пользователях.
    Разбивает лог по timestamp таким образом, чтобы в test оказалась определеннная в test_size доля пользователей,
    позднее всего появившихся в логе. Назовем пользователей, которые появились в логе до даты разбиения, существующими,
    а пользователей, появившихся в эту дату и позднее - новыми. После разбиения в train будет находиться история
    взаимодействия существующих пользователей до даты разбиения, а в test - история новых пользователей.
    Пользователи из test отсутствуют в train и будут являться холодными для обучаемой модели.

    >>> from replay.splitters import NewUsersSplitter
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_id": [1,1,2,2,3,4],
    ...    "item_id": [1,2,3,1,2,3],
    ...    "relevance": [1,2,3,4,5,6],
    ...    "timestamp": [20,40,20,30,10,40]})
    >>> data_frame
       user_id  item_id  relevance  timestamp
    0        1        1          1         20
    1        1        2          2         40
    2        2        3          3         20
    3        2        1          4         30
    4        3        2          5         10
    5        4        3          6         40
    >>> train, test = NewUsersSplitter(test_size=0.1).split(data_frame)
    >>> train.show()
    +-------+-------+---------+---------+
    |user_id|item_id|relevance|timestamp|
    +-------+-------+---------+---------+
    |      1|      1|        1|       20|
    |      2|      3|        3|       20|
    |      2|      1|        4|       30|
    |      3|      2|        5|       10|
    +-------+-------+---------+---------+
    <BLANKLINE>
    >>> test.show()
    +-------+-------+---------+---------+
    |user_id|item_id|relevance|timestamp|
    +-------+-------+---------+---------+
    |      4|      3|        6|       40|
    +-------+-------+---------+---------+
    <BLANKLINE>

    Сплиттер оставляет в train только историю существующих пользователей до даты разбиения, и поэтому,
    если новых пользователей мало или они появились давно и/или одновременно,
    размер train может существенно сократиться даже при небольшом test_size:

    >>> train, test = NewUsersSplitter(test_size=0.3).split(data_frame)
    >>> train.show()
    +-------+-------+---------+---------+
    |user_id|item_id|relevance|timestamp|
    +-------+-------+---------+---------+
    |      3|      2|        5|       10|
    +-------+-------+---------+---------+
    <BLANKLINE>
    """

    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
    ):
        """
        :param test_size: размер тестовой выборки, от 0 до 1
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
           которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
           которых нет в обучающей
        """
        super().__init__(
            drop_cold_items=drop_cold_items, drop_cold_users=drop_cold_users
        )
        self.test_size = test_size

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        start_date_by_user = (
            log.groupby("user_id")
            .agg(sf.min("timestamp").alias("start_dt"))
            .cache()
        )
        test_start_date = (
            start_date_by_user.groupby("start_dt")
            .agg(sf.count("user_id").alias("cnt"))
            .select(
                "start_dt",
                sf.sum("cnt")
                .over(Window.orderBy(sf.desc("start_dt")))
                .alias("cnt"),
                sf.sum("cnt").over(Window.orderBy(sf.lit(1))).alias("total"),
            )
            .filter(sf.col("cnt") >= sf.col("total") * self.test_size)
            .agg(sf.max("start_dt"))
            .head()[0]
        )

        train = log.filter(sf.col("timestamp") < test_start_date).cache()

        test = (
            log.join(
                start_date_by_user.filter(
                    sf.col("start_dt") >= test_start_date
                ),
                how="inner",
                on="user_id",
            )
            .drop("start_dt")
            .cache()
        )

        return train, test


# pylint: disable=too-few-public-methods
class ColdUserRandomSplitter(Splitter):
    """
    В тестовую выборку попадают все действия случайно отобранных пользователей в заданном количествею.
    """

    # для использования в тестах
    seed: Optional[int] = None

    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
    ):
        """
        :param test_size: желаемая доля всех пользователей, которые должны оказаться в тестовой выборке
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
           которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
           которых нет в обучающей
        """
        super().__init__(
            drop_cold_items=drop_cold_items, drop_cold_users=drop_cold_users
        )
        self.test_size = test_size

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        users = log.select("user_id").distinct()
        train_users, test_users = users.randomSplit(
            [1 - self.test_size, self.test_size], seed=self.seed,
        )
        train = log.join(train_users, on="user_id", how="inner").cache()
        test = log.join(test_users, on="user_id", how="inner").cache()
        return train, test
