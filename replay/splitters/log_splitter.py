"""
These kind of splitters process log as a whole:

- by time
- at random by test size
- select cold users for test

"""

from datetime import datetime
from typing import Optional, Union

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, Window

from replay.splitters.base_splitter import (
    Splitter,
    SplitterReturnType,
)


# pylint: disable=too-few-public-methods
class DateSplitter(Splitter):
    """
    Split into train and test by date.
    """

    def __init__(
        self,
        test_start: Union[datetime, float, str, int],
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
    ):
        """
        :param test_start: string``yyyy-mm-dd``, int unix timestamp, datetime or a
            fraction for test size to determine data automatically
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        """
        super().__init__(
            drop_cold_items=drop_cold_items, drop_cold_users=drop_cold_users
        )
        self.test_start = test_start

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        if isinstance(self.test_start, float):
            dates = log.select("timestamp").withColumn(
                "idx", sf.row_number().over(Window.orderBy("timestamp"))
            )
            test_start = int(dates.count() * (1 - self.test_start)) + 1
            test_start = (
                dates.filter(sf.col("idx") == test_start)
                .select("timestamp")
                .collect()[0][0]
            )
        else:
            dtype = dict(log.dtypes)["timestamp"]
            test_start = sf.lit(self.test_start).cast("timestamp").cast(dtype)
        train = log.filter(sf.col("timestamp") < test_start)
        test = log.filter(sf.col("timestamp") >= test_start)
        return train, test


# pylint: disable=too-few-public-methods
class RandomSplitter(Splitter):
    """Assign records into train and test at random."""

    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
    ):
        """
        :param test_size: test size 0 to 1
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        """
        super().__init__(
            drop_cold_items=drop_cold_items, drop_cold_users=drop_cold_users
        )
        self.seed = seed
        self.test_size = test_size
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be 0 to 1")

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        train, test = log.randomSplit(
            [1 - self.test_size, self.test_size], self.seed
        )
        return train, test


# pylint: disable=too-few-public-methods
class NewUsersSplitter(Splitter):
    """
    Only new users will be assigned to test set.
    Splits log by timestamp so that test has `test_size` fraction of most recent users.


    >>> from replay.splitters import NewUsersSplitter
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1,1,2,2,3,4],
    ...    "item_idx": [1,2,3,1,2,3],
    ...    "relevance": [1,2,3,4,5,6],
    ...    "timestamp": [20,40,20,30,10,40]})
    >>> data_frame
       user_idx  item_idx  relevance  timestamp
    0         1         1          1         20
    1         1         2          2         40
    2         2         3          3         20
    3         2         1          4         30
    4         3         2          5         10
    5         4         3          6         40
    >>> train, test = NewUsersSplitter(test_size=0.1).split(data_frame)
    >>> train.show()
    +--------+--------+---------+---------+
    |user_idx|item_idx|relevance|timestamp|
    +--------+--------+---------+---------+
    |       1|       1|        1|       20|
    |       2|       3|        3|       20|
    |       2|       1|        4|       30|
    |       3|       2|        5|       10|
    +--------+--------+---------+---------+
    <BLANKLINE>
    >>> test.show()
    +--------+--------+---------+---------+
    |user_idx|item_idx|relevance|timestamp|
    +--------+--------+---------+---------+
    |       4|       3|        6|       40|
    +--------+--------+---------+---------+
    <BLANKLINE>

    Train DataFrame can be drastically reduced even with moderate
    `test_size` if the amount of new users is small.

    >>> train, test = NewUsersSplitter(test_size=0.3).split(data_frame)
    >>> train.show()
    +--------+--------+---------+---------+
    |user_idx|item_idx|relevance|timestamp|
    +--------+--------+---------+---------+
    |       3|       2|        5|       10|
    +--------+--------+---------+---------+
    <BLANKLINE>
    """

    def __init__(self, test_size: float, drop_cold_items: bool = False):
        """
        :param test_size: test size 0 to 1
        :param drop_cold_items: flag to drop cold items from test
        """
        super().__init__(
            drop_cold_items=drop_cold_items, drop_cold_users=False
        )
        self.test_size = test_size
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be 0 to 1")

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        start_date_by_user = log.groupby("user_idx").agg(
            sf.min("timestamp").alias("start_dt")
        )
        test_start_date = (
            start_date_by_user.groupby("start_dt")
            .agg(sf.count("user_idx").alias("cnt"))
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

        train = log.filter(sf.col("timestamp") < test_start_date)

        test = log.join(
            start_date_by_user.filter(sf.col("start_dt") >= test_start_date),
            how="inner",
            on="user_idx",
        ).drop("start_dt")
        return train, test


# pylint: disable=too-few-public-methods
class ColdUserRandomSplitter(Splitter):
    """
    Test set consists of all actions of randomly chosen users.
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
        :param test_size: fraction of users to be in test
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        """
        super().__init__(
            drop_cold_items=drop_cold_items, drop_cold_users=drop_cold_users
        )
        self.test_size = test_size

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        users = log.select("user_idx").distinct()
        train_users, test_users = users.randomSplit(
            [1 - self.test_size, self.test_size], seed=self.seed,
        )
        train = log.join(train_users, on="user_idx", how="inner")
        test = log.join(test_users, on="user_idx", how="inner")
        return train, test
