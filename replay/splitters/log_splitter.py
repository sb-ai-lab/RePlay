"""
These kind of splitters process log as a whole:

- by time
- at random by test size
- select cold users for test

"""

from typing import Optional, List, Union

from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as sf
from pyspark.sql import Window
from replay.utils.spark_utils import convert2spark

from replay.data import AnyDataFrame
from replay.splitters.base_splitter import (
    Splitter,
    SplitterReturnType,
)


# pylint: disable=too-few-public-methods
class RandomSplitter(Splitter):
    """Assign records into train and test at random."""

    _init_arg_names = [
        "test_size",
        "drop_cold_users",
        "drop_cold_items",
        "seed",
        "user_col",
        "item_col",
        "timestamp_col",
        "rating_col",
        "session_id_col",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: List[float],
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx",
        timestamp_col: Optional[str] = "timestamp",
        rating_col: Optional[str] = "relevance",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: test size 0 to 1
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        :param user_col: user id column name
        :param item_col: item id column name
        :param timestamp_col: timestamp column name
        :param rating_col: rating column name
        :param session_id_col: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            rating_col=rating_col,
            session_id_col=session_id_col,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self._precision = 3
        self.seed = seed
        self.test_size = test_size
        self._sanity_check()

    def _get_order_of_sort(self) -> list:   # pragma: no cover
        pass

    def _sanity_check(self) -> None:
        sum_ratio = round(sum(self.test_size), self._precision)
        if sum_ratio <= 0 or sum_ratio >= 1:
            raise ValueError(f"sum of `ratio` list must be in (0, 1); sum={sum_ratio}")

    def _random_split_spark(self, log: SparkDataFrame, threshold: float) -> Union[SparkDataFrame, SparkDataFrame]:
        log = log.withColumn("_index", sf.row_number().over(Window.orderBy(self.user_col)))
        train, test = log.randomSplit(
            [1 - threshold, threshold], self.seed
        )

        if self.session_id_col:
            test = test.withColumn("is_test", sf.lit(True))
            log = log.join(test, on=log.schema.names, how="left").na.fill({"is_test": False})
            log = self._recalculate_with_session_id_column(log)
            train = log.filter(~sf.col("is_test")).drop("is_test")
            test = log.filter(sf.col("is_test")).drop("is_test")

        train = train.drop("_index")
        test = test.drop("_index")

        return train, test

    def _random_split_pandas(self, log: PandasDataFrame, threshold: float) -> Union[PandasDataFrame, PandasDataFrame]:
        train = log.sample(frac=(1 - threshold), random_state=self.seed)
        test = log.drop(train.index)

        if self.session_id_col:
            log["is_test"] = False
            log.loc[test.index, "is_test"] = True
            log = self._recalculate_with_session_id_column(log)
            train = log[~log["is_test"]].drop(columns=["is_test"])
            test = log[log["is_test"]].drop(columns=["is_test"])
            log = log.drop(columns=["is_test"])

        return train, test

    def _core_split(self, log: AnyDataFrame) -> SplitterReturnType:
        split_method = self._random_split_spark
        if isinstance(log, PandasDataFrame):
            split_method = self._random_split_pandas

        sum_ratio = round(sum(self.test_size), self._precision)
        train, test = split_method(log, sum_ratio)

        res = []
        for ratio in self.test_size:
            test, test1 = split_method(test, round(ratio / sum_ratio, self._precision))
            res.append(test1)
            sum_ratio -= ratio

        return [train] + list(reversed(res))


# pylint: disable=too-few-public-methods
class NewUsersSplitter(Splitter):
    """
    Only new users will be assigned to test set.
    Splits log by timestamp so that test has `test_size` fraction of most recent users.


    >>> from replay.splitters import NewUsersSplitter
    >>> from pyspark.sql import SparkSession
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1,1,2,2,3,4],
    ...    "item_idx": [1,2,3,1,2,3],
    ...    "relevance": [1,2,3,4,5,6],
    ...    "timestamp": [20,40,20,30,10,40]})
    >>> data_frame_spark = SparkSession.builder.getOrCreate().createDataFrame(data_frame)
    >>> data_frame
       user_idx  item_idx  relevance  timestamp
    0         1         1          1         20
    1         1         2          2         40
    2         2         3          3         20
    3         2         1          4         30
    4         3         2          5         10
    5         4         3          6         40
    >>> train, test = NewUsersSplitter(test_size=[0.1]).split(data_frame_spark)
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

    >>> train, test = NewUsersSplitter(test_size=[0.3]).split(data_frame_spark)
    >>> train.show()
    +--------+--------+---------+---------+
    |user_idx|item_idx|relevance|timestamp|
    +--------+--------+---------+---------+
    |       3|       2|        5|       10|
    +--------+--------+---------+---------+
    <BLANKLINE>
    """

    _init_arg_names = [
        "test_size",
        "drop_cold_users",
        "drop_cold_items",
        "user_col",
        "item_col",
        "timestamp_col",
        "rating_col",
        "session_id_col",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: List[float],
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx",
        timestamp_col: Optional[str] = "timestamp",
        rating_col: Optional[str] = "relevance",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: test size 0 to 1
        :param drop_cold_items: flag to drop cold items from test
        :param user_col: user id column name
        :param item_col: item id column name
        :param timestamp_col: timestamp column name
        :param rating_col: rating column name
        :param session_id_col: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            rating_col=rating_col,
            session_id_col=session_id_col,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.test_size = test_size
        self._precision = 3
        self._sanity_check()

    def _sanity_check(self) -> None:
        sum_ratio = round(sum(self.test_size), self._precision)
        if sum_ratio <= 0 or sum_ratio >= 1:
            raise ValueError(f"sum of `ratio` list must be in (0, 1); sum={sum_ratio}")

    def _get_order_of_sort(self) -> list:
        pass

    def _core_split_pandas(self, log: PandasDataFrame, threshold: float) -> Union[PandasDataFrame, PandasDataFrame]:
        start_date_by_user = log.groupby(self.user_col).agg(
            _start_dt_by_user=(self.timestamp_col, "min")
        ).reset_index()
        test_start_date = (
            start_date_by_user
            .groupby("_start_dt_by_user")
            .agg(_num_users_by_start_date=(self.user_col, "count")).reset_index()
            .sort_values(by="_start_dt_by_user", ascending=False)
        )
        test_start_date["_cum_num_users_to_dt"] = test_start_date["_num_users_by_start_date"].cumsum()
        test_start_date["total"] = sum(test_start_date["_num_users_by_start_date"])
        test_start_date = test_start_date[test_start_date["_cum_num_users_to_dt"] >= threshold * test_start_date["total"]]
        test_start = test_start_date["_start_dt_by_user"].max()

        train = log[log[self.timestamp_col] < test_start]
        test = log.merge(
            start_date_by_user[start_date_by_user["_start_dt_by_user"] >= test_start], 
            how="inner",
            on=self.user_col).drop(columns=["_start_dt_by_user"]
        )

        if self.session_id_col:
            log["is_test"] = False
            log.loc[test.index, "is_test"] = True
            log = self._recalculate_with_session_id_column(log)
            train = log[~log["is_test"]].drop(columns=["is_test"])
            test = log[log["is_test"]].drop(columns=["is_test"])
            log = log.drop(columns=["is_test"])

        return train, test

    def _core_split_spark(self, log: SparkDataFrame, threshold: float) -> Union[SparkDataFrame, SparkDataFrame]:
        start_date_by_user = log.groupby(self.user_col).agg(
            sf.min(self.timestamp_col).alias("_start_dt_by_user")
        )
        test_start_date = (
            start_date_by_user.groupby("_start_dt_by_user")
            .agg(sf.count(self.user_col).alias("_num_users_by_start_date"))
            .select(
                "_start_dt_by_user",
                sf.sum("_num_users_by_start_date")
                .over(Window.orderBy(sf.desc("_start_dt_by_user")))
                .alias("_cum_num_users_to_dt"),
                sf.sum("_num_users_by_start_date").over(Window.orderBy(sf.lit(1))).alias("total"),
            )
            .filter(sf.col("_cum_num_users_to_dt") >= sf.col("total") * threshold)
            .agg(sf.max("_start_dt_by_user"))
            .head()[0]
        )

        train = log.filter(sf.col(self.timestamp_col) < test_start_date)
        test = log.join(
            start_date_by_user.filter(sf.col("_start_dt_by_user") >= test_start_date),
            how="inner",
            on=self.user_col,
        ).drop("_start_dt_by_user")

        if self.session_id_col:
            test = test.withColumn("is_test", sf.lit(True))
            log = log.join(test, on=log.schema.names, how="left").na.fill({"is_test": False})
            log = self._recalculate_with_session_id_column(log)
            train = log.filter(~sf.col("is_test")).drop("is_test")
            test = log.filter(sf.col("is_test")).drop("is_test")

        return train, test

    def _core_split(self, log: AnyDataFrame) -> SplitterReturnType:
        split_method = self._core_split_spark
        if isinstance(log, PandasDataFrame):
            split_method = self._core_split_pandas

        sum_ratio = round(sum(self.test_size), self._precision)
        train, test = split_method(log, sum_ratio)

        res = []
        for ratio in self.test_size:
            test, test1 = split_method(test, round(ratio / sum_ratio, self._precision))
            res.append(test1)
            sum_ratio -= ratio

        return [train] + list(reversed(res))


# pylint: disable=too-few-public-methods
class ColdUserRandomSplitter(Splitter):
    """
    Test set consists of all actions of randomly chosen users.
    """

    # для использования в тестах

    _init_arg_names = [
        "test_size",
        "drop_cold_users",
        "drop_cold_items",
        "seed",
        "user_col",
        "item_col",
        "timestamp_col",
        "session_id_col",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: List[float],
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx",
        timestamp_col: Optional[str] = "timestamp",
        rating_col: Optional[str] = "relevance",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: fraction of users to be in test
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        :param user_col: user id column name
        :param item_col: item id column name
        :param timestamp_col: timestamp column name
        :param rating_col: rating column name
        :param session_id_col: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            rating_col=rating_col,
            session_id_col=session_id_col,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.seed = seed
        self._precision = 3
        self.test_size = test_size
        self._sanity_check()

    def _sanity_check(self) -> None:
        sum_ratio = round(sum(self.test_size), self._precision)
        if sum_ratio <= 0 or sum_ratio >= 1:
            raise ValueError(f"sum of `ratio` list must be in (0, 1); sum={sum_ratio}")

    def _get_order_of_sort(self) -> list: # pragma: no cover
        pass

    def _core_split_pandas(self, log: PandasDataFrame, threshold: float) -> Union[PandasDataFrame, PandasDataFrame]:
        index_name = log.index.name
        df = log.reset_index()
        users = PandasDataFrame(df[self.user_col].unique(), columns=["user_idx"])
        train_users = users.sample(frac=(1 - threshold), random_state=self.seed)
        test_users = users.drop(train_users.index)

        train = df.merge(train_users, on=self.user_col, how="inner")
        test = df.merge(test_users, on=self.user_col, how="inner")
        train.set_index("index", inplace=True)
        test.set_index("index", inplace=True)

        train.index.name = index_name
        test.index.name = index_name

        if self.session_id_col:
            log["is_test"] = False
            log.loc[test.index, "is_test"] = True
            log = self._recalculate_with_session_id_column(log)
            train = log[~log["is_test"]].drop(columns=["is_test"])
            test = log[log["is_test"]].drop(columns=["is_test"])
            log = log.drop(columns=["is_test"])

        return train, test

    def _core_split_spark(self, log: SparkDataFrame, threshold: float) -> Union[SparkDataFrame, SparkDataFrame]:
        users = log.select(self.user_col).distinct()
        train_users, test_users = users.randomSplit(
            [1 - threshold, threshold],
            seed=self.seed,
        )
        train = log.join(train_users, on=self.user_col, how="inner")
        test = log.join(test_users, on=self.user_col, how="inner")

        if self.session_id_col:
            test = test.withColumn("is_test", sf.lit(True))
            log = log.join(test, on=log.schema.names, how="left").na.fill({"is_test": False})
            log = self._recalculate_with_session_id_column(log)
            train = log.filter(~sf.col("is_test")).drop("is_test")
            test = log.filter(sf.col("is_test")).drop("is_test")

        return train, test

    def _core_split(self, log: AnyDataFrame) -> SplitterReturnType:
        split_method = self._core_split_spark
        if isinstance(log, PandasDataFrame):
            split_method = self._core_split_pandas

        sum_ratio = round(sum(self.test_size), self._precision)
        train, test = split_method(log, sum_ratio)

        res = []
        for ratio in self.test_size:
            test, test1 = split_method(test, round(ratio / sum_ratio, self._precision))
            res.append(test1)
            sum_ratio -= ratio

        return [train] + list(reversed(res))
