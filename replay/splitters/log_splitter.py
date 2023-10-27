"""
These kind of splitters process interactions as a whole:

- by time
- at random by test size
- select cold users for test

"""

from typing import Optional, Union

from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as sf
from pyspark.sql import Window

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
        "query_column",
        "item_column",
        "timestamp_column",
        "rating_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
        query_column: str = "user_id",
        item_column: Optional[str] = "item_id",
        timestamp_column: Optional[str] = "timestamp",
        rating_column: Optional[str] = "relevance",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: test size 0 to 1
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        :param query_column: query id column name
        :param item_column: item id column name
        :param timestamp_column: timestamp column name
        :param rating_column: rating column name
        :param session_id_column: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            query_column=query_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            rating_column=rating_column,
            session_id_column=session_id_column,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.seed = seed
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be 0 to 1")
        self.test_size = test_size

    def _get_order_of_sort(self) -> list:   # pragma: no cover
        pass

    def _random_split_spark(self, interactions: SparkDataFrame, threshold: float) -> Union[SparkDataFrame, SparkDataFrame]:
        interactions = interactions.withColumn("_index", sf.row_number().over(Window.orderBy(self.query_column)))
        train, test = interactions.randomSplit(
            [1 - threshold, threshold], self.seed
        )

        if self.session_id_column:
            test = test.withColumn("is_test", sf.lit(True))
            interactions = interactions.join(test, on=interactions.schema.names, how="left").na.fill({"is_test": False})
            interactions = self._recalculate_with_session_id_column(interactions)
            train = interactions.filter(~sf.col("is_test")).drop("is_test")
            test = interactions.filter(sf.col("is_test")).drop("is_test")

        train = train.drop("_index")
        test = test.drop("_index")

        return train, test

    def _random_split_pandas(self, interactions: PandasDataFrame, threshold: float) -> Union[PandasDataFrame, PandasDataFrame]:
        train = interactions.sample(frac=(1 - threshold), random_state=self.seed)
        test = interactions.drop(train.index)

        if self.session_id_column:
            interactions["is_test"] = False
            interactions.loc[test.index, "is_test"] = True
            interactions = self._recalculate_with_session_id_column(interactions)
            train = interactions[~interactions["is_test"]].drop(columns=["is_test"])
            test = interactions[interactions["is_test"]].drop(columns=["is_test"])
            interactions = interactions.drop(columns=["is_test"])

        return train, test

    def _core_split(self, interactions: AnyDataFrame) -> SplitterReturnType:
        split_method = self._random_split_spark
        if isinstance(interactions, PandasDataFrame):
            split_method = self._random_split_pandas

        return split_method(interactions, self.test_size)


# pylint: disable=too-few-public-methods
class NewUsersSplitter(Splitter):
    """
    Only new users will be assigned to test set.
    Splits interactions by timestamp so that test has `test_size` fraction of most recent users.


    >>> from replay.splitters import NewUsersSplitter
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_id": [1,1,2,2,3,4],
    ...    "item_id": [1,2,3,1,2,3],
    ...    "relevance": [1,2,3,4,5,6],
    ...    "timestamp": [20,40,20,30,10,40]})
    >>> data_frame
        user_id   item_id  relevance  timestamp
    0         1         1          1         20
    1         1         2          2         40
    2         2         3          3         20
    3         2         1          4         30
    4         3         2          5         10
    5         4         3          6         40
    >>> train, test = NewUsersSplitter(test_size=0.1).split(data_frame)
    >>> train
       user_id  item_id  relevance  timestamp
    0        1        1          1         20
    2        2        3          3         20
    3        2        1          4         30
    4        3        2          5         10
    <BLANKLINE>
    >>> test
       user_id  item_id  relevance  timestamp
    0        4        3          6         40
    <BLANKLINE>

    Train DataFrame can be drastically reduced even with moderate
    `test_size` if the amount of new users is small.

    >>> train, test = NewUsersSplitter(test_size=0.3).split(data_frame)
    >>> train
       user_id  item_id  relevance  timestamp
    4        3        2          5         10
    <BLANKLINE>
    """

    _init_arg_names = [
        "test_size",
        "drop_cold_users",
        "drop_cold_items",
        "query_column",
        "item_column",
        "timestamp_column",
        "rating_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        query_column: str = "user_id",
        item_column: Optional[str] = "item_id",
        timestamp_column: Optional[str] = "timestamp",
        rating_column: Optional[str] = "relevance",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: test size 0 to 1
        :param drop_cold_items: flag to drop cold items from test
        :param query_column: query id column name
        :param item_column: item id column name
        :param timestamp_column: timestamp column name
        :param rating_column: rating column name
        :param session_id_column: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            query_column=query_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            rating_column=rating_column,
            session_id_column=session_id_column,
            session_id_processing_strategy=session_id_processing_strategy
        )
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be 0 to 1")
        self.test_size = test_size

    def _get_order_of_sort(self) -> list:   # pragma: no cover
        pass

    def _core_split_pandas(self, interactions: PandasDataFrame, threshold: float) -> Union[PandasDataFrame, PandasDataFrame]:
        start_date_by_user = interactions.groupby(self.query_column).agg(
            _start_dt_by_user=(self.timestamp_column, "min")
        ).reset_index()
        test_start_date = (
            start_date_by_user
            .groupby("_start_dt_by_user")
            .agg(_num_users_by_start_date=(self.query_column, "count")).reset_index()
            .sort_values(by="_start_dt_by_user", ascending=False)
        )
        test_start_date["_cum_num_users_to_dt"] = test_start_date["_num_users_by_start_date"].cumsum()
        test_start_date["total"] = sum(test_start_date["_num_users_by_start_date"])
        test_start_date = test_start_date[
            test_start_date["_cum_num_users_to_dt"] >= threshold * test_start_date["total"]
        ]
        test_start = test_start_date["_start_dt_by_user"].max()

        train = interactions[interactions[self.timestamp_column] < test_start]
        test = interactions.merge(
            start_date_by_user[start_date_by_user["_start_dt_by_user"] >= test_start],
            how="inner",
            on=self.query_column
        ).drop(columns=["_start_dt_by_user"])

        if self.session_id_column:
            interactions["is_test"] = False
            interactions.loc[test.index, "is_test"] = True
            interactions = self._recalculate_with_session_id_column(interactions)
            train = interactions[~interactions["is_test"]].drop(columns=["is_test"])
            test = interactions[interactions["is_test"]].drop(columns=["is_test"])
            interactions = interactions.drop(columns=["is_test"])

        return train, test

    def _core_split_spark(self, interactions: SparkDataFrame, threshold: float) -> Union[SparkDataFrame, SparkDataFrame]:
        start_date_by_user = interactions.groupby(self.query_column).agg(
            sf.min(self.timestamp_column).alias("_start_dt_by_user")
        )
        test_start_date = (
            start_date_by_user.groupby("_start_dt_by_user")
            .agg(sf.count(self.query_column).alias("_num_users_by_start_date"))
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

        train = interactions.filter(sf.col(self.timestamp_column) < test_start_date)
        test = interactions.join(
            start_date_by_user.filter(sf.col("_start_dt_by_user") >= test_start_date),
            how="inner",
            on=self.query_column,
        ).drop("_start_dt_by_user")

        if self.session_id_column:
            test = test.withColumn("is_test", sf.lit(True))
            interactions = interactions.join(test, on=interactions.schema.names, how="left").na.fill({"is_test": False})
            interactions = self._recalculate_with_session_id_column(interactions)
            train = interactions.filter(~sf.col("is_test")).drop("is_test")
            test = interactions.filter(sf.col("is_test")).drop("is_test")

        return train, test

    def _core_split(self, interactions: AnyDataFrame) -> SplitterReturnType:
        split_method = self._core_split_spark
        if isinstance(interactions, PandasDataFrame):
            split_method = self._core_split_pandas

        return split_method(interactions, self.test_size)


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
        "query_column",
        "item_column",
        "timestamp_column",
        "rating_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
        query_column: str = "user_id",
        item_column: Optional[str] = "item_id",
        timestamp_column: Optional[str] = "timestamp",
        rating_column: Optional[str] = "relevance",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: fraction of users to be in test
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        :param query_column: query id column name
        :param item_column: item id column name
        :param timestamp_column: timestamp column name
        :param rating_column: rating column name
        :param session_id_column: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            query_column=query_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            rating_column=rating_column,
            session_id_column=session_id_column,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.seed = seed
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be 0 to 1")
        self.test_size = test_size

    def _get_order_of_sort(self) -> list:   # pragma: no cover
        pass

    def _core_split_pandas(self, interactions: PandasDataFrame, threshold: float) -> Union[PandasDataFrame, PandasDataFrame]:
        index_name = interactions.index.name
        df = interactions.reset_index()
        users = PandasDataFrame(df[self.query_column].unique(), columns=[self.query_column])
        train_users = users.sample(frac=(1 - threshold), random_state=self.seed)
        test_users = users.drop(train_users.index)

        train = df.merge(train_users, on=self.query_column, how="inner")
        test = df.merge(test_users, on=self.query_column, how="inner")
        train.set_index("index", inplace=True)
        test.set_index("index", inplace=True)

        train.index.name = index_name
        test.index.name = index_name

        if self.session_id_column:
            interactions["is_test"] = False
            interactions.loc[test.index, "is_test"] = True
            interactions = self._recalculate_with_session_id_column(interactions)
            train = interactions[~interactions["is_test"]].drop(columns=["is_test"])
            test = interactions[interactions["is_test"]].drop(columns=["is_test"])
            interactions = interactions.drop(columns=["is_test"])

        return train, test

    def _core_split_spark(self, interactions: SparkDataFrame, threshold: float) -> Union[SparkDataFrame, SparkDataFrame]:
        users = interactions.select(self.query_column).distinct()
        train_users, test_users = users.randomSplit(
            [1 - threshold, threshold],
            seed=self.seed,
        )
        train = interactions.join(train_users, on=self.query_column, how="inner")
        test = interactions.join(test_users, on=self.query_column, how="inner")

        if self.session_id_column:
            test = test.withColumn("is_test", sf.lit(True))
            interactions = interactions.join(test, on=interactions.schema.names, how="left").na.fill({"is_test": False})
            interactions = self._recalculate_with_session_id_column(interactions)
            train = interactions.filter(~sf.col("is_test")).drop("is_test")
            test = interactions.filter(sf.col("is_test")).drop("is_test")

        return train, test

    def _core_split(self, interactions: AnyDataFrame) -> SplitterReturnType:
        split_method = self._core_split_spark
        if isinstance(interactions, PandasDataFrame):
            split_method = self._core_split_pandas

        return split_method(interactions, self.test_size)
