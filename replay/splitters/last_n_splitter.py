from typing import List, Optional, Tuple, Literal

import pandas as pd
import numpy as np
from pandas import DataFrame as PandasDataFrame
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame as SparkDataFrame, Window

from replay.data import AnyDataFrame
from replay.splitters.base_splitter import Splitter


StrategyName = Literal["interactions", "timedelta"]


# pylint: disable=too-few-public-methods
class LastNSplitter(Splitter):
    """
    Split interactions by last N interactions/timedelta per user.
    Type of splitting depends on the ``strategy`` parameter.

    >>> from datetime import datetime
    >>> import pandas as pd
    >>> columns = ["query_id", "item_id", "timestamp"]
    >>> data = [
    ...     (1, 1, "01-01-2020"),
    ...     (1, 2, "02-01-2020"),
    ...     (1, 3, "03-01-2020"),
    ...     (1, 4, "04-01-2020"),
    ...     (1, 5, "05-01-2020"),
    ...     (2, 1, "06-01-2020"),
    ...     (2, 2, "07-01-2020"),
    ...     (2, 3, "08-01-2020"),
    ...     (2, 9, "09-01-2020"),
    ...     (2, 10, "10-01-2020"),
    ...     (3, 1, "01-01-2020"),
    ...     (3, 5, "02-01-2020"),
    ...     (3, 3, "03-01-2020"),
    ...     (3, 1, "04-01-2020"),
    ...     (3, 2, "05-01-2020"),
    ... ]
    >>> dataset = pd.DataFrame(data, columns=columns)
    >>> dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], format="%d-%m-%Y")
    >>> dataset
       query_id  item_id  timestamp
    0         1        1 2020-01-01
    1         1        2 2020-01-02
    2         1        3 2020-01-03
    3         1        4 2020-01-04
    4         1        5 2020-01-05
    5         2        1 2020-01-06
    6         2        2 2020-01-07
    7         2        3 2020-01-08
    8         2        9 2020-01-09
    9         2       10 2020-01-10
    10        3        1 2020-01-01
    11        3        5 2020-01-02
    12        3        3 2020-01-03
    13        3        1 2020-01-04
    14        3        2 2020-01-05
    >>> splitter = LastNSplitter(
    ...     N=2,
    ...     divide_column="query_id",
    ...     time_column_format="yyyy-MM-dd",
    ...     query_column="query_id",
    ...     item_column="item_id"
    ... )
    >>> train, test = splitter.split(dataset)
    >>> train
       query_id  item_id  timestamp
    0         1        1 2020-01-01
    1         1        2 2020-01-02
    2         1        3 2020-01-03
    5         2        1 2020-01-06
    6         2        2 2020-01-07
    7         2        3 2020-01-08
    10        3        1 2020-01-01
    11        3        5 2020-01-02
    12        3        3 2020-01-03
    >>> test
       query_id  item_id  timestamp
    3         1        4 2020-01-04
    4         1        5 2020-01-05
    8         2        9 2020-01-09
    9         2       10 2020-01-10
    13        3        1 2020-01-04
    14        3        2 2020-01-05
    <BLANKLINE>
    """
    _init_arg_names = [
        "N",
        "divide_column",
        "timestamp_col_format",
        "strategy",
        "drop_cold_users",
        "drop_cold_items",
        "query_column",
        "item_column",
        "timestamp_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    # pylint: disable=invalid-name, too-many-arguments
    def __init__(
        self,
        N: int,
        divide_column: str = "query_id",
        time_column_format: str = "yyyy-MM-dd HH:mm:ss",
        strategy: StrategyName = "interactions",
        drop_cold_users: bool = False,
        drop_cold_items: bool = False,
        query_column: str = "query_id",
        item_column: str = "item_id",
        timestamp_column: str = "timestamp",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param N: Array of interactions/timedelta to split.
        :param divide_column: Name of column for dividing
            in dataframe, default: ``query_id``.
        :param time_column_format: Format of time_column,
            needs for convert time_column into unix_timestamp type.
            If strategy is set to 'interactions', then you can omit this parameter.
            If time_column has already transformed into unix_timestamp type,
            then you can omit this parameter.
            default: ``yyyy-MM-dd HH:mm:ss``
        :param strategy: Defines the type of data splitting.
            Must be ``interactions`` or ``timedelta``.
            default: ``interactions``.
        :param query_column: Name of query interaction column.
        :param drop_cold_users: Drop users from test DataFrame.
            which are not in train DataFrame, default: False.
        :param drop_cold_items: Drop items from test DataFrame
            which are not in train DataFrame, default: False.
        :param item_column: Name of item interaction column.
            If ``drop_cold_items`` is ``False``, then you can omit this parameter.
            Default: ``item_id``.
        :param timestamp_column: Name of time column,
            Default: ``timestamp``.
        :param session_id_column: Name of session id column, which values can not be split,
            default: ``None``.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        if strategy not in ["interactions", "timedelta"]:
            raise ValueError("strategy must be equal 'interactions' or 'timedelta'")
        super().__init__(
            drop_cold_users=drop_cold_users,
            drop_cold_items=drop_cold_items,
            query_column=query_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            session_id_column=session_id_column,
            session_id_processing_strategy=session_id_processing_strategy,
        )
        self.N = N
        self.strategy = strategy
        self.divide_column = divide_column
        self.timestamp_col_format = None
        if self.strategy == "timedelta":
            self.timestamp_col_format = time_column_format

    def _add_time_partition(self, interactions: AnyDataFrame) -> AnyDataFrame:
        if isinstance(interactions, SparkDataFrame):
            return self._add_time_partition_to_spark(interactions)

        return self._add_time_partition_to_pandas(interactions)

    def _add_time_partition_to_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        res = interactions.copy(deep=True)
        res.sort_values(by=[self.divide_column, self.timestamp_column], inplace=True)
        res["row_num"] = res.groupby(self.divide_column, sort=False).cumcount() + 1

        return res

    def _add_time_partition_to_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        res = interactions.withColumn(
            "row_num",
            sf.row_number().over(Window.partitionBy(self.divide_column).orderBy(sf.col(self.timestamp_column))),
        )

        return res

    def _to_unix_timestamp(self, interactions: AnyDataFrame) -> AnyDataFrame:
        if isinstance(interactions, SparkDataFrame):
            return self._to_unix_timestamp_spark(interactions)

        return self._to_unix_timestamp_pandas(interactions)

    def _to_unix_timestamp_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        time_column_type = dict(interactions.dtypes)[self.timestamp_column]
        if time_column_type == np.dtype("datetime64[ns]"):
            interactions = interactions.copy(deep=True)
            interactions[self.timestamp_column] = (
                interactions[self.timestamp_column] - pd.Timestamp("1970-01-01")
            ) // pd.Timedelta("1s")

        return interactions

    def _to_unix_timestamp_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        time_column_type = dict(interactions.dtypes)[self.timestamp_column]
        if time_column_type == "date":
            interactions = interactions.withColumn(
                self.timestamp_column, sf.unix_timestamp(self.timestamp_column, self.timestamp_col_format)
            )

        return interactions

    # pylint: disable=invalid-name
    def _partial_split_interactions(self, interactions: AnyDataFrame, N: int) -> Tuple[AnyDataFrame, AnyDataFrame]:
        res = self._add_time_partition(interactions)
        if isinstance(interactions, SparkDataFrame):
            return self._partial_split_interactions_spark(res, N)

        return self._partial_split_interactions_pandas(res, N)

    def _partial_split_interactions_pandas(
        self, interactions: PandasDataFrame, N: int
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        interactions["count"] = interactions.groupby(self.divide_column, sort=False)[self.divide_column].transform(len)
        interactions["is_test"] = interactions["row_num"] > (interactions["count"] - float(N))
        if self.session_id_column:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions[~interactions["is_test"]].drop(columns=["row_num", "count", "is_test"])
        test = interactions[interactions["is_test"]].drop(columns=["row_num", "count", "is_test"])

        return train, test

    def _partial_split_interactions_spark(
        self, interactions: SparkDataFrame, N: int
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        interactions = interactions.withColumn(
            "count", sf.count(self.timestamp_column).over(Window.partitionBy(self.divide_column))
        )
        # float(n) - because DataFrame.filter is changing order
        # of sorted DataFrame to descending
        interactions = interactions.withColumn("is_test", sf.col("row_num") > sf.col("count") - sf.lit(float(N)))
        if self.session_id_column:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions.filter("is_test == 0").drop("row_num", "count", "is_test")
        test = interactions.filter("is_test").drop("row_num", "count", "is_test")

        return train, test

    def _partial_split_timedelta(self, interactions: AnyDataFrame, timedelta: int) -> Tuple[AnyDataFrame, AnyDataFrame]:
        if isinstance(interactions, SparkDataFrame):
            return self._partial_split_timedelta_spark(interactions, timedelta)

        return self._partial_split_timedelta_pandas(interactions, timedelta)

    def _partial_split_timedelta_pandas(
        self, interactions: PandasDataFrame, timedelta: int
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        res = interactions.copy(deep=True)
        res["diff_timestamp"] = (
            res.groupby(self.divide_column)[self.timestamp_column].transform(max) - res[self.timestamp_column]
        )
        res["is_test"] = res["diff_timestamp"] < timedelta
        if self.session_id_column:
            res = self._recalculate_with_session_id_column(res)

        train = res[~res["is_test"]].drop(columns=["diff_timestamp", "is_test"])
        test = res[res["is_test"]].drop(columns=["diff_timestamp", "is_test"])

        return train, test

    def _partial_split_timedelta_spark(
        self, interactions: SparkDataFrame, timedelta: int
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        inter_with_max_time = interactions.withColumn(
            "max_timestamp", sf.max(self.timestamp_column).over(Window.partitionBy(self.divide_column))
        )
        inter_with_diff = inter_with_max_time.withColumn(
            "diff_timestamp", sf.col("max_timestamp") - sf.col(self.timestamp_column)
        )
        # drop unnecessary column
        inter_with_diff = inter_with_diff.drop("max_timestamp")

        res = inter_with_diff.withColumn("is_test", sf.col("diff_timestamp") < sf.lit(timedelta))
        if self.session_id_column:
            res = self._recalculate_with_session_id_column(res)

        train = res.filter("is_test == 0").drop("diff_timestamp", "is_test")
        test = res.filter("is_test").drop("diff_timestamp", "is_test")

        return train, test

    def _core_split(self, interactions: AnyDataFrame) -> List[AnyDataFrame]:
        if self.strategy == "timedelta":
            interactions = self._to_unix_timestamp(interactions)
        train, test = getattr(self, "_partial_split_" + self.strategy)(interactions, self.N)

        return train, test
