from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from pandas import DataFrame as PandasDataFrame
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame as SparkDataFrame, Window

from replay.data import AnyDataFrame
from replay.splitters.base_splitter import Splitter


class LastNSplitter(Splitter):
    """
    Split interactions by last N interactions/seconds per user.
    Type of splitting depends on the ``strategy`` parameter.

    >>> dataset
        user_id  item_id  timestamp
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
    >>> train, test = LastNSplitter(N=[2], time_column_format="yyyy-MM-dd").split(dataset)
    >>> train
        user_id  item_id  timestamp
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
        user_id  item_id  timestamp
    3         1        4 2020-01-04
    4         1        5 2020-01-05
    8         2        9 2020-01-09
    9         2       10 2020-01-10
    13        3        1 2020-01-04
    14        3        2 2020-01-05
    <BLANKLINE>
    """

    # pylint: disable=invalid-name
    def __init__(
        self,
        N: List[int],
        divide_column: str = "user_id",
        time_column_format: str = "yyyy-MM-dd HH:mm:ss",
        strategy: str = "interactions",
        drop_cold_users: bool = False,
        drop_cold_items: bool = False,
        drop_zero_rel_in_test: bool = False,
        user_col: str = "user_id",
        item_col: str = "item_id",
        timestamp_col: str = "timestamp",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        Args:
            N (array of int): Array of interactions/seconds to split.
            divide_column (str): Name of column for dividing
                in dataframe, default: ``user_id``.
            time_column_format (str): Format of time_column,
                needs for convert time_column into unix_timestamp type.
                If strategy is set to 'interactions', then you can omit this parameter.
                If time_column has already transformed into unix_timestamp type,
                then you can omit this parameter.
                default: ``yyyy-MM-dd HH:mm:ss``
            strategy (str): Defines the type of data splitting.
                Must be ``interactions`` or ``seconds``.
                default: ``interactions``.
            drop_cold_users (bool): Drop users from test DataFrame
                which are not in train DataFrame, default: False.
            drop_cold_items (bool): Drop items from test DataFrame
                which are not in train DataFrame, default: False.
            drop_zero_rel_in_test (bool): Flag to remove entries with relevance <= 0
                from the test part of the dataset.
                Default: ``False``.
            user_col (str): Name of user interaction column.
                If ``drop_cold_users`` is ``False``, then you can omit this parameter.
                Default: ``user_id``.
            item_col (str): Name of item interaction column.
                If ``drop_cold_items`` is ``False``, then you can omit this parameter.
                Default: ``item_id``.
            timestamp_col (str): Name of time column,
                default: ``timestamp``.
            session_id_column (str, optional): Name of session id column, which values can not be split,
                default: ``None``.
            session_id_processing_strategy (str): strategy of processing session if it is split,
                Values: ``train, test``, train: whole split session goes to train. test: same but to test.
                default: ``test``.
        """
        if strategy not in ["interactions", "seconds"]:
            raise ValueError("strategy must be equal 'interactions' or 'seconds'")
        super().__init__(
            drop_cold_users=drop_cold_users,
            drop_cold_items=drop_cold_items,
            drop_zero_rel_in_test=drop_zero_rel_in_test,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            session_id_col=session_id_col,
            session_id_processing_strategy=session_id_processing_strategy,
        )
        self.N = list(reversed(N))
        self.strategy = strategy
        self.divide_column = divide_column
        if self.strategy == "seconds":
            self.timestamp_col_format = time_column_format

    def _get_order_of_sort(self) -> list:
        return [self.divide_column, self.timestamp_col]

    def _add_time_partition(self, interactions: AnyDataFrame) -> AnyDataFrame:
        if isinstance(interactions, SparkDataFrame):
            return self._add_time_partition_to_spark(interactions)

        return self._add_time_partition_to_pandas(interactions)

    def _add_time_partition_to_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        res = interactions.copy(deep=True)
        res.sort_values(by=[self.divide_column, self.timestamp_col], inplace=True)
        res["row_num"] = res.groupby(self.divide_column, sort=False).cumcount() + 1

        return res

    def _add_time_partition_to_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        res = interactions.withColumn(
            "row_num",
            sf.row_number().over(Window.partitionBy(self.divide_column).orderBy(sf.col(self.timestamp_col))),
        )

        return res

    def _to_unix_timestamp(self, interactions: AnyDataFrame) -> AnyDataFrame:
        if isinstance(interactions, SparkDataFrame):
            return self._to_unix_timestamp_spark(interactions)

        return self._to_unix_timestamp_pandas(interactions)

    def _to_unix_timestamp_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        time_column_type = dict(interactions.dtypes)[self.timestamp_col]
        if time_column_type == np.dtype("datetime64[ns]"):
            interactions = interactions.copy(deep=True)
            interactions[self.timestamp_col] = (
                interactions[self.timestamp_col] - pd.Timestamp("1970-01-01")
            ) // pd.Timedelta("1s")

        return interactions

    def _to_unix_timestamp_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        time_column_type = dict(interactions.dtypes)[self.timestamp_col]
        if time_column_type == "date":
            interactions = interactions.withColumn(
                self.timestamp_col, sf.unix_timestamp(self.timestamp_col, self.timestamp_col_format)
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
        if self.session_id_col:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions[~interactions["is_test"]].drop(columns=["row_num", "count", "is_test"])
        test = interactions[interactions["is_test"]].drop(columns=["row_num", "count", "is_test"])

        return train, test

    def _partial_split_interactions_spark(
        self, interactions: SparkDataFrame, N: int
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        interactions = interactions.withColumn(
            "count", sf.count(self.timestamp_col).over(Window.partitionBy(self.divide_column))
        )
        # float(n) - because DataFrame.filter is changing order
        # of sorted DataFrame to descending
        interactions = interactions.withColumn("is_test", sf.col("row_num") > sf.col("count") - sf.lit(float(N)))
        if self.session_id_col:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions.filter("is_test == 0").drop("row_num", "count", "is_test")
        test = interactions.filter("is_test").drop("row_num", "count", "is_test")

        return train, test

    def _partial_split_seconds(self, interactions: AnyDataFrame, seconds: int) -> Tuple[AnyDataFrame, AnyDataFrame]:
        if isinstance(interactions, SparkDataFrame):
            return self._partial_split_seconds_spark(interactions, seconds)

        return self._partial_split_seconds_pandas(interactions, seconds)

    def _partial_split_seconds_pandas(
        self, interactions: PandasDataFrame, seconds: int
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        res = interactions.copy(deep=True)
        res["diff_timestamp"] = res.groupby(self.divide_column)[self.timestamp_col].transform(max) - res[self.timestamp_col]
        res["is_test"] = res["diff_timestamp"] < seconds
        if self.session_id_col:
            res = self._recalculate_with_session_id_column(res)

        train = res[~res["is_test"]].drop(columns=["diff_timestamp", "is_test"])
        test = res[res["is_test"]].drop(columns=["diff_timestamp", "is_test"])

        return train, test

    def _partial_split_seconds_spark(
        self, interactions: SparkDataFrame, seconds: int
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        inter_with_max_time = interactions.withColumn(
            "max_timestamp", sf.max(self.timestamp_col).over(Window.partitionBy(self.divide_column))
        )
        inter_with_diff = inter_with_max_time.withColumn(
            "diff_timestamp", sf.col("max_timestamp") - sf.col(self.timestamp_col)
        )
        # drop unnecessary column
        inter_with_diff = inter_with_diff.drop("max_timestamp")

        res = inter_with_diff.withColumn("is_test", sf.col("diff_timestamp") < sf.lit(seconds))
        if self.session_id_col:
            res = self._recalculate_with_session_id_column(res)

        train = res.filter("is_test == 0").drop("diff_timestamp", "is_test")
        test = res.filter("is_test").drop("diff_timestamp", "is_test")

        return train, test

    def _core_split(self, interactions: AnyDataFrame) -> List[AnyDataFrame]:
        if self.strategy == "seconds":
            interactions = self._to_unix_timestamp(interactions)
        train, test = getattr(self, "_partial_split_" + self.strategy)(interactions, self.N[0])
        res = []
        for r in self.N[1:]:
            train, train1 = getattr(self, "_partial_split_" + self.strategy)(train, r)
            res.append(train1)
        res.append(train)
        res = res[::-1] + [test]
        return res
