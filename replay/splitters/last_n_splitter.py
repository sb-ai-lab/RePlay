from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, PolarsDataFrame, SparkDataFrame

from .base_splitter import Splitter

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf
    from pyspark.sql import Window

StrategyName = Literal["interactions", "timedelta"]


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
        "time_column_format",
        "strategy",
        "drop_cold_users",
        "drop_cold_items",
        "query_column",
        "item_column",
        "timestamp_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    def __init__(
        self,
        N: int,  # noqa: N803
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
            msg = "strategy must be equal 'interactions' or 'timedelta'"
            raise ValueError(msg)
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
        self.time_column_format = None
        if self.strategy == "timedelta":
            self.time_column_format = time_column_format

    def _add_time_partition(self, interactions: DataFrameLike) -> DataFrameLike:
        if isinstance(interactions, SparkDataFrame):
            return self._add_time_partition_to_spark(interactions)
        if isinstance(interactions, PandasDataFrame):
            return self._add_time_partition_to_pandas(interactions)
        if isinstance(interactions, PolarsDataFrame):
            return self._add_time_partition_to_polars(interactions)

        msg = f"{self} is not implemented for {type(interactions)}"
        raise NotImplementedError(msg)

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

    def _add_time_partition_to_polars(self, interactions: PolarsDataFrame) -> PolarsDataFrame:
        res = interactions.sort(self.timestamp_column).with_columns(
            pl.col(self.divide_column).cum_count().over(pl.col(self.divide_column)).alias("row_num")
        )

        return res

    def _to_unix_timestamp(self, interactions: DataFrameLike) -> DataFrameLike:
        if isinstance(interactions, SparkDataFrame):
            return self._to_unix_timestamp_spark(interactions)
        if isinstance(interactions, PandasDataFrame):
            return self._to_unix_timestamp_pandas(interactions)
        if isinstance(interactions, PolarsDataFrame):
            return self._to_unix_timestamp_polars(interactions)

        msg = f"{self} is not implemented for {type(interactions)}"
        raise NotImplementedError(msg)

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
                self.timestamp_column, sf.unix_timestamp(self.timestamp_column, self.time_column_format)
            )

        return interactions

    def _to_unix_timestamp_polars(self, interactions: PolarsDataFrame) -> PolarsDataFrame:
        time_column_type = interactions.dtypes[interactions.get_column_index(self.timestamp_column)]
        if isinstance(time_column_type, pl.Datetime):
            interactions = interactions.with_columns(pl.col(self.timestamp_column).dt.epoch("s"))

        return interactions

    def _partial_split_interactions(self, interactions: DataFrameLike, n: int) -> Tuple[DataFrameLike, DataFrameLike]:
        res = self._add_time_partition(interactions)
        if isinstance(interactions, SparkDataFrame):
            return self._partial_split_interactions_spark(res, n)
        if isinstance(interactions, PandasDataFrame):
            return self._partial_split_interactions_pandas(res, n)
        return self._partial_split_interactions_polars(res, n)

    def _partial_split_interactions_pandas(
        self, interactions: PandasDataFrame, n: int
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        interactions["count"] = interactions.groupby(self.divide_column, sort=False)[self.divide_column].transform(len)
        interactions["is_test"] = interactions["row_num"] > (interactions["count"] - float(n))
        if self.session_id_column:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions[~interactions["is_test"]].drop(columns=["row_num", "count", "is_test"])
        test = interactions[interactions["is_test"]].drop(columns=["row_num", "count", "is_test"])

        return train, test

    def _partial_split_interactions_spark(
        self, interactions: SparkDataFrame, n: int
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        interactions = interactions.withColumn(
            "count", sf.count(self.timestamp_column).over(Window.partitionBy(self.divide_column))
        )
        # float(n) - because DataFrame.filter is changing order
        # of sorted DataFrame to descending
        interactions = interactions.withColumn("is_test", sf.col("row_num") > sf.col("count") - sf.lit(float(n)))
        if self.session_id_column:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions.filter("is_test == 0").drop("row_num", "count", "is_test")
        test = interactions.filter("is_test").drop("row_num", "count", "is_test")

        return train, test

    def _partial_split_interactions_polars(
        self, interactions: PolarsDataFrame, n: int
    ) -> Tuple[PolarsDataFrame, PolarsDataFrame]:
        interactions = interactions.with_columns(
            pl.col(self.timestamp_column).count().over(self.divide_column).alias("count")
        )
        interactions = interactions.with_columns((pl.col("row_num") > (pl.col("count") - n)).alias("is_test"))
        if self.session_id_column:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions.filter(~pl.col("is_test")).drop("row_num", "count", "is_test")
        test = interactions.filter(pl.col("is_test")).drop("row_num", "count", "is_test")

        return train, test

    def _partial_split_timedelta(
        self, interactions: DataFrameLike, timedelta: int
    ) -> Tuple[DataFrameLike, DataFrameLike]:
        if isinstance(interactions, SparkDataFrame):
            return self._partial_split_timedelta_spark(interactions, timedelta)
        if isinstance(interactions, PandasDataFrame):
            return self._partial_split_timedelta_pandas(interactions, timedelta)
        return self._partial_split_timedelta_polars(interactions, timedelta)

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

    def _partial_split_timedelta_polars(
        self, interactions: PolarsDataFrame, timedelta: int
    ) -> Tuple[PolarsDataFrame, PolarsDataFrame]:
        res = interactions.with_columns(
            (pl.col(self.timestamp_column).max().over(self.divide_column) - pl.col(self.timestamp_column)).alias(
                "diff_timestamp"
            )
        ).with_columns((pl.col("diff_timestamp") < timedelta).alias("is_test"))

        if self.session_id_column:
            res = self._recalculate_with_session_id_column(res)

        train = res.filter(~pl.col("is_test")).drop("diff_timestamp", "is_test")
        test = res.filter(pl.col("is_test")).drop("diff_timestamp", "is_test")

        return train, test

    def _core_split(self, interactions: DataFrameLike) -> List[DataFrameLike]:
        if self.strategy == "timedelta":
            interactions = self._to_unix_timestamp(interactions)
        train, test = getattr(self, "_partial_split_" + self.strategy)(interactions, self.N)

        return train, test
