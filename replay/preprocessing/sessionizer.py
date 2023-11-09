from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame as PandasDataFrame
import pyspark.sql.functions as sf
from pyspark.sql.window import Window
from pyspark.sql import DataFrame as SparkDataFrame

from replay.data import AnyDataFrame


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class Sessionizer:
    """
    Create and filter sessions from given interactions.
    Session ids are formed as subtraction between unique
    users cumulative sum and number of entries inside each
    user history that are greater than session gap.

    >>> import pandas as pd
    >>> time_interactions = pd.DataFrame({
    ...    "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    ...    "item_id": [3, 7, 10, 5, 8, 11, 4, 9, 2, 5],
    ...    "timestamp": [1, 2, 3, 3, 2, 1, 3, 12, 1, 4]
    ... })
    >>> time_interactions
       user_id  item_id  timestamp
    0        1        3          1
    1        1        7          2
    2        1       10          3
    3        2        5          3
    4        2        8          2
    5        2       11          1
    6        3        4          3
    7        3        9         12
    8        3        2          1
    9        3        5          4
    >>> Sessionizer(session_gap=5).transform(time_interactions)
       user_id  item_id  timestamp  session_id
    0        1        3          1           2
    1        1        7          2           2
    2        1       10          3           2
    3        2        5          3           5
    4        2        8          2           5
    5        2       11          1           5
    6        3        4          3           9
    7        3        9         12           8
    8        3        2          1           9
    9        3        5          4           9
    <BLANKLINE>
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        user_column: str = "user_id",
        time_column: str = "timestamp",
        session_column: str = "session_id",
        session_gap: int = 86400,
        time_column_format: str = "yyyy-MM-dd HH:mm:ss",
        min_inter_per_session: Optional[int] = None,
        max_inter_per_session: Optional[int] = None,
        min_sessions_per_user: Optional[int] = None,
        max_sessions_per_user: Optional[int] = None,
    ):
        """
        :param user_column: Name of user interaction column,
            default: ``user_id``.
        :param time_column: Name of time column,
            default: ``timestamp``.
        :param session_column: Name of column with resulting sessions,
            default: ``session_id``.
        :param session_gap: The interval between interactions,
            beyond which the division into different sessions takes place.
            If time_column has `date` type, then it will be converted
            in `unix_timestamp` type using parameter `time_column_format`.
            default: 86400.
        :param time_column_format: Format of time_column,
            needs for convert `time_column` into `unix_timestamp` type.
            If `time_column` has already transformed into unix_timestamp type,
            then you can omit this parameter.
            default: ``yyyy-MM-dd HH:mm:ss``
        :param min_inter_per_session: Minimum number of interactions per session.
            Sessions with less number of interactions are ignored.
            If None, filter doesn't apply,
            Default: ``None``.
        :param max_inter_per_session: Maximum number of interactions per session.
            Sessions with greater number of interactions are ignored.
            If None, filter doesn't apply. Must be less than `min_inter_per_session`.
            Default: ``None``.
        :param min_sessions_per_user: Minimum number of sessions per user.
            Users with less number of sessions are ignored.
            If None, filter doesn't apply.
            Default: ``None``.
        :param max_sessions_per_user: Maximum number of sessions per user.
            Users with greater number of sessions are ignored.
            If None, filter doesn't apply. Must be less
            than `min_sessions_per_user`. Default: ``None``.
        """
        self.user_column = user_column
        self.time_column = time_column
        self.session_column = session_column
        self.session_gap = session_gap
        self.time_column_format = time_column_format
        self.min_inter_per_session = min_inter_per_session
        self.max_inter_per_session = max_inter_per_session
        self.min_sessions_per_user = min_sessions_per_user
        self.max_sessions_per_user = max_sessions_per_user
        self._sanity_check()

    def _sanity_check(self) -> None:
        if self.min_inter_per_session:
            assert self.min_inter_per_session > 0
        if self.min_sessions_per_user:
            assert self.min_sessions_per_user > 0
        if self.min_inter_per_session and self.max_inter_per_session:
            assert self.min_inter_per_session <= self.max_inter_per_session
        if self.min_sessions_per_user and self.max_sessions_per_user:
            assert self.min_sessions_per_user <= self.max_sessions_per_user

    def _to_unix_timestamp(self, interactions: AnyDataFrame) -> AnyDataFrame:
        if isinstance(interactions, SparkDataFrame):
            return self._to_unix_timestamp_spark(interactions)

        return self._to_unix_timestamp_pandas(interactions)

    def _to_unix_timestamp_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        time_column_type = dict(interactions.dtypes)[self.time_column]
        if time_column_type == np.dtype("datetime64[ns]"):
            interactions = interactions.copy(deep=True)
            interactions[self.time_column] = (
                interactions[self.time_column] - pd.Timestamp("1970-01-01")
            ) // pd.Timedelta("1s")

        return interactions

    def _to_unix_timestamp_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        time_column_type = dict(interactions.dtypes)[self.time_column]
        if time_column_type == "date":
            interactions = interactions.withColumn(
                self.time_column, sf.unix_timestamp(self.time_column, self.time_column_format)
            )

        return interactions

    def _create_sessions(self, data: AnyDataFrame) -> AnyDataFrame:
        if isinstance(data, SparkDataFrame):
            return self._create_sessions_spark(data)

        return self._create_sessions_pandas(data)

    def _create_sessions_pandas(self, data: PandasDataFrame) -> PandasDataFrame:
        res = data.copy(deep=True)

        diff = res[self.time_column] - res.sort_values([self.user_column, self.time_column]).groupby(self.user_column)[
            self.time_column
        ].shift(1)
        nan_mask = diff.isna()
        diff = diff >= self.session_gap
        diff[nan_mask] = True
        res["timestamp_diff"] = diff
        res["cumsum_timestamp_diff"] = (
            res.sort_values([self.user_column, self.time_column, "timestamp_diff"], ascending=[True, True, False])
            .groupby(self.user_column, sort=False)["timestamp_diff"]
            .cumsum()
        )

        user_count = res.groupby(self.user_column)[self.user_column].count().cumsum().to_frame()
        user_count.rename(columns={self.user_column: "count"}, inplace=True)
        res = res.join(user_count, how="left", on=self.user_column)
        res[self.session_column] = res["count"] - res["cumsum_timestamp_diff"]

        res.drop(columns=["timestamp_diff", "cumsum_timestamp_diff", "count"], inplace=True)

        return res

    def _create_sessions_spark(self, data: SparkDataFrame) -> SparkDataFrame:
        data_with_diff = data.withColumn(
            "timestamp_diff",
            sf.col(self.time_column)
            - sf.lag(self.time_column, 1).over(Window.partitionBy(self.user_column).orderBy(self.time_column))
            >= self.session_gap,
        )
        data_with_diff = data_with_diff.na.fill(True).withColumn(
            "timestamp_diff", sf.col("timestamp_diff").cast("long")
        )
        data_with_sum_timediff = data_with_diff.withColumn(
            "cumsum_timestamp_diff",
            sf.sum("timestamp_diff").over(
                Window.partitionBy(self.user_column).orderBy(sf.col(self.time_column), sf.col("timestamp_diff").desc())
            ),
        )
        # data_with_sum_timediff.cache()

        grouped_users = data_with_sum_timediff.groupBy(self.user_column).count()
        grouped_users_with_cumsum = grouped_users.withColumn(
            "cumsum_user_count",
            sf.sum("count").over(Window.partitionBy(sf.lit(0)).orderBy(self.user_column)),
        ).drop("count")

        result = (
            data_with_sum_timediff.join(grouped_users_with_cumsum, self.user_column, "left")
            .withColumn(
                self.session_column,
                sf.col("cumsum_user_count") - sf.col("cumsum_timestamp_diff"),
            )
            .drop(
                "timestamp_diff",
                "cumsum_timestamp_diff",
                "cumsum_user_count",
            )
        )

        # data_with_sum_timediff.unpersist()
        return result

    def _filter_sessions(self, interactions: AnyDataFrame) -> AnyDataFrame:
        # interactions.cache()
        if isinstance(interactions, SparkDataFrame):
            return self._filter_sessions_spark(interactions)

        return self._filter_sessions_pandas(interactions)

    def _filter_sessions_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        interactions["count"] = interactions.groupby(self.session_column, sort=False)[self.session_column].transform(
            "count"
        )
        if self.min_inter_per_session:
            interactions = interactions[interactions["count"] >= self.min_inter_per_session]
        if self.max_inter_per_session:
            interactions = interactions[interactions["count"] <= self.max_inter_per_session]

        interactions["nunique"] = interactions.groupby(self.user_column, sort=False)[self.session_column].transform(
            "nunique"
        )
        if self.min_sessions_per_user:
            interactions = interactions[interactions["nunique"] >= self.min_sessions_per_user]
        if self.max_sessions_per_user:
            interactions = interactions[interactions["nunique"] <= self.max_sessions_per_user]

        interactions.drop(columns=["count", "nunique"], inplace=True)

        return interactions

    def _filter_sessions_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        entries_counter = interactions.groupby(self.session_column).count()
        if self.min_inter_per_session:
            entries_counter = entries_counter.filter(sf.col("count") >= self.min_inter_per_session)
        if self.max_inter_per_session:
            entries_counter = entries_counter.filter(sf.col("count") <= self.max_inter_per_session)

        filtered_interactions = interactions.join(
            entries_counter.select(self.session_column), self.session_column, how="right"
        )

        # filtered_interactions.cache()

        nunique = filtered_interactions.groupby(self.user_column).agg(
            sf.expr("count(distinct session_id)").alias("nunique")
        )
        if self.min_sessions_per_user:
            nunique = nunique.filter(sf.col("nunique") >= self.min_sessions_per_user)
        if self.max_sessions_per_user:
            nunique = nunique.filter(sf.col("nunique") <= self.max_sessions_per_user)

        result = filtered_interactions.join(
            nunique.select(self.user_column),
            self.user_column,
            how="right",
        )
        return result

    def transform(self, interactions: AnyDataFrame) -> AnyDataFrame:
        r"""Create and filter sessions from given interactions.

        :param interactions: DataFrame containing columns ``user_column``, ``time_column``.

        :returns: DataFrame with created and filtered sessions.
        """
        columns_order = list(interactions.columns)
        interactions = self._to_unix_timestamp(interactions)
        result = self._create_sessions(interactions)
        result = self._filter_sessions(result)
        columns_order += [self.session_column]

        if isinstance(result, SparkDataFrame):
            result = result.select(*columns_order)
        else:
            result = result[columns_order]

        return result
