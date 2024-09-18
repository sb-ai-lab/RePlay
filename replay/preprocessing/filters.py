"""
Select or remove data by some criteria
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Callable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, PolarsDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    from pyspark.sql import (
        Window,
        functions as sf,
    )
    from pyspark.sql.functions import col
    from pyspark.sql.types import TimestampType


class _BaseFilter(ABC):
    def transform(self, interactions: DataFrameLike) -> DataFrameLike:
        r"""Filter interactions.

        :param interactions: DataFrame with interactions.

        :returns: filtered DataFrame.
        """
        if isinstance(interactions, SparkDataFrame):
            return self._filter_spark(interactions)
        elif isinstance(interactions, PandasDataFrame):
            return self._filter_pandas(interactions)
        elif isinstance(interactions, PolarsDataFrame):
            return self._filter_polars(interactions)
        else:
            msg = f"{self.__class__.__name__} is not implemented for {type(interactions)}"
            raise NotImplementedError(msg)

    @abstractmethod
    def _filter_spark(self, interactions: SparkDataFrame):  # pragma: no cover
        pass

    @abstractmethod
    def _filter_pandas(self, interactions: PandasDataFrame):  # pragma: no cover
        pass

    @abstractmethod
    def _filter_polars(self, interactions: PolarsDataFrame):  # pragma: no cover
        pass


class InteractionEntriesFilter(_BaseFilter):
    """
    Remove interactions less than minimum constraint value and greater
    than maximum constraint value for each column.

    >>> import pandas as pd
    >>> interactions = pd.DataFrame({
    ...    "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    ...    "item_id": [3, 7, 10, 5, 8, 11, 4, 9, 2, 5],
    ...    "rating": [1, 2, 3, 3, 2, 1, 3, 12, 1, 4]
    ... })
    >>> interactions
        user_id  item_id  rating
    0        1        3       1
    1        1        7       2
    2        1       10       3
    3        2        5       3
    4        2        8       2
    5        2       11       1
    6        3        4       3
    7        3        9      12
    8        3        2       1
    9        3        5       4
    >>> filtered_interactions = InteractionEntriesFilter(min_inter_per_user=4).transform(interactions)
    >>> filtered_interactions
        user_id  item_id  rating
    6        3        4       3
    7        3        9      12
    8        3        2       1
    9        3        5       4
    <BLANKLINE>
    """

    def __init__(
        self,
        query_column: str = "user_id",
        item_column: str = "item_id",
        min_inter_per_user: Optional[int] = None,
        max_inter_per_user: Optional[int] = None,
        min_inter_per_item: Optional[int] = None,
        max_inter_per_item: Optional[int] = None,
        allow_caching: bool = True,
    ):
        r"""
        :param user_column: Name of user interaction column,
            default: ``user_id``.
        :param item_column: Name of item interaction column,
            default: ``item_id``.
        :param min_inter_per_user: Minimum positive value of
            interactions per user. If None, filter doesn't apply,
            default: ``None``.
        :param max_inter_per_user: Maximum positive value of
            interactions per user. If None, filter doesn't apply. Must be
            less than `min_inter_per_user`, default: ``None``.
        :param min_inter_per_item: Minimum positive value of
            interactions per item. If None, filter doesn't apply,
            default: ``None``.
        :param max_inter_per_item: Maximum positive value of
            interactions per item. If None, filter doesn't apply. Must be
            less than `min_inter_per_item`, default: ``None``.
        :param allow_caching: The flag for using caching to optimize calculations.
            default: `True`.
        """
        self.query_column = query_column
        self.item_column = item_column
        self.min_inter_per_user = min_inter_per_user
        self.max_inter_per_user = max_inter_per_user
        self.min_inter_per_item = min_inter_per_item
        self.max_inter_per_item = max_inter_per_item
        self.total_dropped_interactions = 0
        self.allow_caching = allow_caching
        self._sanity_check()

    def _sanity_check(self) -> None:
        if self.min_inter_per_user:
            assert self.min_inter_per_user > 0
        if self.min_inter_per_item:
            assert self.min_inter_per_item > 0
        if self.min_inter_per_user and self.max_inter_per_user:
            assert self.min_inter_per_user < self.max_inter_per_user
        if self.min_inter_per_item and self.max_inter_per_item:
            assert self.min_inter_per_item < self.max_inter_per_item

    def _filter_spark(self, interactions: SparkDataFrame):
        interaction_count = interactions.count()
        return self._iterative_filter(interactions, interaction_count, self._filter_column_spark)

    def _filter_pandas(self, interactions: PandasDataFrame):
        interaction_count = len(interactions)
        return self._iterative_filter(interactions, interaction_count, self._filter_column_pandas)

    def _filter_polars(self, interactions: PolarsDataFrame):
        interaction_count = len(interactions)
        return self._iterative_filter(interactions, interaction_count, self._filter_column_polars)

    def _iterative_filter(self, interactions: DataFrameLike, interaction_count: int, filter_func: Callable):
        is_dropped_user_item = [True, True]
        current_index = 0
        while is_dropped_user_item[0] or is_dropped_user_item[1]:
            if current_index == 0:
                min_inter = self.min_inter_per_user
                max_inter = self.max_inter_per_user
                agg_column = self.query_column
                non_agg_column = self.item_column
            else:
                min_inter = self.min_inter_per_item
                max_inter = self.max_inter_per_item
                agg_column = self.item_column
                non_agg_column = self.query_column

            if min_inter is None and max_inter is None:
                dropped_interact = 0
            else:
                interactions, dropped_interact, interaction_count = filter_func(
                    interactions, interaction_count, agg_column, non_agg_column, min_inter, max_inter
                )
            is_dropped_user_item[current_index] = bool(dropped_interact)
            current_index = (current_index + 1) % 2  # current_index only in (0, 1)

        return interactions

    def _filter_column_pandas(
        self,
        interactions: PandasDataFrame,
        interaction_count: int,
        agg_column: str,
        non_agg_column: str,
        min_inter: Optional[int] = None,
        max_inter: Optional[int] = None,
    ) -> Tuple[PandasDataFrame, int, int]:
        filtered_interactions = interactions.copy(deep=True)

        filtered_interactions["count"] = filtered_interactions.groupby(agg_column, sort=False)[
            non_agg_column
        ].transform(len)
        if min_inter:
            filtered_interactions = filtered_interactions[filtered_interactions["count"] >= min_inter]
        if max_inter:
            filtered_interactions = filtered_interactions[filtered_interactions["count"] <= max_inter]
        filtered_interactions.drop(columns=["count"], inplace=True)

        end_len_dataframe = len(filtered_interactions)
        different_len = interaction_count - end_len_dataframe

        return filtered_interactions, different_len, end_len_dataframe

    def _filter_column_spark(
        self,
        interactions: SparkDataFrame,
        interaction_count: int,
        agg_column: str,
        non_agg_column: str,
        min_inter: Optional[int] = None,
        max_inter: Optional[int] = None,
    ) -> Tuple[SparkDataFrame, int, int]:
        filtered_interactions = interactions.withColumn(
            "count", sf.count(non_agg_column).over(Window.partitionBy(agg_column))
        )
        if min_inter:
            filtered_interactions = filtered_interactions.filter(sf.col("count") >= min_inter)
        if max_inter:
            filtered_interactions = filtered_interactions.filter(sf.col("count") <= max_inter)
        filtered_interactions = filtered_interactions.drop("count")

        if self.allow_caching:
            filtered_interactions.cache()
            interactions.unpersist()
        end_len_dataframe = filtered_interactions.count()
        different_len = interaction_count - end_len_dataframe

        return filtered_interactions, different_len, end_len_dataframe

    def _filter_column_polars(
        self,
        interactions: PolarsDataFrame,
        interaction_count: int,
        agg_column: str,
        non_agg_column: str,
        min_inter: Optional[int] = None,
        max_inter: Optional[int] = None,
    ) -> Tuple[PolarsDataFrame, int, int]:
        filtered_interactions = interactions.with_columns(
            pl.col(non_agg_column).count().over(pl.col(agg_column)).alias("count")
        )
        if min_inter:
            filtered_interactions = filtered_interactions.filter(pl.col("count") >= min_inter)
        if max_inter:
            filtered_interactions = filtered_interactions.filter(pl.col("count") <= max_inter)
        filtered_interactions = filtered_interactions.drop("count")

        end_len_dataframe = len(filtered_interactions)
        different_len = interaction_count - end_len_dataframe

        return filtered_interactions, different_len, end_len_dataframe


class MinCountFilter(_BaseFilter):
    """
    Remove entries with entities (e.g. users, items) which are presented in `interactions`
    less than `num_entries` times. The `interactions` is grouped by `groupby_column`,
    which is entry column name, to calculate counts.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_id": [1, 1, 2]})
    >>> MinCountFilter(2).transform(data_frame)
        user_id
    0         1
    1         1
    <BLANKLINE>
    """

    def __init__(
        self,
        num_entries: int,
        groupby_column: str = "user_id",
    ):
        r"""
        :param num_entries: minimal number of times the entry should appear in dataset
            in order to remain.
        :param group_by: entity column, which is used to calculate entity occurrence couns,
            default: ``user_id``.
        """
        self.num_entries = num_entries
        self.groupby_column = groupby_column
        self._sanity_check()

    def _sanity_check(self) -> None:
        assert self.num_entries > 0

    def _filter_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        count_by_group = interactions.groupBy(self.groupby_column).agg(
            sf.count(self.groupby_column).alias(f"{self.groupby_column}_temp_count")
        )
        remaining_entities = count_by_group.filter(
            count_by_group[f"{self.groupby_column}_temp_count"] >= self.num_entries
        ).select(self.groupby_column)

        return interactions.join(remaining_entities, on=self.groupby_column, how="inner")

    def _filter_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        filtered_interactions = interactions.copy(deep=True)
        filtered_interactions["count"] = filtered_interactions.groupby(self.groupby_column)[
            self.groupby_column
        ].transform(len)
        return filtered_interactions[filtered_interactions["count"] >= self.num_entries].drop(columns=["count"])

    def _filter_polars(self, interactions: PolarsDataFrame) -> PolarsDataFrame:
        filtered_interactions = interactions.clone()
        count_by_group = (
            filtered_interactions.group_by(self.groupby_column)
            .agg(pl.col(self.groupby_column).count().alias(f"{self.groupby_column}_temp_count"))
            .filter(pl.col(f"{self.groupby_column}_temp_count") >= self.num_entries)
        )
        return filtered_interactions.join(count_by_group, on=self.groupby_column).drop(
            f"{self.groupby_column}_temp_count"
        )


class LowRatingFilter(_BaseFilter):
    """
    Remove records with records less than ``value`` in ``column``.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"rating": [1, 5, 3.5, 4]})
    >>> LowRatingFilter(3.5).transform(data_frame)
         rating
    1       5.0
    2       3.5
    3       4.0
    <BLANKLINE>
    """

    def __init__(
        self,
        value: float,
        rating_column: str = "rating",
    ):
        r"""
        :param value: minimal value the entry should appear in dataset
            in order to remain.
        :param rating_column: the column in which filtering is performed.
        """
        self.value = value
        self.rating_column = rating_column

    def _filter_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        return interactions.filter(interactions[self.rating_column] >= self.value)

    def _filter_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        return interactions[interactions[self.rating_column] >= self.value]

    def _filter_polars(self, interactions: PolarsDataFrame) -> PolarsDataFrame:
        return interactions.filter(pl.col(self.rating_column) >= self.value)


class NumInteractionsFilter(_BaseFilter):
    """
    Get first/last ``num_interactions`` interactions for each query.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rating": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01 00:00:00",
    ...                                   "2020-02-01 00:00:01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"], format="ISO8601")
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u2|     i2|   0.5|2020-02-01 00:00:00|
    |     u2|     i3|   3.0|2020-02-01 00:00:01|
    |     u3|     i1|   1.0|2020-01-01 00:04:15|
    |     u3|     i2|   0.0|2020-01-02 00:04:14|
    |     u3|     i3|   1.0|2020-01-05 23:59:59|
    +-------+-------+------+-------------------+
    <BLANKLINE>

    Only first interaction:

    >>> NumInteractionsFilter(1, True, item_column='item_id').transform(log_sp).orderBy('user_id').show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u2|     i2|   0.5|2020-02-01 00:00:00|
    |     u3|     i1|   1.0|2020-01-01 00:04:15|
    +-------+-------+------+-------------------+
    <BLANKLINE>

    Only last interaction:

    >>> NumInteractionsFilter(1, False).transform(log_sp).orderBy('user_id').show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u2|     i3|   3.0|2020-02-01 00:00:01|
    |     u3|     i3|   1.0|2020-01-05 23:59:59|
    +-------+-------+------+-------------------+
    <BLANKLINE>

    >>> NumInteractionsFilter(1, False, item_column='item_id').transform(log_sp).orderBy('user_id').show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u2|     i3|   3.0|2020-02-01 00:00:01|
    |     u3|     i3|   1.0|2020-01-05 23:59:59|
    +-------+-------+------+-------------------+
    <BLANKLINE>
    """

    def __init__(
        self,
        num_interactions: int = 10,
        first: bool = True,
        query_column: str = "user_id",
        timestamp_column: str = "timestamp",
        item_column: Optional[str] = None,
    ):
        r"""
        :param num_interactions: number of interactions to leave per user.
        :param first: take either first ``num_interactions`` or last.
            default: `True`.
        :param query_column: query column.
            default: ``user_id``.
        :param timestamp_column: timestamp column.
            default: ``timestamp``.
        :param item_column: item column to help sort simultaneous interactions.
            If None, it is ignored.
            default: `None`.
        """
        self.num_interactions = num_interactions
        self.first = first
        self.query_column = query_column
        self.timestamp_column = timestamp_column
        self.item_column = item_column
        self.first = first
        self._sanity_check()

    def _sanity_check(self) -> None:
        assert self.num_interactions >= 0

    def _filter_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        sorting_order = [col(self.timestamp_column)]
        if self.item_column is not None:
            sorting_order.append(col(self.item_column))

        if not self.first:
            sorting_order = [col_.desc() for col_ in sorting_order]

        window = Window().orderBy(*sorting_order).partitionBy(col(self.query_column))

        return (
            interactions.withColumn("temp_rank", sf.row_number().over(window))
            .filter(col("temp_rank") <= self.num_interactions)
            .drop("temp_rank")
        )

    def _filter_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        filtered_interactions = interactions.copy(deep=True)
        sorting_columns = [self.timestamp_column]
        if self.item_column is not None:
            sorting_columns.append(self.item_column)

        ascending = [self.first] * len(sorting_columns)

        filtered_interactions["temp_rank"] = (
            filtered_interactions.sort_values(sorting_columns, ascending=ascending)
            .groupby(self.query_column)
            .cumcount()
        )
        return filtered_interactions[filtered_interactions["temp_rank"] < self.num_interactions].drop(
            columns=["temp_rank"]
        )

    def _filter_polars(self, interactions: PolarsDataFrame) -> PolarsDataFrame:
        sorting_columns = [self.timestamp_column]
        if self.item_column is not None:
            sorting_columns.append(self.item_column)

        descending = not self.first

        return (
            interactions.sort(sorting_columns, descending=descending)
            .with_columns(pl.col(self.query_column).cum_count().over(self.query_column).alias("temp_rank"))
            .filter(pl.col("temp_rank") <= self.num_interactions)
            .drop("temp_rank")
        )


class EntityDaysFilter(_BaseFilter):
    """
    Get first/last ``days`` of interactions by entity.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rating": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01 00:00:00",
    ...                                   "2020-02-01 00:00:01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"], format="ISO8601")
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.orderBy('user_id', 'item_id').show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u2|     i2|   0.5|2020-02-01 00:00:00|
    |     u2|     i3|   3.0|2020-02-01 00:00:01|
    |     u3|     i1|   1.0|2020-01-01 00:04:15|
    |     u3|     i2|   0.0|2020-01-02 00:04:14|
    |     u3|     i3|   1.0|2020-01-05 23:59:59|
    +-------+-------+------+-------------------+
    <BLANKLINE>

    Get first day by users:

    >>> EntityDaysFilter(1, True, entity_column='user_id').transform(log_sp).orderBy('user_id', 'item_id').show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u2|     i2|   0.5|2020-02-01 00:00:00|
    |     u2|     i3|   3.0|2020-02-01 00:00:01|
    |     u3|     i1|   1.0|2020-01-01 00:04:15|
    |     u3|     i2|   0.0|2020-01-02 00:04:14|
    +-------+-------+------+-------------------+
    <BLANKLINE>

    Get last day by item:

    >>> EntityDaysFilter(1, False, entity_column='item_id').transform(log_sp).orderBy('item_id', 'user_id').show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u3|     i1|   1.0|2020-01-01 00:04:15|
    |     u2|     i2|   0.5|2020-02-01 00:00:00|
    |     u2|     i3|   3.0|2020-02-01 00:00:01|
    +-------+-------+------+-------------------+
    <BLANKLINE>
    """

    def __init__(
        self,
        days: int = 10,
        first: bool = True,
        entity_column: str = "user_id",
        timestamp_column: str = "timestamp",
    ):
        r"""
        :param days: how many days to return per entity.
            default: `10`.
        :param first: take either first ``num_interactions`` or last.
            default: `True`.
        :param entity_column: query/item column.
            default: ``user_id``.
        :param timestamp_column: timestamp column.
            default: ``timestamp``.
        """
        self.days = days
        self.first = first
        self.entity_column = entity_column
        self.timestamp_column = timestamp_column
        self.first = first
        self._sanity_check()

    def _sanity_check(self) -> None:
        assert self.days > 0

    def _filter_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        window = Window.partitionBy(self.entity_column)
        if self.first:
            filtered_interactions = (
                interactions.withColumn("min_date", sf.min(col(self.timestamp_column)).over(window))
                .filter(col(self.timestamp_column) < col("min_date") + sf.expr(f"INTERVAL {self.days} days"))
                .drop("min_date")
            )
        else:
            filtered_interactions = (
                interactions.withColumn("max_date", sf.max(col(self.timestamp_column)).over(window))
                .filter(col(self.timestamp_column) > col("max_date") - sf.expr(f"INTERVAL {self.days} days"))
                .drop("max_date")
            )
        return filtered_interactions

    def _filter_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        filtered_interactions = interactions.copy(deep=True)

        if self.first:
            filtered_interactions["min_date"] = filtered_interactions.groupby(self.entity_column)[
                self.timestamp_column
            ].transform(min)
            return filtered_interactions[
                (filtered_interactions[self.timestamp_column] - filtered_interactions["min_date"]).dt.days < self.days
            ].drop(columns=["min_date"])
        filtered_interactions["max_date"] = filtered_interactions.groupby(self.entity_column)[
            self.timestamp_column
        ].transform(max)
        return filtered_interactions[
            (filtered_interactions["max_date"] - filtered_interactions[self.timestamp_column]).dt.days < self.days
        ].drop(columns=["max_date"])

    def _filter_polars(self, interactions: PolarsDataFrame) -> PolarsDataFrame:
        if self.first:
            return (
                interactions.with_columns(
                    (
                        pl.col(self.timestamp_column).min().over(pl.col(self.entity_column))
                        + pl.duration(days=self.days)
                    ).alias("min_date")
                )
                .filter(pl.col(self.timestamp_column) < pl.col("min_date"))
                .drop("min_date")
            )
        return (
            interactions.with_columns(
                (
                    pl.col(self.timestamp_column).max().over(pl.col(self.entity_column)) - pl.duration(days=self.days)
                ).alias("max_date")
            )
            .filter(pl.col(self.timestamp_column) > pl.col("max_date"))
            .drop("max_date")
        )


class GlobalDaysFilter(_BaseFilter):
    """
    Select first/last days from ``interactions``.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rating": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01 00:00:00",
    ...                                   "2020-02-01 00:00:01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"], format="ISO8601")
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u2|     i2|   0.5|2020-02-01 00:00:00|
    |     u2|     i3|   3.0|2020-02-01 00:00:01|
    |     u3|     i1|   1.0|2020-01-01 00:04:15|
    |     u3|     i2|   0.0|2020-01-02 00:04:14|
    |     u3|     i3|   1.0|2020-01-05 23:59:59|
    +-------+-------+------+-------------------+
    <BLANKLINE>

    >>> GlobalDaysFilter(1).transform(log_sp).show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u3|     i1|   1.0|2020-01-01 00:04:15|
    |     u3|     i2|   0.0|2020-01-02 00:04:14|
    +-------+-------+------+-------------------+
    <BLANKLINE>

    >>> GlobalDaysFilter(1, first=False).transform(log_sp).show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u2|     i2|   0.5|2020-02-01 00:00:00|
    |     u2|     i3|   3.0|2020-02-01 00:00:01|
    +-------+-------+------+-------------------+
    <BLANKLINE>
    """

    def __init__(
        self,
        days: int = 10,
        first: bool = True,
        timestamp_column: str = "timestamp",
    ):
        r"""
        :param days: length of selected data in days.
            default: `10`.
        :param first: take either first ``day`` or last.
            default: `True`.
        :param timestamp_column: timestamp column.
            default: ``timestamp``.
        """
        self.days = days
        self.first = first
        self.timestamp_column = timestamp_column
        self.first = first
        self._sanity_check()

    def _sanity_check(self) -> None:
        assert self.days > 0

    def _filter_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        if self.first:
            start_date = interactions.agg(sf.min(self.timestamp_column)).first()[0]
            end_date = sf.lit(start_date).cast(TimestampType()) + sf.expr(f"INTERVAL {self.days} days")
            return interactions.filter(col(self.timestamp_column) < end_date)

        end_date = interactions.agg(sf.max(self.timestamp_column)).first()[0]
        start_date = sf.lit(end_date).cast(TimestampType()) - sf.expr(f"INTERVAL {self.days} days")
        return interactions.filter(col(self.timestamp_column) > start_date)

    def _filter_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        filtered_interactions = interactions.copy(deep=True)

        if self.first:
            start_date = filtered_interactions[self.timestamp_column].min()
            return filtered_interactions[
                (filtered_interactions[self.timestamp_column] - start_date).dt.days < self.days
            ]
        end_date = filtered_interactions[self.timestamp_column].max()
        return filtered_interactions[(end_date - filtered_interactions[self.timestamp_column]).dt.days < self.days]

    def _filter_polars(self, interactions: PolarsDataFrame) -> PolarsDataFrame:
        if self.first:
            return interactions.filter(
                pl.col(self.timestamp_column) < (pl.col(self.timestamp_column).min() + pl.duration(days=self.days))
            )
        return interactions.filter(
            pl.col(self.timestamp_column) > (pl.col(self.timestamp_column).max() - pl.duration(days=self.days))
        )


class TimePeriodFilter(_BaseFilter):
    """
    Select a part of data between ``[start_date, end_date)``.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rating": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01 00:00:00",
    ...                                   "2020-02-01 00:00:01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"], format="ISO8601")
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u2|     i2|   0.5|2020-02-01 00:00:00|
    |     u2|     i3|   3.0|2020-02-01 00:00:01|
    |     u3|     i1|   1.0|2020-01-01 00:04:15|
    |     u3|     i2|   0.0|2020-01-02 00:04:14|
    |     u3|     i3|   1.0|2020-01-05 23:59:59|
    +-------+-------+------+-------------------+
    <BLANKLINE>

    >>> TimePeriodFilter(
    ...    start_date="2020-01-01 14:00:00",
    ...    end_date=datetime(2020, 1, 3, 0, 0, 0)
    ... ).transform(log_sp).show()
    +-------+-------+------+-------------------+
    |user_id|item_id|rating|          timestamp|
    +-------+-------+------+-------------------+
    |     u1|     i1|   1.0|2020-01-01 23:59:59|
    |     u3|     i2|   0.0|2020-01-02 00:04:14|
    +-------+-------+------+-------------------+
    <BLANKLINE>
    """

    def __init__(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        timestamp_column: str = "timestamp",
        time_column_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        r"""
        :param start_date: datetime or str with format ``time_column_format``.
            default: `None`.
        :param end_date: datetime or str with format ``time_column_format``.
            default: `None`.
        :param timestamp_column: timestamp column.
            default: ``timestamp``.
        """
        self.start_date = self._format_datetime(start_date, time_column_format)
        self.end_date = self._format_datetime(end_date, time_column_format)
        self.timestamp_column = timestamp_column

    def _format_datetime(self, date: Optional[Union[str, datetime]], time_format: str) -> datetime:
        if isinstance(date, str):
            date = datetime.strptime(date, time_format)
        return date

    def _filter_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        if self.start_date is None:
            self.start_date = interactions.agg(sf.min(self.timestamp_column)).first()[0]
        if self.end_date is None:
            self.end_date = interactions.agg(sf.max(self.timestamp_column)).first()[0] + timedelta(seconds=1)

        return interactions.filter(
            (col(self.timestamp_column) >= sf.lit(self.start_date))
            & (col(self.timestamp_column) < sf.lit(self.end_date))
        )

    def _filter_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        if self.start_date is None:
            self.start_date = interactions[self.timestamp_column].min()
        if self.end_date is None:
            self.end_date = interactions[self.timestamp_column].max() + timedelta(seconds=1)

        return interactions[
            (interactions[self.timestamp_column] >= self.start_date)
            & (interactions[self.timestamp_column] < self.end_date)
        ]

    def _filter_polars(self, interactions: PolarsDataFrame) -> PolarsDataFrame:
        if self.start_date is None:
            self.start_date = interactions.select(self.timestamp_column).min()[0, 0]
        if self.end_date is None:
            self.end_date = interactions.select(self.timestamp_column).max()[0, 0] + pl.duration(seconds=1)

        return interactions.filter(
            pl.col(self.timestamp_column).is_between(self.start_date, self.end_date, closed="left")
        )


class QuantileItemsFilter(_BaseFilter):
    """
    Filter is aimed on undersampling the interactions dataset.

    Filter algorithm performs undersampling by removing `items_proportion` of interactions
    for each items counts that exceeds the `alpha_quantile` value in distribution. Filter firstly
    removes popular items (items that have most interactions). Filter also keeps the original
    relation of items popularity among each other by removing interactions only in range of
    current item count and quantile count (specified by `alpha_quantile`).

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({
    ...        "user_id": [0, 0, 1, 2, 2, 2, 2],
    ...        "item_id": [0, 2, 1, 1, 2, 2, 2]
    ... })
    >>> log_spark = convert2spark(log_pd)
    >>> log_spark.show()
    +-------+-------+
    |user_id|item_id|
    +-------+-------+
    |      0|      0|
    |      0|      2|
    |      1|      1|
    |      2|      1|
    |      2|      2|
    |      2|      2|
    |      2|      2|
    +-------+-------+
    <BLANKLINE>

    >>> QuantileItemsFilter(query_column="user_id").transform(log_spark).show()
    +-------+-------+
    |user_id|item_id|
    +-------+-------+
    |      0|      0|
    |      1|      1|
    |      2|      1|
    |      2|      2|
    |      2|      2|
    |      0|      2|
    +-------+-------+
    <BLANKLINE>
    """

    def __init__(
        self,
        alpha_quantile: float = 0.99,
        items_proportion: float = 0.5,
        query_column: str = "query_id",
        item_column: str = "item_id",
    ) -> None:
        """
        :param alpha_quantile: Quantile value of items counts distribution to keep unchanged.
            Every items count that exceeds this value will be undersampled.
            Default: ``0.99``.
        :param items_proportion: proportion of items counts to remove for items that
            exceeds `alpha_quantile` value in range of current item count and quantile count
            to make sure we keep original relation between items unchanged.
            Default: ``0.5``.
        :param query_column: query column name.
            Default: ``query_id``.
        :param item_column: item column name.
            Default: ``item_id``.
        """
        if not 0 < alpha_quantile < 1:
            msg = "`alpha_quantile` value must be in (0, 1)"
            raise ValueError(msg)
        if not 0 < items_proportion < 1:
            msg = "`items_proportion` value must be in (0, 1)"
            raise ValueError(msg)

        self.alpha_quantile = alpha_quantile
        self.items_proportion = items_proportion
        self.query_column = query_column
        self.item_column = item_column

    def _filter_pandas(self, df: pd.DataFrame):
        items_distribution = df.groupby(self.item_column).size().reset_index().rename(columns={0: "counts"})
        users_distribution = df.groupby(self.query_column).size().reset_index().rename(columns={0: "counts"})
        count_threshold = items_distribution.loc[:, "counts"].quantile(self.alpha_quantile, interpolation="midpoint")
        df_with_counts = df.merge(items_distribution, how="left", on=self.item_column).merge(
            users_distribution, how="left", on=self.query_column, suffixes=["_items", "_users"]
        )
        long_tail = df_with_counts.loc[df_with_counts["counts_items"] <= count_threshold]
        short_tail = df_with_counts.loc[df_with_counts["counts_items"] > count_threshold]
        short_tail["num_items_to_delete"] = self.items_proportion * (
            short_tail["counts_items"] - long_tail["counts_items"].max()
        )
        short_tail["num_items_to_delete"] = short_tail["num_items_to_delete"].astype("int")
        short_tail = short_tail.sort_values("counts_users", ascending=False)

        def get_mask(x):
            mask = np.ones_like(x)
            threshold = x.iloc[0]
            mask[:threshold] = 0
            return mask

        mask = short_tail.groupby(self.item_column)["num_items_to_delete"].transform(get_mask).astype(bool)
        return pd.concat([long_tail[df.columns], short_tail.loc[mask][df.columns]])

    def _filter_polars(self, df: pl.DataFrame):
        items_distribution = df.group_by(self.item_column).len()
        users_distribution = df.group_by(self.query_column).len()
        count_threshold = items_distribution.select("len").quantile(self.alpha_quantile, "midpoint")["len"][0]
        df_with_counts = (
            df.join(items_distribution, how="left", on=self.item_column).join(
                users_distribution, how="left", on=self.query_column
            )
        ).rename({"len": "counts_items", "len_right": "counts_users"})
        long_tail = df_with_counts.filter(pl.col("counts_items") <= count_threshold)
        short_tail = df_with_counts.filter(pl.col("counts_items") > count_threshold)
        max_long_tail_count = long_tail["counts_items"].max()
        items_to_delete = (
            short_tail.select(
                self.query_column,
                self.item_column,
                self.items_proportion * (pl.col("counts_items") - max_long_tail_count),
            )
            .with_columns(pl.col("literal").cast(pl.Int64).alias("num_items_to_delete"))
            .select(self.item_column, "num_items_to_delete")
            .unique(maintain_order=True)
        )
        short_tail = short_tail.join(items_to_delete, how="left", on=self.item_column).sort(
            "counts_users", descending=True
        )
        short_tail = short_tail.with_columns(index=pl.int_range(short_tail.shape[0]))
        grouped = short_tail.group_by(self.item_column, maintain_order=True).agg(
            pl.col("index"), pl.col("num_items_to_delete")
        )
        grouped = grouped.with_columns(
            pl.col("num_items_to_delete").list.get(0),
            (pl.col("index").list.len() - pl.col("num_items_to_delete").list.get(0)).alias("tail"),
        )
        grouped = grouped.with_columns(pl.col("index").list.tail(pl.col("tail")))
        grouped = grouped.explode("index").select("index")
        short_tail = grouped.join(short_tail, how="left", on="index")
        return pl.concat([long_tail.select(df.columns), short_tail.select(df.columns)])

    def _filter_spark(self, df: SparkDataFrame):
        items_distribution = df.groupBy(self.item_column).agg(sf.count(self.query_column).alias("counts_items"))
        users_distribution = df.groupBy(self.query_column).agg(sf.count(self.item_column).alias("counts_users"))
        count_threshold = items_distribution.toPandas().loc[:, "counts_items"].quantile(self.alpha_quantile, "midpoint")
        df_with_counts = df.join(items_distribution, on=self.item_column).join(users_distribution, on=self.query_column)
        long_tail = df_with_counts.filter(sf.col("counts_items") <= count_threshold)
        short_tail = df_with_counts.filter(sf.col("counts_items") > count_threshold)
        max_long_tail_count = long_tail.agg({"counts_items": "max"}).collect()[0][0]
        items_to_delete = (
            short_tail.withColumn(
                "num_items_to_delete",
                (self.items_proportion * (sf.col("counts_items") - max_long_tail_count)).cast("int"),
            )
            .select(self.item_column, "num_items_to_delete")
            .distinct()
        )
        short_tail = short_tail.join(items_to_delete, on=self.item_column, how="left")
        short_tail = short_tail.withColumn(
            "index", sf.row_number().over(Window.partitionBy(self.item_column).orderBy(sf.col("counts_users").desc()))
        )
        short_tail = short_tail.filter(sf.col("index") > sf.col("num_items_to_delete"))
        return long_tail.select(df.columns).union(short_tail.select(df.columns))
