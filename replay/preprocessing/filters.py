"""
Select or remove data by some criteria
"""
from datetime import datetime, timedelta
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame, Window, functions as sf
from pyspark.sql.functions import col
from pyspark.sql.types import TimestampType
from typing import Union, Optional, Tuple

from replay.data import AnyDataFrame
from replay.utils.spark_utils import convert2spark
from replay.utils.session_handler import State


# pylint: disable=too-few-public-methods, too-many-instance-attributes
class InteractionEntriesFilter:
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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        user_column: str = "user_id",
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
            Default: `True`.
        """
        self.user_column = user_column
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

    def _filter_column(
        self,
        interactions: AnyDataFrame,
        interaction_count: int,
        min_inter: Optional[int],
        max_inter: Optional[int],
        agg_column: str,
        non_agg_column: str,
    ) -> Tuple[AnyDataFrame, int, int]:
        if not min_inter and not max_inter:
            return interactions, 0, interaction_count

        if isinstance(interactions, SparkDataFrame):
            return self._filter_column_spark(
                interactions, interaction_count, min_inter, max_inter, agg_column, non_agg_column
            )

        return self._filter_column_pandas(
            interactions, interaction_count, min_inter, max_inter, agg_column, non_agg_column
        )

    # pylint: disable=no-self-use
    def _filter_column_pandas(
        self,
        interactions: PandasDataFrame,
        interaction_count: int,
        min_inter: Optional[int],
        max_inter: Optional[int],
        agg_column: str,
        non_agg_column: str,
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

    # pylint: disable=no-self-use
    def _filter_column_spark(
        self,
        interactions: SparkDataFrame,
        interaction_count: int,
        min_inter: Optional[int],
        max_inter: Optional[int],
        agg_column: str,
        non_agg_column: str,
    ) -> Tuple[SparkDataFrame, int, int]:
        filtered_interactions = interactions.withColumn(
            "count", sf.count(non_agg_column).over(Window.partitionBy(agg_column))
        )
        if min_inter:
            filtered_interactions = filtered_interactions.filter(sf.col("count") >= min_inter)
        if max_inter:
            filtered_interactions = filtered_interactions.filter(sf.col("count") <= max_inter)
        filtered_interactions = filtered_interactions.drop("count")

        if self.allow_caching is True:
            filtered_interactions.cache()
            interactions.unpersist()
        end_len_dataframe = filtered_interactions.count()
        different_len = interaction_count - end_len_dataframe

        return filtered_interactions, different_len, end_len_dataframe

    # pylint: disable=too-many-function-args
    def transform(self, interactions: AnyDataFrame) -> AnyDataFrame:
        r"""Filter interactions.

        :param interactions: DataFrame containing columns ``user_column``, ``item_column``.

        :returns: filtered DataFrame.
        """
        is_no_dropped_user_item = [False, False]
        current_index = 0
        interaction_count = interactions.count() if isinstance(interactions, SparkDataFrame) else len(interactions)
        while is_no_dropped_user_item[0] is False or is_no_dropped_user_item[1] is False:
            if current_index == 0:
                min_inter = self.min_inter_per_user
                max_inter = self.max_inter_per_user
                agg_column = self.user_column
                non_agg_column = self.item_column
            else:
                min_inter = self.min_inter_per_item
                max_inter = self.max_inter_per_item
                agg_column = self.item_column
                non_agg_column = self.user_column

            interactions, dropped_interact, interaction_count = self._filter_column(
                interactions, interaction_count, min_inter, max_inter, agg_column, non_agg_column
            )
            is_no_dropped_user_item[current_index] = not dropped_interact
            current_index = (current_index + 1) % 2     # current_index only in (0, 1)

        return interactions


def filter_by_min_count(
    data_frame: AnyDataFrame, num_entries: int, group_by: str = "user_idx"
) -> SparkDataFrame:
    """
    Remove entries with entities (e.g. users, items) which are presented in `data_frame`
    less than `num_entries` times. The `data_frame` is grouped by `group_by` column,
    which is entry column name, to calculate counts.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1, 1, 2]})
    >>> filter_by_min_count(data_frame, 2).toPandas()
       user_idx
    0         1
    1         1

    :param data_frame: spark or pandas dataframe to apply filter
    :param num_entries: minimal number of times the entry should appear in dataset
        in order to remain
    :param group_by: entity column, which is used to calculate entity occurrence couns
    :return: filteder `data_frame`
    """
    data_frame = convert2spark(data_frame)
    input_count = data_frame.count()
    count_by_group = data_frame.groupBy(group_by).agg(
        sf.count(group_by).alias(f"{group_by}_temp_count")
    )
    remaining_entities = count_by_group.filter(
        count_by_group[f"{group_by}_temp_count"] >= num_entries
    ).select(group_by)
    data_frame = data_frame.join(remaining_entities, on=group_by, how="inner")
    output_count = data_frame.count()
    diff = (input_count - output_count) / input_count
    if diff > 0.5:
        logger_level = State().logger.warning
    else:
        logger_level = State().logger.info
    logger_level(
        "current threshold removes %s%% of data",
        diff,
    )
    return data_frame


def filter_out_low_ratings(
    data_frame: AnyDataFrame, value: float, rating_column="relevance"
) -> SparkDataFrame:
    """
    Remove records with records less than ``value`` in ``column``.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"relevance": [1, 5, 3.5, 4]})
    >>> filter_out_low_ratings(data_frame, 3.5).show()
    +---------+
    |relevance|
    +---------+
    |      5.0|
    |      3.5|
    |      4.0|
    +---------+
    <BLANKLINE>

    :param data_frame: spark or pandas dataframe to apply filter
    :param value: minimal value the entry should appear in dataset
        in order to remain
    :param rating_column: the column in which filtering is performed.
    :return: filtered DataFrame
    """
    data_frame = convert2spark(data_frame)
    data_frame = data_frame.filter(data_frame[rating_column] >= value)
    return data_frame


# pylint: disable=too-many-arguments,
def take_num_user_interactions(
    log: SparkDataFrame,
    num_interactions: int = 10,
    first: bool = True,
    date_col: str = "timestamp",
    user_col: str = "user_idx",
    item_col: Optional[str] = "item_idx",
) -> SparkDataFrame:
    """
    Get first/last ``num_interactions`` interactions for each user.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_idx": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_idx": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    |      u3|      i1|1.0|2020-01-01 00:04:15|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    |      u3|      i3|1.0|2020-01-05 23:59:59|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    Only first interaction:

    >>> take_num_user_interactions(log_sp, 1, True).orderBy('user_idx').show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u3|      i1|1.0|2020-01-01 00:04:15|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    Only last interaction:

    >>> take_num_user_interactions(log_sp, 1, False, item_col=None).orderBy('user_idx').show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u3|      i3|1.0|2020-01-05 23:59:59|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    >>> take_num_user_interactions(log_sp, 1, False).orderBy('user_idx').show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    |      u3|      i3|1.0|2020-01-05 23:59:59|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    :param log: historical interactions DataFrame
    :param num_interactions: number of interactions to leave per user
    :param first: take either first ``num_interactions`` or last.
    :param date_col: date column
    :param user_col: user column
    :param item_col: item column to help sort simultaneous interactions.
        If None, it is ignored.
    :return: filtered DataFrame
    """
    sorting_order = [col(date_col)]
    if item_col is not None:
        sorting_order.append(col(item_col))

    if not first:
        sorting_order = [col_.desc() for col_ in sorting_order]

    window = Window().orderBy(*sorting_order).partitionBy(col(user_col))

    return (
        log.withColumn("temp_rank", sf.row_number().over(window))
        .filter(col("temp_rank") <= num_interactions)
        .drop("temp_rank")
    )


def take_num_days_of_user_hist(
    log: SparkDataFrame,
    days: int = 10,
    first: bool = True,
    date_col: str = "timestamp",
    user_col: str = "user_idx",
) -> SparkDataFrame:
    """
    Get first/last ``days`` of users' interactions.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_idx": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_idx": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.orderBy('user_idx', 'item_idx').show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    |      u3|      i1|1.0|2020-01-01 00:04:15|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    |      u3|      i3|1.0|2020-01-05 23:59:59|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    Get first day:

    >>> take_num_days_of_user_hist(log_sp, 1, True).orderBy('user_idx', 'item_idx').show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    |      u3|      i1|1.0|2020-01-01 00:04:15|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    Get last day:

    >>> take_num_days_of_user_hist(log_sp, 1, False).orderBy('user_idx', 'item_idx').show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    |      u3|      i3|1.0|2020-01-05 23:59:59|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    :param log: historical DataFrame
    :param days: how many days to return per user
    :param first: take either first ``days`` or last
    :param date_col: date column
    :param user_col: user column
    """

    window = Window.partitionBy(user_col)
    if first:
        return (
            log.withColumn("min_date", sf.min(col(date_col)).over(window))
            .filter(
                col(date_col)
                < col("min_date") + sf.expr(f"INTERVAL {days} days")
            )
            .drop("min_date")
        )

    return (
        log.withColumn("max_date", sf.max(col(date_col)).over(window))
        .filter(
            col(date_col) > col("max_date") - sf.expr(f"INTERVAL {days} days")
        )
        .drop("max_date")
    )


def take_time_period(
    log: SparkDataFrame,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    date_column: str = "timestamp",
) -> SparkDataFrame:
    """
    Select a part of data between ``[start_date, end_date)``.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_idx": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_idx": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    |      u3|      i1|1.0|2020-01-01 00:04:15|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    |      u3|      i3|1.0|2020-01-05 23:59:59|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    >>> take_time_period(log_sp, start_date="2020-01-01 14:00:00", end_date=datetime(2020, 1, 3, 0, 0, 0)).show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    :param log: historical DataFrame
    :param start_date: datetime or str with format "yyyy-MM-dd HH:mm:ss".
    :param end_date: datetime or str with format "yyyy-MM-dd HH:mm:ss".
    :param date_column: date column
    """
    if start_date is None:
        start_date = log.agg(sf.min(date_column)).first()[0]
    if end_date is None:
        end_date = log.agg(sf.max(date_column)).first()[0] + timedelta(
            seconds=1
        )

    return log.filter(
        (col(date_column) >= sf.lit(start_date).cast(TimestampType()))
        & (col(date_column) < sf.lit(end_date).cast(TimestampType()))
    )


def take_num_days_of_global_hist(
    log: SparkDataFrame,
    duration_days: int,
    first: bool = True,
    date_column: str = "timestamp",
) -> SparkDataFrame:
    """
    Select first/last days from ``log``.

    >>> import pandas as pd
    >>> from replay.utils.spark_utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_idx": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_idx": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    |      u3|      i1|1.0|2020-01-01 00:04:15|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    |      u3|      i3|1.0|2020-01-05 23:59:59|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    >>> take_num_days_of_global_hist(log_sp, 1).show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u1|      i1|1.0|2020-01-01 23:59:59|
    |      u3|      i1|1.0|2020-01-01 00:04:15|
    |      u3|      i2|0.0|2020-01-02 00:04:14|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    >>> take_num_days_of_global_hist(log_sp, 1, first=False).show()
    +--------+--------+---+-------------------+
    |user_idx|item_idx|rel|          timestamp|
    +--------+--------+---+-------------------+
    |      u2|      i2|0.5|2020-02-01 00:00:00|
    |      u2|      i3|3.0|2020-02-01 00:00:00|
    +--------+--------+---+-------------------+
    <BLANKLINE>

    :param log: historical DataFrame
    :param duration_days: length of selected data in days
    :param first: take either first ``duration_days`` or last
    :param date_column: date column
    """
    if first:
        start_date = log.agg(sf.min(date_column)).first()[0]
        end_date = sf.lit(start_date).cast(TimestampType()) + sf.expr(
            f"INTERVAL {duration_days} days"
        )
        return log.filter(col(date_column) < end_date)

    end_date = log.agg(sf.max(date_column)).first()[0]
    start_date = sf.lit(end_date).cast(TimestampType()) - sf.expr(
        f"INTERVAL {duration_days} days"
    )
    return log.filter(col(date_column) > start_date)
