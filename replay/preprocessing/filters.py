"""
Select or remove data by some criteria
"""
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, Window, functions as sf
from pyspark.sql.functions import col
from pyspark.sql.types import TimestampType
from typing import Union, Optional

from replay.data import AnyDataFrame
from replay.utils.spark_utils import convert2spark
from replay.utils.session_handler import State


def filter_by_min_count(
    data_frame: AnyDataFrame, num_entries: int, group_by: str = "user_idx"
) -> DataFrame:
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
) -> DataFrame:
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
    """
    data_frame = convert2spark(data_frame)
    data_frame = data_frame.filter(data_frame[rating_column] >= value)
    return data_frame


# pylint: disable=too-many-arguments,
def take_num_user_interactions(
    log: DataFrame,
    num_interactions: int = 10,
    first: bool = True,
    date_col: str = "timestamp",
    user_col: str = "user_idx",
    item_col: Optional[str] = "item_idx",
) -> DataFrame:
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
    log: DataFrame,
    days: int = 10,
    first: bool = True,
    date_col: str = "timestamp",
    user_col: str = "user_idx",
) -> DataFrame:
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
    log: DataFrame,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    date_column: str = "timestamp",
) -> DataFrame:
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
    log: DataFrame,
    duration_days: int,
    first: bool = True,
    date_column: str = "timestamp",
) -> DataFrame:
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
