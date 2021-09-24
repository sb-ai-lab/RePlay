"""
Select or remove data by some criteria
"""
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, Window, functions as sf
from pyspark.sql.functions import col
from pyspark.sql.types import TimestampType
from typing import Union, Optional

from replay.constants import AnyDataFrame
from replay.utils import convert2spark
from replay.session_handler import State


def min_entries(data_frame: AnyDataFrame, num_entries: int) -> DataFrame:
    """
    Remove users with less than ``num_entries`` ratings.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_id": [1, 1, 2]})
    >>> min_entries(data_frame, 2).toPandas()
       user_id
    0        1
    1        1
    """
    data_frame = convert2spark(data_frame)
    input_count = data_frame.count()
    entries_by_user = data_frame.groupBy("user_id").count()  # type: ignore
    remaining_users = entries_by_user.filter(
        entries_by_user["count"] >= num_entries
    )[["user_id"]]
    data_frame = data_frame.join(
        remaining_users, on="user_id", how="inner"
    )  # type: ignore
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


def min_rating(
    data_frame: AnyDataFrame, value: float, column="relevance"
) -> DataFrame:
    """
    Remove records with records less than ``value`` in ``column``.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"relevance": [1, 5, 3, 4]})
    >>> min_rating(data_frame, 3.5).toPandas()
       relevance
    0          5
    1          4
    """
    data_frame = convert2spark(data_frame)
    data_frame = data_frame.filter(data_frame[column] > value)  # type: ignore
    return data_frame


# pylint: disable=too-many-arguments,
def filter_user_interactions(
    log: DataFrame,
    num_interactions: int = 10,
    first: bool = True,
    date_col: str = "timestamp",
    user_col: str = "user_id",
    item_col: Optional[str] = "item_id",
) -> DataFrame:
    """
     Get first/last ``num_interactions`` interactions for each user.

    >>> import pandas as pd
    >>> from replay.utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    |     u3|     i1|1.0|2020-01-01 00:04:15|
    |     u3|     i2|0.0|2020-01-02 00:04:14|
    |     u3|     i3|1.0|2020-01-05 23:59:59|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    Only first interaction:

    >>> filter_user_interactions(log_sp, 1, True).orderBy('user_id').show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u3|     i1|1.0|2020-01-01 00:04:15|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    Only last interaction:

    >>> filter_user_interactions(log_sp, 1, False, item_col=None).orderBy('user_id').show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u3|     i3|1.0|2020-01-05 23:59:59|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    >>> filter_user_interactions(log_sp, 1, False).orderBy('user_id').show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    |     u3|     i3|1.0|2020-01-05 23:59:59|
    +-------+-------+---+-------------------+
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
        log.withColumn("rank", sf.row_number().over(window))
        .filter(col("rank") <= num_interactions)
        .drop("rank")
    )


def filter_by_user_duration(
    log: DataFrame,
    days: int = 10,
    first: bool = True,
    date_col: str = "timestamp",
    user_col: str = "user_id",
) -> DataFrame:
    """
    Get first/last ``days`` of user interactions.

    >>> import pandas as pd
    >>> from replay.utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.orderBy('user_id', 'item_id').show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    |     u3|     i1|1.0|2020-01-01 00:04:15|
    |     u3|     i2|0.0|2020-01-02 00:04:14|
    |     u3|     i3|1.0|2020-01-05 23:59:59|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    Get first day:

    >>> filter_by_user_duration(log_sp, 1, True).orderBy('user_id', 'item_id').show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    |     u3|     i1|1.0|2020-01-01 00:04:15|
    |     u3|     i2|0.0|2020-01-02 00:04:14|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    Get last day:

    >>> filter_by_user_duration(log_sp, 1, False).orderBy('user_id', 'item_id').show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    |     u3|     i3|1.0|2020-01-05 23:59:59|
    +-------+-------+---+-------------------+
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
            col(date_col)
            > col("max_date") - sf.expr(f"INTERVAL {days} days")
        )
        .drop("max_date")
    )


def filter_between_dates(
    log: DataFrame,
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    date_column: str = "timestamp",
) -> DataFrame:
    """
    Select a part of data between ``[start_date, end_date)``.

    >>> import pandas as pd
    >>> from replay.utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    |     u3|     i1|1.0|2020-01-01 00:04:15|
    |     u3|     i2|0.0|2020-01-02 00:04:14|
    |     u3|     i3|1.0|2020-01-05 23:59:59|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    >>> filter_between_dates(log_sp, start_date="2020-01-01 14:00:00", end_date=datetime(2020, 1, 3, 0, 0, 0)).show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u3|     i2|0.0|2020-01-02 00:04:14|
    +-------+-------+---+-------------------+
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


def filter_by_duration(
    log: DataFrame,
    duration_days: int,
    first: bool = True,
    date_column: str = "timestamp",
) -> DataFrame:
    """
    Select first/last days from ``log``.

    >>> import pandas as pd
    >>> from replay.utils import convert2spark
    >>> log_pd = pd.DataFrame({"user_id": ["u1", "u2", "u2", "u3", "u3", "u3"],
    ...                     "item_id": ["i1", "i2","i3", "i1", "i2","i3"],
    ...                     "rel": [1., 0.5, 3, 1, 0, 1],
    ...                     "timestamp": ["2020-01-01 23:59:59", "2020-02-01",
    ...                                   "2020-02-01", "2020-01-01 00:04:15",
    ...                                   "2020-01-02 00:04:14", "2020-01-05 23:59:59"]},
    ...             )
    >>> log_pd["timestamp"] = pd.to_datetime(log_pd["timestamp"])
    >>> log_sp = convert2spark(log_pd)
    >>> log_sp.show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    |     u3|     i1|1.0|2020-01-01 00:04:15|
    |     u3|     i2|0.0|2020-01-02 00:04:14|
    |     u3|     i3|1.0|2020-01-05 23:59:59|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    >>> filter_by_duration(log_sp, 1).show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u3|     i1|1.0|2020-01-01 00:04:15|
    |     u3|     i2|0.0|2020-01-02 00:04:14|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    >>> filter_by_duration(log_sp, 1, first=False).show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    +-------+-------+---+-------------------+
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
