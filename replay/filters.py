"""
Содержит функции, позволяющие отобрать данные по некоторому критерию.
"""
from datetime import datetime, timedelta
from pyspark.sql import DataFrame, Window, functions as sf
from pyspark.sql.functions import col
from pyspark.sql.types import TimestampType
from typing import Union, Optional

from replay.constants import AnyDataFrame
from replay.utils import convert2spark


def min_entries(data_frame: AnyDataFrame, num_entries: int) -> DataFrame:
    """
    Удаляет из датафрейма записи всех пользователей,
    имеющих менее ``num_entries`` оценок.

    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_id": [1, 1, 2]})
    >>> min_entries(data_frame, 2).toPandas()
       user_id
    0        1
    1        1
    """
    data_frame = convert2spark(data_frame)
    entries_by_user = data_frame.groupBy("user_id").count()  # type: ignore
    remaining_users = entries_by_user.filter(
        entries_by_user["count"] >= num_entries
    )[["user_id"]]
    data_frame = data_frame.join(
        remaining_users, on="user_id", how="inner"
    )  # type: ignore
    return data_frame


def min_rating(
    data_frame: AnyDataFrame, value: float, column="relevance"
) -> DataFrame:
    """
    Удаляет из датафрейма записи с оценкой меньше ``value`` в колонке ``column``.

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
     Для каждого пользователя возвращает первые/последние `n` взаимодействий из лога.

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

    Первое взаимодействие из лога:

    >>> filter_user_interactions(log_sp, 1, True).show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u3|     i1|1.0|2020-01-01 00:04:15|
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    Последнее взаимодействие из лога:

    >>> filter_user_interactions(log_sp, 1, False, item_col=None).show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u3|     i3|1.0|2020-01-05 23:59:59|
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    >>> filter_user_interactions(log_sp, 1, False).show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u3|     i3|1.0|2020-01-05 23:59:59|
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    :param log: лог взаимодействия пользователей с объектами, спарк-датафрейм
    :param num_interactions: число взаимодействий, которое будет выбрано для каждого пользователя
    :param first: выбор первых/последних взаимодействий. Выбираются взаимодействия от начала истории, если True,
        последние, если False
    :param date_col: имя столбца с датой взаимодействия
    :param user_col: имя столбца с id пользователей
    :param item_col: имя столбца с id объекта для дополнительной сортировки взаимодействий, произошедших одновременно.
        Если None, не участвует в сортировке.
    :return: спарк-датафрейм, содержащий выбранные взаимодействия
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
    Для каждого пользователя возвращает историю взаимодействия за первые/последние days
    с момента первого/последнего взаимодействия пользователя.

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

    Взаимодействия за первый день истории пользователя:

    >>> filter_by_user_duration(log_sp, 1, True).show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u3|     i1|1.0|2020-01-01 00:04:15|
    |     u3|     i2|0.0|2020-01-02 00:04:14|
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    Взаимодействия за последний день истории пользователя:

    >>> filter_by_user_duration(log_sp, 1, False).show()
    +-------+-------+---+-------------------+
    |user_id|item_id|rel|          timestamp|
    +-------+-------+---+-------------------+
    |     u3|     i3|1.0|2020-01-05 23:59:59|
    |     u1|     i1|1.0|2020-01-01 23:59:59|
    |     u2|     i2|0.5|2020-02-01 00:00:00|
    |     u2|     i3|3.0|2020-02-01 00:00:00|
    +-------+-------+---+-------------------+
    <BLANKLINE>

    :param log: лог взаимодействия пользователей с объектами, спарк-датафрейм
    :param days: длительность лога взаимодействия для каждого пользователя после фильтрации
    :param first: выбор первых/последних взаимодействий. Выбираются взаимодействия от начала истории, если True,
        последние, если False
    :param date_col: имя столбца с датой взаимодействия
    :param user_col: имя столбца с id пользователей
    :return: спарк-датафрейм, содержащий выбранные взаимодействия
    """

    window = Window.partitionBy(user_col)
    if first:
        return (
            log.withColumn("min_date", sf.min(col(date_col)).over(window))
            .filter(
                col(date_col)
                < col("min_date") + sf.expr("INTERVAL {} days".format(days))
            )
            .drop("min_date")
        )

    return (
        log.withColumn("max_date", sf.max(col(date_col)).over(window))
        .filter(
            col(date_col)
            > col("max_date") - sf.expr("INTERVAL {} days".format(days))
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
    Возвращает лог за период в интервале [start_date, end_date). Если start_date или end_date не указаны,
    возвращается лог от начала/до конца соотвественно.

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

    :param log: лог взаимодействия пользователей с объектами, спарк-датафрейм
    :param start_date: дата в datetime или строка в формате "yyyy-MM-dd HH:mm:ss".
        Начиная с этой даты данные будут включены в отфильтрованный лог
    :param end_date: дата в datetime или строка в формате "yyyy-MM-dd HH:mm:ss".
        Данные до этой даты будут включены в отфильтрованный лог
    :param date_column: имя столбца с датой взаимодействия
    :return: спарк-датафрейм, содержащий выбранные взаимодействия
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
    Возвращает лог взаимодействия за выбранное число дней от начала/конца лога
    в зависимости от значения параметра first.

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

    :param log: лог взаимодействия пользователей с объектами, спарк-датафрейм
    :param duration_days: число дней, которые нужно включить в отфильтрованный лог
    :param first: выбор первых/последних взаимодействий. Выбираются взаимодействия от начала истории,
        если True, последние, если False
    :param date_column: имя столбца с датой взаимодействия
    :return: спарк-датафрейм, содержащий выбранные взаимодействия
    """
    if first:
        start_date = log.agg(sf.min(date_column)).first()[0]
        end_date = sf.lit(start_date).cast(TimestampType()) + sf.expr(
            "INTERVAL {} days".format(duration_days)
        )
        return log.filter(col(date_column) < end_date)

    end_date = log.agg(sf.max(date_column)).first()[0]
    start_date = sf.lit(end_date).cast(TimestampType()) - sf.expr(
        "INTERVAL {} days".format(duration_days)
    )
    return log.filter(col(date_column) > start_date)
