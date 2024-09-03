import numpy as np

from .spark_utils import convert2spark
from .types import PYSPARK_AVAILABLE, DataFrameLike

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf


def get_item_recency(
    log: DataFrameLike,
    decay: float = 30,
    limit: float = 0.1,
    kind: str = "exp",
):
    """
    Calculate item weight showing when the majority of interactions with this item happened.

    :param log: interactions log
    :param decay: number of days after which the weight is reduced by half, must be grater than 1
    :param limit: minimal value the weight can reach
    :param kind: type of smoothing, one of [power, exp, linear]
        Corresponding functions are ``power``: ``age^c``,
        ``exp``: ``c^age``, ``linear``: ``1-c*age``
    :return: DataFrame with item weights

    >>> import pandas as pd
    >>> from pyspark.sql.functions import round
    >>> d = {
    ...     "item_idx": [1, 1, 2, 3, 3],
    ...     "timestamp": ["2099-03-19", "2099-03-20", "2099-03-22", "2099-03-27", "2099-03-25"],
    ...     "relevance": [1, 1, 1, 1, 1],
    ... }
    >>> df = pd.DataFrame(d)
    >>> df
       item_idx   timestamp  relevance
    0         1  2099-03-19          1
    1         1  2099-03-20          1
    2         2  2099-03-22          1
    3         3  2099-03-27          1
    4         3  2099-03-25          1

    Age in days is calculated for every item,
    which is transformed into a weight using some function.
    There are three types of smoothing types available: power, exp and linear.
    Each type calculates a parameter ``c`` based on the ``decay`` argument,
    so that an item with ``age==decay`` has weight 0.5.

    Power smoothing falls quickly in the beginning but decays slowly afterwards as ``age^c``.

    >>> (
    ...     get_item_recency(df, kind="power")
    ...     .select("item_idx", "timestamp", round("relevance", 4).alias("relevance"))
    ...     .orderBy("item_idx")
    ...     .show()
    ... )
    +--------+-------------------+---------+
    |item_idx|          timestamp|relevance|
    +--------+-------------------+---------+
    |       1|2099-03-19 12:00:00|   0.6632|
    |       2|2099-03-22 00:00:00|   0.7204|
    |       3|2099-03-26 00:00:00|      1.0|
    +--------+-------------------+---------+
    <BLANKLINE>

    Exponential smoothing is the other way around. Old objects decay more quickly as ``c^age``.

    >>> (
    ...     get_item_recency(df, kind="exp")
    ...     .select("item_idx", "timestamp", round("relevance", 4).alias("relevance"))
    ...     .orderBy("item_idx")
    ...     .show()
    ... )
    +--------+-------------------+---------+
    |item_idx|          timestamp|relevance|
    +--------+-------------------+---------+
    |       1|2099-03-19 12:00:00|   0.8606|
    |       2|2099-03-22 00:00:00|   0.9117|
    |       3|2099-03-26 00:00:00|      1.0|
    +--------+-------------------+---------+
    <BLANKLINE>

    Last type is a linear smoothing: ``1 - c*age``.

    >>> (
    ...     get_item_recency(df, kind="linear")
    ...     .select("item_idx", "timestamp", round("relevance", 4).alias("relevance"))
    ...     .orderBy("item_idx")
    ...     .show()
    ... )
    +--------+-------------------+---------+
    |item_idx|          timestamp|relevance|
    +--------+-------------------+---------+
    |       1|2099-03-19 12:00:00|   0.8917|
    |       2|2099-03-22 00:00:00|   0.9333|
    |       3|2099-03-26 00:00:00|      1.0|
    +--------+-------------------+---------+
    <BLANKLINE>

    This function **does not** take relevance values of interactions into account.
    Only item age is used.
    """
    log = convert2spark(log)
    items = log.select(
        "item_idx",
        sf.unix_timestamp(sf.to_timestamp("timestamp")).alias("timestamp"),
    )
    items = items.groupBy("item_idx").agg(sf.mean("timestamp").alias("timestamp"))
    items = items.withColumn("relevance", sf.lit(1))
    items = smoothe_time(items, decay, limit, kind)
    return items


def smoothe_time(
    log: DataFrameLike,
    decay: float = 30,
    limit: float = 0.1,
    kind: str = "exp",
):
    """
    Weighs ``relevance`` column with a time-dependent weight.

    :param log: interactions log
    :param decay: number of days after which the weight is reduced by half, must be grater than 1
    :param limit: minimal value the weight can reach
    :param kind: type of smoothing, one of [power, exp, linear].
        Corresponding functions are ``power``: ``age^c``,
        ``exp``: ``c^age``, ``linear``: ``1-c*age``
    :return: modified DataFrame

    >>> import pandas as pd
    >>> from pyspark.sql.functions import round
    >>> d = {
    ...     "item_idx": [1, 1, 2, 3, 3],
    ...     "timestamp": ["2099-03-19", "2099-03-20", "2099-03-22", "2099-03-27", "2099-03-25"],
    ...     "relevance": [1, 1, 1, 1, 1],
    ... }
    >>> df = pd.DataFrame(d)
    >>> df
       item_idx   timestamp  relevance
    0         1  2099-03-19          1
    1         1  2099-03-20          1
    2         2  2099-03-22          1
    3         3  2099-03-27          1
    4         3  2099-03-25          1

    Power smoothing falls quickly in the beginning but decays slowly afterwards as ``age^c``.

    >>> (
    ...     smoothe_time(df, kind="power")
    ...     .select("item_idx", "timestamp", round("relevance", 4).alias("relevance"))
    ...     .orderBy("timestamp")
    ...     .show()
    ... )
    +--------+-------------------+---------+
    |item_idx|          timestamp|relevance|
    +--------+-------------------+---------+
    |       1|2099-03-19 00:00:00|    0.639|
    |       1|2099-03-20 00:00:00|   0.6546|
    |       2|2099-03-22 00:00:00|   0.6941|
    |       3|2099-03-25 00:00:00|   0.7994|
    |       3|2099-03-27 00:00:00|      1.0|
    +--------+-------------------+---------+
    <BLANKLINE>

    Exponential smoothing is the other way around. Old objects decay more quickly as ``c^age``.

    >>> (
    ...     smoothe_time(df, kind="exp")
    ...     .select("item_idx", "timestamp", round("relevance", 4).alias("relevance"))
    ...     .orderBy("timestamp")
    ...     .show()
    ... )
    +--------+-------------------+---------+
    |item_idx|          timestamp|relevance|
    +--------+-------------------+---------+
    |       1|2099-03-19 00:00:00|   0.8312|
    |       1|2099-03-20 00:00:00|   0.8507|
    |       2|2099-03-22 00:00:00|   0.8909|
    |       3|2099-03-25 00:00:00|   0.9548|
    |       3|2099-03-27 00:00:00|      1.0|
    +--------+-------------------+---------+
    <BLANKLINE>

    Last type is a linear smoothing: ``1 - c*age``.

    >>> (
    ...     smoothe_time(df, kind="linear")
    ...     .select("item_idx", "timestamp", round("relevance", 4).alias("relevance"))
    ...     .orderBy("timestamp")
    ...     .show()
    ... )
    +--------+-------------------+---------+
    |item_idx|          timestamp|relevance|
    +--------+-------------------+---------+
    |       1|2099-03-19 00:00:00|   0.8667|
    |       1|2099-03-20 00:00:00|   0.8833|
    |       2|2099-03-22 00:00:00|   0.9167|
    |       3|2099-03-25 00:00:00|   0.9667|
    |       3|2099-03-27 00:00:00|      1.0|
    +--------+-------------------+---------+
    <BLANKLINE>

    These examples use constant relevance 1, so resulting weight equals the time dependent weight.
    But actually this value is an updated relevance.

    >>> d = {
    ...     "item_idx": [1, 2, 3],
    ...     "timestamp": ["2099-03-19", "2099-03-20", "2099-03-22"],
    ...     "relevance": [10, 3, 0.1],
    ... }
    >>> df = pd.DataFrame(d)
    >>> df
       item_idx   timestamp  relevance
    0         1  2099-03-19       10.0
    1         2  2099-03-20        3.0
    2         3  2099-03-22        0.1
    >>> (
    ...     smoothe_time(df)
    ...     .select("item_idx", "timestamp", round("relevance", 4).alias("relevance"))
    ...     .orderBy("timestamp")
    ...     .show()
    ... )
    +--------+-------------------+---------+
    |item_idx|          timestamp|relevance|
    +--------+-------------------+---------+
    |       1|2099-03-19 00:00:00|   9.3303|
    |       2|2099-03-20 00:00:00|   2.8645|
    |       3|2099-03-22 00:00:00|      0.1|
    +--------+-------------------+---------+
    <BLANKLINE>
    """
    log = convert2spark(log)
    log = log.withColumn("timestamp", sf.unix_timestamp(sf.to_timestamp("timestamp")))
    last_date = log.agg({"timestamp": "max"}).collect()[0].asDict()["max(timestamp)"]
    day_in_secs = 86400
    log = log.withColumn("age", (last_date - sf.col("timestamp")) / day_in_secs)
    if kind == "power":
        power = np.log(0.5) / np.log(decay)
        log = log.withColumn("age", sf.pow(sf.col("age") + 1, power))
    elif kind == "exp":
        base = np.exp(np.log(0.5) / decay)
        log = log.withColumn("age", sf.pow(base, "age"))
    elif kind == "linear":
        k = 0.5 / decay
        log = log.withColumn("age", 1 - k * sf.col("age"))
    else:
        msg = f"parameter kind must be one of [power, exp, linear], got {kind}"
        raise ValueError(msg)

    log = log.withColumn("age", sf.when(sf.col("age") < limit, limit).otherwise(sf.col("age")))
    log = log.withColumn("relevance", sf.col("relevance") * sf.col("age")).drop("age")
    log = log.withColumn("timestamp", sf.to_timestamp("timestamp"))
    return log
