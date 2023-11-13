import pyspark.sql.functions as sf
import numpy as np

from replay.utils.spark_utils import convert2spark
from replay.data import AnyDataFrame


def get_item_recency(
    log: AnyDataFrame,
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
    >>> d = {}
    >>> d["item_idx"] = [1, 1, 2, 3, 3]
    >>> d["timestamp"] = ["2099-03-19", "2099-03-20", "2099-03-22", "2099-03-27", "2099-03-25"]
    >>> d["relevance"] = [1, 1, 1, 1, 1]
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

    >>> get_item_recency(df, kind="power").orderBy("item_idx").show()
    +--------+-------------------+------------------+
    |item_idx|          timestamp|         relevance|
    +--------+-------------------+------------------+
    |       1|2099-03-19 12:00:00|0.6632341020947187|
    |       2|2099-03-22 00:00:00|0.7203662792445817|
    |       3|2099-03-26 00:00:00|               1.0|
    +--------+-------------------+------------------+
    <BLANKLINE>

    Exponential smoothing is the other way around. Old objects decay more quickly as ``c^age``.

    >>> get_item_recency(df, kind="exp").orderBy("item_idx").show()
    +--------+-------------------+------------------+
    |item_idx|          timestamp|         relevance|
    +--------+-------------------+------------------+
    |       1|2099-03-19 12:00:00|0.8605514372443304|
    |       2|2099-03-22 00:00:00| 0.911722488558217|
    |       3|2099-03-26 00:00:00|               1.0|
    +--------+-------------------+------------------+
    <BLANKLINE>

    Last type is a linear smoothing: ``1 - c*age``.

    >>> get_item_recency(df, kind="linear").orderBy("item_idx").show()
    +--------+-------------------+------------------+
    |item_idx|          timestamp|         relevance|
    +--------+-------------------+------------------+
    |       1|2099-03-19 12:00:00|0.8916666666666666|
    |       2|2099-03-22 00:00:00|0.9333333333333333|
    |       3|2099-03-26 00:00:00|               1.0|
    +--------+-------------------+------------------+
    <BLANKLINE>

    This function **does not** take relevance values of interactions into account.
    Only item age is used.
    """
    log = convert2spark(log)
    items = log.select(
        "item_idx",
        sf.unix_timestamp(sf.to_timestamp("timestamp")).alias("timestamp"),
    )
    items = items.groupBy("item_idx").agg(
        sf.mean("timestamp").alias("timestamp")
    )
    items = items.withColumn("relevance", sf.lit(1))
    items = smoothe_time(items, decay, limit, kind)
    return items


def smoothe_time(
    log: AnyDataFrame,
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
    >>> d = {}
    >>> d["item_idx"] = [1, 1, 2, 3, 3]
    >>> d["timestamp"] = ["2099-03-19", "2099-03-20", "2099-03-22", "2099-03-27", "2099-03-25"]
    >>> d["relevance"] = [1, 1, 1, 1, 1]
    >>> df = pd.DataFrame(d)
    >>> df
       item_idx   timestamp  relevance
    0         1  2099-03-19          1
    1         1  2099-03-20          1
    2         2  2099-03-22          1
    3         3  2099-03-27          1
    4         3  2099-03-25          1

    Power smoothing falls quickly in the beginning but decays slowly afterwards as ``age^c``.

    >>> smoothe_time(df, kind="power").orderBy("timestamp").show()
    +--------+-------------------+------------------+
    |item_idx|          timestamp|         relevance|
    +--------+-------------------+------------------+
    |       1|2099-03-19 00:00:00|0.6390430306850825|
    |       1|2099-03-20 00:00:00| 0.654567945027101|
    |       2|2099-03-22 00:00:00|0.6940913454809814|
    |       3|2099-03-25 00:00:00|0.7994016704292545|
    |       3|2099-03-27 00:00:00|               1.0|
    +--------+-------------------+------------------+
    <BLANKLINE>

    Exponential smoothing is the other way around. Old objects decay more quickly as ``c^age``.

    >>> smoothe_time(df, kind="exp").orderBy("timestamp").show()
    +--------+-------------------+------------------+
    |item_idx|          timestamp|         relevance|
    +--------+-------------------+------------------+
    |       1|2099-03-19 00:00:00|0.8312378961427882|
    |       1|2099-03-20 00:00:00| 0.850667160950856|
    |       2|2099-03-22 00:00:00|0.8908987181403396|
    |       3|2099-03-25 00:00:00|0.9548416039104167|
    |       3|2099-03-27 00:00:00|               1.0|
    +--------+-------------------+------------------+
    <BLANKLINE>

    Last type is a linear smoothing: ``1 - c*age``.

    >>> smoothe_time(df, kind="linear").orderBy("timestamp").show()
    +--------+-------------------+------------------+
    |item_idx|          timestamp|         relevance|
    +--------+-------------------+------------------+
    |       1|2099-03-19 00:00:00|0.8666666666666667|
    |       1|2099-03-20 00:00:00|0.8833333333333333|
    |       2|2099-03-22 00:00:00|0.9166666666666666|
    |       3|2099-03-25 00:00:00|0.9666666666666667|
    |       3|2099-03-27 00:00:00|               1.0|
    +--------+-------------------+------------------+
    <BLANKLINE>

    These examples use constant relevance 1, so resulting weight equals the time dependent weight.
    But actually this value is an updated relevance.

    >>> d = {}
    >>> d["item_idx"] = [1, 2, 3]
    >>> d["timestamp"] = ["2099-03-19", "2099-03-20", "2099-03-22"]
    >>> d["relevance"] = [10, 3, 0.1]
    >>> df = pd.DataFrame(d)
    >>> df
       item_idx   timestamp  relevance
    0         1  2099-03-19       10.0
    1         2  2099-03-20        3.0
    2         3  2099-03-22        0.1
    >>> smoothe_time(df).orderBy("timestamp").show()
    +--------+-------------------+-----------------+
    |item_idx|          timestamp|        relevance|
    +--------+-------------------+-----------------+
    |       1|2099-03-19 00:00:00|9.330329915368075|
    |       2|2099-03-20 00:00:00| 2.86452481173125|
    |       3|2099-03-22 00:00:00|              0.1|
    +--------+-------------------+-----------------+
    <BLANKLINE>
    """
    log = convert2spark(log)
    log = log.withColumn(
        "timestamp", sf.unix_timestamp(sf.to_timestamp("timestamp"))
    )
    last_date = (
        log.agg({"timestamp": "max"}).collect()[0].asDict()["max(timestamp)"]
    )
    day_in_secs = 86400
    log = log.withColumn(
        "age", (last_date - sf.col("timestamp")) / day_in_secs
    )
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
        raise ValueError(
            f"parameter kind must be one of [power, exp, linear], got {kind}"
        )

    log = log.withColumn(
        "age", sf.when(sf.col("age") < limit, limit).otherwise(sf.col("age"))
    )
    log = log.withColumn(
        "relevance", sf.col("relevance") * sf.col("age")
    ).drop("age")
    log = log.withColumn("timestamp", sf.to_timestamp("timestamp"))
    return log
