"""Distribution calculations"""
from .types import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf

    from replay.utils.spark_utils import convert2spark, get_top_k_recs, spark_to_pandas


def item_distribution(
    log: DataFrameLike,
    recommendations: DataFrameLike,
    k: int,
    allow_collect_to_master: bool = False,
) -> PandasDataFrame:
    """
    Calculate item distribution in ``log`` and ``recommendations``.

    :param log: historical DataFrame used to calculate popularity
    :param recommendations: model recommendations
    :param k: length of a recommendation list
    :return: DataFrame with results
    """
    log = convert2spark(log)
    res = log.groupBy("item_idx").agg(sf.countDistinct("user_idx").alias("user_count")).select("item_idx", "user_count")

    rec = convert2spark(recommendations)
    rec = get_top_k_recs(rec, k)
    rec = rec.groupBy("item_idx").agg(sf.countDistinct("user_idx").alias("rec_count")).select("item_idx", "rec_count")

    res = res.join(rec, on="item_idx", how="outer").fillna(0).orderBy(["user_count", "item_idx"])
    return spark_to_pandas(res, allow_collect_to_master)
