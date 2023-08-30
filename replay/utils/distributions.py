"""Distribution calculations"""

import pandas as pd
import seaborn as sns
from pyspark.sql import functions as sf

from replay.data import AnyDataFrame
from replay.utils.spark_utils import convert2spark, get_top_k_recs


def plot_user_dist(
    user_dist: pd.DataFrame, window: int = 1, title: str = ""
):  # pragma: no cover
    """
    Plot mean metric value by the number of user ratings

    :param user_dist: output of ``user_distribution`` method for a metric
    :param window: the number of closest values to average for smoothing
    :param title: plot title
    :return: plot object
    """
    user_dist["smoothed"] = (
        user_dist["value"].rolling(window, center=True).mean()
    )
    plot = sns.lineplot(x="count", y="smoothed", data=user_dist)
    plot.set(
        xlabel="# of ratings",
        ylabel="smoothed value",
        title=title,
        xscale="log",
    )
    return plot


def plot_item_dist(
    item_dist: pd.DataFrame, palette: str = "magma", col: str = "rec_count"
):  # pragma: no cover
    """
    Show the results of  ``item_distribution`` method

    :param item_dist: ``pd.DataFrame``
    :param palette: colour scheme for seaborn
    :param col: column to use for a plot
    :return: plot
    """
    limits = list(range(len(item_dist), 0, -len(item_dist) // 10))[::-1]
    values = [(item_dist.iloc[:limit][col]).sum() for limit in limits]
    # pylint: disable=too-many-function-args
    plot = sns.barplot(
        list(range(1, 11)),
        values / max(values),
        palette=sns.color_palette(palette, 10),
    )
    plot.set(
        xlabel="popularity decentile",
        ylabel="proportion",
        title="Popularity distribution",
    )
    return plot


def item_distribution(
    log: AnyDataFrame, recommendations: AnyDataFrame, k: int
) -> pd.DataFrame:
    """
    Calculate item distribution in ``log`` and ``recommendations``.

    :param log: historical DataFrame used to calculate popularity
    :param recommendations: model recommendations
    :param k: length of a recommendation list
    :return: DataFrame with results
    """
    log = convert2spark(log)
    res = (
        log.groupBy("item_idx")
        .agg(sf.countDistinct("user_idx").alias("user_count"))
        .select("item_idx", "user_count")
    )

    rec = convert2spark(recommendations)
    rec = get_top_k_recs(rec, k)
    rec = (
        rec.groupBy("item_idx")
        .agg(sf.countDistinct("user_idx").alias("rec_count"))
        .select("item_idx", "rec_count")
    )

    res = (
        res.join(rec, on="item_idx", how="outer")
        .fillna(0)
        .orderBy(["user_count", "item_idx"])
        .toPandas()
    )
    return res
