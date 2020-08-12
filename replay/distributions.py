"""Построение распределений"""

import pandas as pd
import seaborn as sns
from pyspark.sql import functions as sf

from replay.constants import AnyDataFrame
from replay.utils import convert2spark, get_top_k_recs


def plot_user_dist(user_dist: pd.DataFrame, window: int = 1, title: str = ""):
    """
    Отрисовывает распределение значение метрики от количества оценок у пользователя

    :param user_dist: результат применения метода ``user_distribution`` у метрики
    :param window: какое количество ближайших значений усреднить
    :param title: название графика
    :return: график
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
):
    """
    Отрисовывает результат применения ``item_distribution``

    :param item_dist: ``pd.DataFrame``
    :param palette: цветовая схема
    :param col: по какой колонке строить
    :return: график
    """
    limits = list(range(len(item_dist), 0, -len(item_dist) // 10))[::-1]
    values = [(item_dist.iloc[:limit][col]).sum() for limit in limits]
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
    Посчитать количество вхождений айтемов в логах и рекомендациях.

    :param log: исторические данные для рассчета популярности
    :param recommendations: список рекомендаций для юзеров
    :param k: сколько рекомендаций брать
    :return: датафрейм с количеством вхождений
    """
    log = convert2spark(log)
    res = (
        log.groupBy("item_id")
        .agg(sf.countDistinct("user_id").alias("user_count"))
        .select("item_id", "user_count")
    )

    rec = convert2spark(recommendations)
    rec = get_top_k_recs(rec, k)
    rec = (
        rec.groupBy("item_id")
        .agg(sf.countDistinct("user_id").alias("rec_count"))
        .select("item_id", "rec_count")
    )

    res = (
        res.join(rec, on="item_id", how="outer")
        .fillna(0)
        .orderBy(["user_count", "item_id"])
        .toPandas()
    )
    return res
