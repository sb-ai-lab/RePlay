"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.constants import AnyDataFrame
from replay.utils import convert2spark
from replay.metrics.base_metric import RecOnlyMetric


# pylint: disable=too-few-public-methods
class Unexpectedness(RecOnlyMetric):
    """
    Доля объектов в рекомендациях, которая не содержится в рекомендациях некоторого базового алгоритма.

    >>> import pandas as pd
    >>> from replay.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> log = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [1, 2, 3], "relevance": [5, 5, 5], "timestamp": [1, 1, 1]})
    >>> recs = pd.DataFrame({"user_id": [1, 1, 1], "item_id": [0, 0, 1], "relevance": [5, 5, 5], "timestamp": [1, 1, 1]})
    >>> metric = Unexpectedness(log)
    >>> round(metric(recs, 3), 2)
    0.67
    """

    def __init__(
        self, pred: AnyDataFrame
    ):  # pylint: disable=super-init-not-called
        """
        :param pred: предсказания модели, относительно которых необходимо посчитать метрику.
        :param rec: одна из проинициализированных моделей библиотеки, либо ``None``
        """
        self.pred = convert2spark(pred)

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        recs = pandas_df["item_id"]
        pandas_df["cum_agg"] = pandas_df.apply(
            lambda row: (
                row["k"]
                - np.isin(recs[: row["k"]], row["items_id"][: row["k"]]).sum()
            )
            / row["k"],
            axis=1,
        )
        return pandas_df

    def _get_enriched_recommendations(
        self, recommendations: DataFrame, ground_truth: DataFrame
    ) -> DataFrame:
        items_by_users = self.pred.groupby("user_id").agg(
            sf.collect_list("item_id").alias("items_id")
        )
        res = recommendations.join(items_by_users, how="inner", on=["user_id"])
        return res
