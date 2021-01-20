"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as sf
from pyspark.sql import types as st

from replay.constants import AnyDataFrame
from replay.utils import convert2spark
from replay.metrics.base_metric import RecOnlyMetric


# pylint: disable=too-few-public-methods
class Unexpectedness(RecOnlyMetric):
    """
    Доля объектов в рекомендациях, которая не содержится в рекомендациях некоторого базового алгоритма.

    >>> from replay.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> import pandas as pd
    >>> log = pd.DataFrame({"user_id": [1, 1, 2, 3], "item_id": [1, 2, 1, 3], "relevance": [5, 5, 5, 5], "timestamp": [1, 1, 1, 1]})
    >>> recs = pd.DataFrame({"user_id": [1, 2, 1, 2], "item_id": [1, 2, 3, 1], "relevance": [5, 5, 5, 5], "timestamp": [1, 1, 1, 1]})
    >>> metric = Unexpectedness(log)
    >>> metric(recs,[1, 2])
    {1: 0.5, 2: 0.5}
    """

    def __init__(
        self, log: AnyDataFrame
    ):  # pylint: disable=super-init-not-called
        """
        Есть два варианта инициализации в зависимости от значения параметра ``rec``.
        Если ``rec`` -- рекомендатель, то ``log`` считается данными для обучения.
        Если ``rec is None``, то ``log`` считается готовыми предсказаниями какой-то внешней модели,
        с которой необходимо сравниться.

        :param log: пандас или спарк датафрейм
        :param rec: одна из проинициализированных моделей библиотеки, либо ``None``
        """
        self.log = convert2spark(log)

    @staticmethod
    def _get_metric_value_by_user(k, *args) -> float:
        pred = args[0]
        base_pred = args[1]
        if len(pred) == 0:
            return 0
        return 1.0 - len(set(pred[:k]) & set(base_pred[:k])) / k

    def _get_enriched_recommendations(
        self, recommendations: DataFrame, ground_truth: DataFrame
    ) -> DataFrame:
        base_pred = self.log
        sort_udf = sf.udf(
            self._sorter,
            returnType=st.ArrayType(base_pred.schema["item_id"].dataType),
        )
        base_recs = (
            base_pred.groupby("user_id")
            .agg(
                sf.collect_list(sf.struct("relevance", "item_id")).alias(
                    "base_pred"
                )
            )
            .select(
                "user_id", sort_udf(sf.col("base_pred")).alias("base_pred")
            )
        )

        recommendations = (
            recommendations.groupby("user_id")
            .agg(
                sf.collect_list(sf.struct("relevance", "item_id")).alias(
                    "pred"
                )
            )
            .select("user_id", sort_udf(sf.col("pred")).alias("pred"))
            .join(base_recs, how="left", on=["user_id"])
        )
        return recommendations.withColumn(
            "base_pred",
            sf.coalesce(
                "base_pred",
                sf.array().cast(
                    st.ArrayType(base_pred.schema["item_id"].dataType)
                ),
            ),
        )
