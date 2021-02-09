import logging
from typing import Dict, Union

from pyspark.sql import Window
from pyspark.sql import functions as sf

from replay.constants import AnyDataFrame, IntOrList, NumType
from replay.utils import convert2spark
from replay.metrics.base_metric import RecOnlyMetric


# pylint: disable=too-few-public-methods
class Coverage(RecOnlyMetric):
    """
    Метрика вычисляется так:

    * берём ``K`` рекомендаций с наибольшей ``relevance`` для каждого ``user_id``
    * считаем, сколько всего различных ``item_id`` встречается в отобранных рекомендациях
    * делим полученное количество объектов в рекомендациях на количество объектов в изначальном логе (до разбиения на train и test)

    """

    def __init__(
        self, log: AnyDataFrame
    ):  # pylint: disable=super-init-not-called
        """
        :param log: pandas или Spark DataFrame, содержащий лог *до* разбиения на train и test.
                    Важно, чтобы log содержал все доступные объекты (items). Coverage будет рассчитываться как доля по отношению к ним.
        """
        self.items = (
            convert2spark(log)
            .select("item_id")
            .distinct()  # type: ignore
            .cache()
        )
        self.item_count = self.items.count()
        self.logger = logging.getLogger("replay")

    @staticmethod
    def _get_metric_value_by_user(k, *args):
        # эта метрика не является средним по всем пользователям
        pass

    def conf_interval(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
        alpha: float = 0.95,
    ) -> Union[Dict[int, float], float]:
        if isinstance(k, int):
            return 0.0
        return {i: 0.0 for i in k}

    def median(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        return self.mean(recommendations, ground_truth, k)

    def mean(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        k: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        recommendations_spark = convert2spark(recommendations)
        unknown_item_count = (
            recommendations_spark.select("item_id")  # type: ignore
            .distinct()
            .exceptAll(self.items)
            .count()
        )
        if unknown_item_count > 0:
            self.logger.warning(
                "В рекомендациях есть объекты, которых не было в изначальном логе! "
                "Значение метрики может получиться больше единицы ¯\_(ツ)_/¯"
            )

        best_positions = (
            recommendations_spark.withColumn(
                "row_num",
                sf.row_number().over(
                    Window.partitionBy("user_id").orderBy(sf.desc("relevance"))
                ),
            )
            .select("item_id", "row_num")
            .groupBy("item_id")
            .agg(sf.min("row_num").alias("best_position"))
            .cache()
        )

        if isinstance(k, int):
            k_set = {k}
        else:
            k_set = set(k)

        res = {}
        for current_k in k_set:
            res[current_k] = (
                best_positions.filter(
                    sf.col("best_position") <= current_k
                ).count()
                / self.item_count
            )

        best_positions.unpersist()
        return self.unpack_if_int(res, k)
