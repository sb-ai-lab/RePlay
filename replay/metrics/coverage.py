"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from typing import Dict, Set, Union

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
        unknows_item_count = (
            recommendations_spark.select("item_id")  # type: ignore
            .distinct()
            .exceptAll(self.items)
            .count()
        )
        if unknows_item_count > 0:
            self.logger.warning(
                "В рекомендациях есть объекты, которых не было в изначальном логе! "
                "Значение метрики может получиться больше единицы ¯\_(ツ)_/¯"
            )
        coverage_spark = (
            recommendations_spark.withColumn(  # type: ignore
                "row_num",
                sf.row_number().over(
                    Window.partitionBy("user_id").orderBy(sf.desc("relevance"))
                ),
            )
            .withColumn(
                "cum_cov",
                sf.size(
                    sf.collect_set("item_id").over(
                        Window.orderBy("row_num").rowsBetween(
                            Window.unboundedPreceding, Window.currentRow
                        )
                    )
                )
                / self.item_count,
            )
            .groupBy("row_num")
            .agg(sf.max("cum_cov").alias("coverage"))
        ).cache()

        if isinstance(k, int):
            k_set: Set[int] = {k}
        else:
            k_set: Set[int] = set(k)

        res = (
            coverage_spark.filter(coverage_spark.row_num.isin(k_set))
            .toPandas()
            .set_index("row_num")
            .to_dict()["coverage"]
        )

        if not res:
            return self.unpack_if_int(
                {current_k: 0.0 for current_k in k_set}, k
            )

        if len(k_set) > len(res.keys()):
            max_coverage = coverage_spark.agg({"coverage": "max"}).collect()[
                0
            ][0]
            for current_k in k_set.difference(res.keys()):
                res[current_k] = max_coverage

        return self.unpack_if_int(res, k)
