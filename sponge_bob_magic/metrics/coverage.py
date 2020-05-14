"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
import logging
from typing import Dict, Set, Union

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf

from sponge_bob_magic.constants import AnyDataFrame, IntOrList, NumType
from sponge_bob_magic.converter import convert
from sponge_bob_magic.metrics.base_metric import RecOnlyMetric


# pylint: disable=too-few-public-methods
class Coverage(RecOnlyMetric):
    """
    Метрика вычисляется так:

    * берём ``K`` рекомендаций с наибольшей ``relevance`` для каждого ``user_id``
    * считаем, сколько всего различных ``item_id`` встречается в отобранных рекомендациях
    * делим полученное количество объектов в рекомендациях на количество объектов в изначальном логе (до разбиения на train и test)

    """

    def __init__(self, log: AnyDataFrame):
        """
        :param log: pandas или Spark DataFrame, содержащий лог *до* разбиения на train и test.
                    Важно, чтобы log содержал все доступные объекты (items). Coverage будет рассчитываться как доля по отношению к ним.
        """
        self.items = convert(log).select("item_id").distinct().cache()
        self.item_count = self.items.count()
        self.logger = logging.getLogger("sponge_bob_magic")

    @staticmethod
    def _get_metric_value_by_user(pandas_df):
        # эта метрика не является средним по всем пользователям
        pass

    def conf_interval(
        self,
        recommendations: DataFrame,
        ground_truth: DataFrame,
        k: IntOrList,
        alpha: float = 0.95,
    ) -> Union[Dict[int, NumType], NumType]:
        if isinstance(k, int):
            return 0.0
        else:
            return {i: 0.0 for i in k}

    def median(
        self, recommendations: DataFrame, ground_truth: DataFrame, k: IntOrList
    ) -> Union[Dict[int, NumType], NumType]:
        return self.mean(recommendations, ground_truth, k)

    def mean(
        self, recommendations: DataFrame, ground_truth: DataFrame, k: IntOrList
    ) -> Union[Dict[int, NumType], NumType]:
        if isinstance(k, int):
            k_set = {k}
        else:
            k_set = set(k)
        unknows_item_count = (
            recommendations.select("item_id")
            .distinct()
            .exceptAll(self.items)
            .count()
        )
        if unknows_item_count > 0:
            self.logger.warning(
                "В рекомендациях есть объекты, которых не было в изначальном логе! "
                "Значение метрики может получиться больше единицы ¯\_(ツ)_/¯"
            )
        item_sets = (
            recommendations.withColumn(
                "row_num",
                sf.row_number().over(
                    Window.partitionBy("user_id").orderBy(sf.desc("relevance"))
                ),
            )
            .groupBy("row_num")
            .agg(sf.collect_set("item_id").alias("items"))
            .orderBy("row_num")
        ).collect()
        cum_set: Set[str] = set()
        res = {}
        for row in item_sets:
            cum_set = cum_set.union(set(row.items))
            if row.row_num in k_set:
                res[row.row_num] = len(cum_set) / self.item_count
        if isinstance(k, int):
            res = res[k]
        return res
