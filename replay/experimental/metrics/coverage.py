from typing import Dict, Optional, Union

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, IntOrList, NumType, SparkDataFrame
from replay.utils.spark_utils import convert2spark

from .base_metric import RecOnlyMetric, process_k

if PYSPARK_AVAILABLE:
    from pyspark.sql import (
        Window,
        functions as sf,
    )


class Coverage(RecOnlyMetric):
    """
    Metric calculation is as follows:

    * take ``K`` recommendations with the biggest ``relevance`` for each ``user_id``
    * count the number of distinct ``item_id`` in these recommendations
    * divide it by the number of items in the whole data set

    """

    def __init__(self, log: DataFrameLike):
        """
        :param log: pandas or Spark DataFrame
                    It is important for ``log`` to contain all available items.
        """
        self.items = convert2spark(log).select("item_idx").distinct()
        self.item_count = self.items.count()

    @staticmethod
    def _get_metric_value_by_user(k, *args):
        # not averaged by users
        pass

    def _get_enriched_recommendations(
        self,
        recommendations: DataFrameLike,
        ground_truth: DataFrameLike,  # noqa: ARG002
        max_k: int,  # noqa: ARG002
        ground_truth_users: Optional[DataFrameLike] = None,
    ) -> SparkDataFrame:
        recommendations = convert2spark(recommendations)
        if ground_truth_users is not None:
            ground_truth_users = convert2spark(ground_truth_users)
            return recommendations.join(ground_truth_users, on="user_idx", how="inner")
        return recommendations

    def _conf_interval(
        self,
        recs: DataFrameLike,  # noqa: ARG002
        k_list: IntOrList,
        alpha: float = 0.95,  # noqa: ARG002
    ) -> Union[Dict[int, float], float]:
        if isinstance(k_list, int):
            return 0.0
        return {i: 0.0 for i in k_list}

    def _median(
        self,
        recs: DataFrameLike,
        k_list: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        return self._mean(recs, k_list)

    @process_k
    def _mean(
        self,
        recs: SparkDataFrame,
        k_list: list,
    ) -> Union[Dict[int, NumType], NumType]:
        unknown_item_count = recs.select("item_idx").distinct().exceptAll(self.items).count()
        if unknown_item_count > 0:
            self.logger.warning(
                "Recommendations contain items that were not present in the log. "
                r"The resulting metric value can be more than 1.0 ¯\_(ツ)_/¯"
            )

        best_positions = (
            recs.withColumn(
                "row_num",
                sf.row_number().over(Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))),
            )
            .select("item_idx", "row_num")
            .groupBy("item_idx")
            .agg(sf.min("row_num").alias("best_position"))
            .cache()
        )

        res = {}
        for current_k in k_list:
            res[current_k] = best_positions.filter(sf.col("best_position") <= current_k).count() / self.item_count

        best_positions.unpersist()
        return res
