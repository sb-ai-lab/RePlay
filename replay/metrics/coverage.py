from typing import Dict, Optional, Union

from pyspark.sql import Window, DataFrame
from pyspark.sql import functions as sf

from replay.data import AnyDataFrame, IntOrList, NumType
from replay.utils.spark_utils import convert2spark
from replay.metrics.base_metric import RecOnlyMetric, process_k


# pylint: disable=too-few-public-methods, arguments-differ, unused-argument
class Coverage(RecOnlyMetric):
    """
    Metric calculation is as follows:

    * take ``K`` recommendations with the biggest ``relevance`` for each ``user_id``
    * count the number of distinct ``item_id`` in these recommendations
    * divide it by the number of items in the whole data set

    """

    def __init__(
        self,
        interactions: AnyDataFrame,
        query_column: str,
        item_column: str,
        rating_column: str,
    ):  # pylint: disable=super-init-not-called
        """
        :param log: pandas or Spark DataFrame
                    It is important for ``log`` to contain all available items.
        """
        self.query_column = query_column
        self.item_column = item_column
        self.rating_column = rating_column
        self.items = (
            convert2spark(interactions).select(self.item_column).distinct()  # type: ignore
        )
        self.item_count = self.items.count()

    @staticmethod
    def _get_metric_value_by_user(k, *args):    # pragma: no cover
        # not averaged by users
        pass

    # pylint: disable=no-self-use
    def _get_enriched_recommendations(
        self,
        recommendations: AnyDataFrame,
        ground_truth: AnyDataFrame,
        max_k: int,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> DataFrame:
        recommendations = convert2spark(recommendations)
        if ground_truth_users is not None:
            ground_truth_users = convert2spark(ground_truth_users)
            return recommendations.join(
                ground_truth_users, on=self.query_column, how="inner"
            )
        return recommendations

    def _conf_interval(
        self,
        recs: AnyDataFrame,
        k_list: IntOrList,
        alpha: float = 0.95,
    ) -> Union[Dict[int, float], float]:
        if isinstance(k_list, int):
            return 0.0
        return {i: 0.0 for i in k_list}

    def _median(
        self,
        recs: AnyDataFrame,
        k_list: IntOrList,
    ) -> Union[Dict[int, NumType], NumType]:
        return self._mean(recs, k_list)

    @process_k
    def _mean(
        self,
        recs: DataFrame,
        k_list: list,
    ) -> Union[Dict[int, NumType], NumType]:
        unknown_item_count = (
            recs.select(self.item_column)  # type: ignore
            .distinct()
            .exceptAll(self.items)
            .count()
        )
        if unknown_item_count > 0:
            self.logger.warning(
                "Recommendations contain items that were not present in the log. "
                "The resulting metric value can be more than 1.0 ¯\_(ツ)_/¯"
            )

        best_positions = (
            recs.withColumn(
                "row_num",
                sf.row_number().over(
                    Window.partitionBy(self.query_column).orderBy(
                        sf.desc(self.rating_column)
                    )
                ),
            )
            .select(self.item_column, "row_num")
            .groupBy(self.item_column)
            .agg(sf.min("row_num").alias("best_position"))
            .cache()
        )

        res = {}
        for current_k in k_list:
            res[current_k] = (
                best_positions.filter(
                    sf.col("best_position") <= current_k
                ).count()
                / self.item_count
            )

        best_positions.unpersist()
        return res
