from typing import List, Optional

from replay.utils import PandasDataFrame, SparkDataFrame

from .base_metric import Metric, MetricsDataFrameLike, MetricsReturnType


# pylint: disable=too-few-public-methods
class Unexpectedness(Metric):
    """
    Fraction of recommended items that are not present in some baseline\
        recommendations.

    .. math::
        Unexpectedness@K(i) = 1 -
            \\frac {\parallel R^{i}_{1..\min(K, \parallel R^{i} \parallel)} \cap BR^{i}_{1..\min(K, \parallel BR^{i} \parallel)} \parallel}
            {K}

    .. math::
        Unexpectedness@K = \\frac {1}{N}\sum_{i=1}^{N}Unexpectedness@K(i)

    :math:`R_{1..j}^{i}` -- the first :math:`j` recommendations for the :math:`i`-th user.

    :math:`BR_{1..j}^{i}` -- the first :math:`j` base recommendations for the :math:`i`-th user.

    >>> recommendations
       query_id  item_id  rating
    0         1        3    0.6
    1         1        7    0.5
    2         1       10    0.4
    3         1       11    0.3
    4         1        2    0.2
    5         2        5    0.6
    6         2        8    0.5
    7         2       11    0.4
    8         2        1    0.3
    9         2        3    0.2
    10        3        4    1.0
    11        3        9    0.5
    12        3        2    0.1
    >>> base_rec
       query_id  item_id  rating
    0        1        3    0.5
    1        1        7    0.5
    2        1        2    0.7
    3        2        5    0.6
    4        2        8    0.6
    5        2        3    0.3
    6        3        4    1.0
    7        3        9    0.5
    >>> from replay.metrics import Median, ConfidenceInterval, PerUser
    >>> Unexpectedness([2, 4])(recommendations, base_rec)
    {'Unexpectedness@2': 0.16666666666666666, 'Unexpectedness@4': 0.5}
    >>> Unexpectedness([2, 4], mode=PerUser())(recommendations, base_rec)
    {'Unexpectedness-PerUser@2': {1: 0.5, 2: 0.0, 3: 0.0},
     'Unexpectedness-PerUser@4': {1: 0.5, 2: 0.5, 3: 0.5}}
    >>> Unexpectedness([2, 4], mode=Median())(recommendations, base_rec)
    {'Unexpectedness-Median@2': 0.0, 'Unexpectedness-Median@4': 0.5}
    >>> Unexpectedness([2, 4], mode=ConfidenceInterval(alpha=0.95))(recommendations, base_rec)
    {'Unexpectedness-ConfidenceInterval@2': 0.32666066409000905,
     'Unexpectedness-ConfidenceInterval@4': 0.0}
    <BLANKLINE>
    """

    def _get_enriched_recommendations(  # pylint: disable=arguments-renamed
        self, recommendations: SparkDataFrame, base_recommendations: SparkDataFrame
    ) -> SparkDataFrame:
        sorted_by_score_recommendations = self._get_items_list_per_user(recommendations)

        sorted_by_score_base_recommendations = self._get_items_list_per_user(
            base_recommendations
        ).withColumnRenamed("pred_item_id", "base_pred_item_id")

        enriched_recommendations = sorted_by_score_recommendations.join(
            sorted_by_score_base_recommendations, how="left", on=self.query_column
        )

        return self._rearrange_columns(enriched_recommendations)

    def __call__(
        self,
        recommendations: MetricsDataFrameLike,
        base_recommendations: MetricsDataFrameLike,
    ) -> MetricsReturnType:
        """
        Compute metric.

        :param recommendations: (PySpark DataFrame or Pandas DataFrame or dict): model predictions.
            If DataFrame then it must contains user, item and score columns.
            If dict then key represents user_ids, value represents list of tuple(item_id, score).
        :param base_recommendations: (PySpark DataFrame or Pandas DataFrame or dict): base model predictions.
            If DataFrame then it must contains user, item and score columns.
            If dict then key represents user_ids, value represents list of tuple(item_id, score).

        :return: metric values
        """
        self._check_dataframes_equal_types(recommendations, base_recommendations)
        if isinstance(recommendations, SparkDataFrame):
            self._check_duplicates_spark(recommendations)
            self._check_duplicates_spark(base_recommendations)
            assert isinstance(base_recommendations, SparkDataFrame)
            return self._spark_call(recommendations, base_recommendations)
        recommendations = (
            self._convert_pandas_to_dict_with_score(recommendations)
            if isinstance(recommendations, PandasDataFrame)
            else self._convert_dict_to_dict_with_score(recommendations)
        )
        self._check_duplicates_dict(recommendations)
        assert isinstance(base_recommendations, (dict, PandasDataFrame))
        base_recommendations = (
            self._convert_pandas_to_dict_with_score(base_recommendations)
            if isinstance(base_recommendations, PandasDataFrame)
            else self._convert_dict_to_dict_with_score(base_recommendations)
        )
        self._check_duplicates_dict(base_recommendations)

        return self._dict_call(
            list(recommendations),
            recs=recommendations,
            base_recs=base_recommendations,
        )

    @staticmethod
    def _get_metric_value_by_user(  # pylint: disable=arguments-differ
        ks: List[int], base_recs: Optional[List], recs: Optional[List]
    ) -> List[float]:  # pragma: no cover
        if not base_recs or not recs:
            return [0.0 for _ in ks]
        res = []
        for k in ks:
            res.append(1.0 - len(set(recs[:k]) & set(base_recs[:k])) / k)
        return res
