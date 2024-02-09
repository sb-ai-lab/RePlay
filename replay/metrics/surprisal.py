from collections import defaultdict
from typing import Dict, List

import numpy as np

from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame

from .base_metric import Metric, MetricsDataFrameLike, MetricsReturnType

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


# pylint: disable=too-few-public-methods
class Surprisal(Metric):
    """
    Measures how many surprising rare items are present in recommendations.

    .. math::
        \\textit{Self-Information}(j)= -\log_2 \\frac {u_j}{N}

    :math:`u_j` -- number of users that interacted with item :math:`j`.
    Cold items are treated as if they were rated by 1 user.
    That is, if they appear in recommendations it will be completely unexpected.

    Surprisal for item :math:`j` is

    .. math::
        Surprisal(j)= \\frac {\\textit{Self-Information}(j)}{log_2 N}

    Recommendation list surprisal is the average surprisal of items in it.

    .. math::
        Surprisal@K(i) = \\frac {\sum_{j=1}^{K}Surprisal(j)} {K}

    Final metric is averaged by users.

    .. math::
        Surprisal@K = \\frac {\sum_{i=1}^{N}Surprisal@K(i)}{N}

    :math:`N` -- the number of users.

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
    >>> train
       query_id  item_id
    0         1        5
    1         1        6
    2         1        8
    3         1        9
    4         1        2
    5         2        5
    6         2        8
    7         2       11
    8         2        1
    9         2        3
    10        3        4
    11        3        9
    12        3        2
    >>> from replay.metrics import Median, ConfidenceInterval, PerUser
    >>> Surprisal(2)(recommendations, train)
    {'Surprisal@2': 0.6845351232142715}
    >>> Surprisal(2, mode=PerUser())(recommendations, train)
    {'Surprisal-PerUser@2': {1: 1.0, 2: 0.3690702464285426, 3: 0.6845351232142713}}
    >>> Surprisal(2, mode=Median())(recommendations, train)
    {'Surprisal-Median@2': 0.6845351232142713}
    >>> Surprisal(2, mode=ConfidenceInterval(alpha=0.95))(recommendations, train)
    {'Surprisal-ConfidenceInterval@2': 0.3569755541728279}
    <BLANKLINE>
    """

    # pylint: disable=no-self-use
    def _get_weights(self, train: Dict) -> Dict:
        n_users = len(train.keys())
        items_counter = defaultdict(set)
        for user, items in train.items():
            for item in items:
                items_counter[item].add(user)
        weights = {}
        for item, users in items_counter.items():
            weights[item] = np.log2(n_users / len(users)) / np.log2(n_users)
        return weights

    def _get_recommendation_weights(self, recommendations: Dict, train: Dict) -> Dict:
        weights = self._get_weights(train)
        recs_with_weights = {}
        for user, items in recommendations.items():
            recs_with_weights[user] = [weights.get(i, 1) for i in items]
        return recs_with_weights

    def _get_enriched_recommendations(  # pylint: disable=arguments-renamed
        self, recommendations: SparkDataFrame, train: SparkDataFrame
    ) -> SparkDataFrame:
        n_users = train.select(self.query_column).distinct().count()
        item_weights = train.groupby(self.item_column).agg(
            (
                sf.log2(n_users / sf.countDistinct(self.query_column)) / np.log2(n_users)
            ).alias("weight")
        )
        recommendations = recommendations.join(
            item_weights, on=self.item_column, how="left"
        ).fillna(1.0)

        sorted_by_score_recommendations = self._get_items_list_per_user(
            recommendations, "weight"
        )
        return self._rearrange_columns(sorted_by_score_recommendations)

    def __call__(
        self,
        recommendations: MetricsDataFrameLike,
        train: MetricsDataFrameLike,
    ) -> MetricsReturnType:
        """
        Compute metric.

        Args:
            recommendations (PySpark DataFrame or Pandas DataFrame or dict): model predictions.
                If DataFrame then it must contains user, item and score columns.
                If dict then items must be sorted in decreasing order of their scores.
            train (PySpark DataFrame or Pandas DataFrame or dict, optional): train data.
                If DataFrame then it must contains user and item columns.

        Returns:
            dict: metric values
        """
        self._check_dataframes_equal_types(recommendations, train)
        if isinstance(recommendations, SparkDataFrame):
            self._check_duplicates_spark(recommendations)
            assert isinstance(train, SparkDataFrame)
            return self._spark_call(recommendations, train)
        is_pandas = isinstance(recommendations, PandasDataFrame)
        recommendations = (
            self._convert_pandas_to_dict_with_score(recommendations)
            if is_pandas
            else self._convert_dict_to_dict_with_score(recommendations)
        )
        self._check_duplicates_dict(recommendations)
        train = (
            self._convert_pandas_to_dict_without_score(train) if is_pandas else train
        )
        assert isinstance(train, dict)

        weights = self._get_recommendation_weights(recommendations, train)
        return self._dict_call(
            list(train),
            pred_item_id=recommendations,
            pred_weight=weights,
        )

    @staticmethod
    def _get_metric_value_by_user(  # pylint: disable=arguments-differ
        ks: List[int], pred_item_ids: List, pred_weights: List
    ) -> List[float]:
        if not pred_item_ids:
            return [0.0 for _ in ks]
        res = []
        for k in ks:
            ans = 0
            for weight, _ in zip(pred_weights[:k], pred_item_ids[:k]):
                ans += weight
            res.append(ans / k)
        return res
