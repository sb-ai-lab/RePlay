from typing import TYPE_CHECKING, List, Type

from replay.utils import PandasDataFrame, SparkDataFrame, PolarsDataFrame

from .base_metric import Metric, MetricsDataFrameLike, MetricsReturnType

if TYPE_CHECKING:  # pragma: no cover
    __class__: Type


# pylint: disable=too-few-public-methods
class Novelty(Metric):
    """
    Measure the fraction of shown items in recommendation list, that users\
        didn't see in train dataset.

    .. math::
        Novelty@K(i) = \\frac
        {\parallel {R^{i}_{1..\min(K, \parallel R^{i} \parallel)} \setminus train^{i}} \parallel}
        {K}

    .. math::
        Novelty@K = \\frac {1}{N}\sum_{i=1}^{N}Novelty@K(i)

    :math:`R^{i}` -- the recommendations for the :math:`i`-th user.

    :math:`R^{i}_{j}` -- the :math:`j`-th recommended item for the :math:`i`-th user.

    :math:`R_{1..j}^{i}` -- the first :math:`j` recommendations for the :math:`i`-th user.

    :math:`train^{i}` -- the train items of the :math:`i`-th user.

    :math:`N` -- the number of users.

    :Based on:

        P. Castells, S. Vargas, and J. Wang, Novelty and diversity metrics for recommender systems:
        choice, discovery and relevance, ECIR 2011.
        `Link <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=ecb23352b3fd8abd32332790fda7aca59c498fdf>`_.

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
    >>> Novelty(2)(recommendations, train)
    {'Novelty@2': 0.3333333333333333}
    >>> Novelty(2, mode=PerUser())(recommendations, train)
    {'Novelty-PerUser@2': {1: 1.0, 2: 0.0, 3: 0.0}}
    >>> Novelty(2, mode=Median())(recommendations, train)
    {'Novelty-Median@2': 0.0}
    >>> Novelty(2, mode=ConfidenceInterval(alpha=0.95))(recommendations, train)
    {'Novelty-ConfidenceInterval@2': 0.6533213281800181}
    <BLANKLINE>
    """

    def __call__(
        self,
        recommendations: MetricsDataFrameLike,
        train: MetricsDataFrameLike,
    ) -> MetricsReturnType:
        """
        Compute metric.

        :param recommendations: (PySpark DataFrame or Polars DataFrame or Pandas DataFrame or dict):
            model predictions.
            If DataFrame then it must contains user, item and score columns.
            If dict then items must be sorted in decreasing order of their scores.
        :param train: (PySpark DataFrame or Polars DataFrame or Pandas DataFrame or dict, optional):
            train data.
            If DataFrame then it must contains user and item columns.

        :return: metric values
        """
        self._check_dataframes_equal_types(recommendations, train)
        if isinstance(recommendations, SparkDataFrame):
            self._check_duplicates_spark(recommendations)
            assert isinstance(train, SparkDataFrame)
            return self._spark_call(recommendations, train)
        if isinstance(recommendations, PolarsDataFrame):
            self._check_duplicates_polars(recommendations)
            assert isinstance(train, PolarsDataFrame)
            return self._polars_call(recommendations, train)
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

        return self._dict_call(
            list(train),
            pred_item_id=recommendations,
            train=train,
        )

    # pylint: disable=arguments-renamed
    def _spark_call(
        self, recommendations: SparkDataFrame, train: SparkDataFrame
    ) -> MetricsReturnType:
        """
        Implementation for Pyspark DataFrame.
        """
        recs = self._get_enriched_recommendations(
            recommendations, train
        ).withColumnRenamed("ground_truth", "train")
        recs = self._rearrange_columns(recs)
        return self._spark_compute(recs)

    # pylint: disable=arguments-renamed
    def _polars_call(
        self, recommendations: PolarsDataFrame, train: PolarsDataFrame
    ) -> MetricsReturnType:
        """
        Implementation for Polars DataFrame.
        """
        recs = self._get_enriched_recommendations(
            recommendations, train
        ).rename({"ground_truth": "train"})
        recs = self._rearrange_columns(recs)
        return self._polars_compute(recs)

    # pylint: disable=arguments-differ
    @staticmethod
    def _get_metric_value_by_user(
        ks: List[int], pred: List, train: List
    ) -> List[float]:
        if not train or not pred:
            return [1.0 for _ in ks]
        set_train = set(train)
        res = []
        for k in ks:
            res.append(1.0 - len(set(pred[:k]) & set_train) / len(pred[:k]))
        return res
