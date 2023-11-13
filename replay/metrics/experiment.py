from typing import Dict, List, Optional, Union

import pandas as pd

from .base_metric import Metric, MetricsDataFrameLike
from .offline_metrics import OfflineMetrics


# pylint: disable=too-many-instance-attributes
# pylint: disable=too-few-public-methods
class Experiment:
    """
    The class is designed for calculating, storing and comparing metrics
    from different models in the Pandas DataFrame format.

    The main difference from the ``OfflineMetrics`` class is that
    ``OfflineMetrics`` are only responsible for calculating metrics.
    The ``Experiment`` class is responsible for storing metrics from different models,
    clear and their convenient comparison with each other.

    Calculated metrics are available with ``results`` attribute.

    Example:

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
    >>> groundtruth
       query_id  item_id
    0         1        5
    1         1        6
    2         1        7
    3         1        8
    4         1        9
    5         1       10
    6         2        6
    7         2        7
    8         2        4
    9         2       10
    10        2       11
    11        3        1
    12        3        2
    13        3        3
    14        3        4
    15        3        5
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
    >>> from replay.metrics import NDCG, Surprisal, Precision, Coverage, Median, ConfidenceInterval
    >>> ex = Experiment([NDCG([2, 3]), Surprisal(3)], groundtruth, train)
    >>> ex.add_result("baseline", base_rec)
    >>> ex.add_result("model", recommendations)
    >>> ex.results
                NDCG@2    NDCG@3  Surprisal@3
    baseline  0.204382  0.234639     0.608476
    model     0.333333  0.489760     0.719587
    >>> ex.compare("baseline")
              NDCG@2   NDCG@3 Surprisal@3
    baseline       –        –           –
    model     63.09%  108.73%      18.26%
    >>> ex = Experiment([Precision(3, mode=Median()), Precision(3, mode=ConfidenceInterval(0.95))], groundtruth)
    >>> ex.add_result("baseline", base_rec)
    >>> ex.add_result("model", recommendations)
    >>> ex.results
              Precision-Median@3  Precision-ConfidenceInterval@3
    baseline            0.333333                        0.217774
    model               0.666667                        0.217774
    <BLANKLINE>
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        metrics: List[Metric],
        ground_truth: MetricsDataFrameLike,
        train: Optional[MetricsDataFrameLike] = None,
        base_recommendations: Optional[
            Union[MetricsDataFrameLike, Dict[str, MetricsDataFrameLike]]
        ] = None,
        query_column: str = "query_id",
        item_column: str = "item_id",
        rating_column: str = "rating",
        category_column: str = "category_id",
    ):
        """
        :param metrics: (list of metrics): List of metrics to be calculated.
        :param ground_truth: (PySpark DataFrame or Pandas DataFrame or dict): test data.
            If DataFrame then it must contains user and item columns.
            If dict then key represents user_ids, value represents list of item_ids.
        :param train: (PySpark DataFrame or Pandas DataFrame or dict, optional): train data.
            If DataFrame then it must contains user and item columns.
            If dict then key represents user_ids, value represents list of item_ids.
            Default: ``None``.
        :param base_recommendations: (PySpark DataFrame or Pandas DataFrame or dict or Dict[str, DataFrameLike]):
            predictions from baseline model.
            If DataFrame then it must contains user, item and score columns.
            If dict then key represents user_ids, value represents list of tuple(item_id, score).
            If ``Unexpectedness`` is not in given metrics list, then you can omit this parameter.
            Default: ``None``.
        :param query_column: (str): The name of the user column.
            Note that you do not need to specify the value of this parameter for each metric separately.
            It is enough to specify the value of this parameter here once.
        :param item_column: (str): The name of the item column.
            Note that you do not need to specify the value of this parameter for each metric separately.
            It is enough to specify the value of this parameter here once.
        :param rating_column: (str): The name of the score column.
            Note that you do not need to specify the value of this parameter for each metric separately.
            It is enough to specify the value of this parameter here once.
        :param category_column: (str): The name of the category column.
            Note that you do not need to specify the value of this parameter for each metric separately.
            It is enough to specify the value of this parameter here once.

            It is used only for calculating the ``Diversity`` metric.
            If you don't calculate this metric, you can omit this parameter.
        """
        self._offline_metrics = OfflineMetrics(
            metrics=metrics,
            query_column=query_column,
            item_column=item_column,
            rating_column=rating_column,
            category_column=category_column,
        )
        self._ground_truth = ground_truth
        self._train = train
        self._base_recommendations = base_recommendations
        self.results = pd.DataFrame()

    def add_result(
        self,
        name: str,
        recommendations: MetricsDataFrameLike,
    ) -> None:
        """
        Calculate metrics for predictions

        :param name: name of the run to store in the resulting DataFrame
        :param recommendations: (PySpark DataFrame or Pandas DataFrame or dict): model predictions.
            If DataFrame then it must contains user, item and score columns.
            If dict then key represents user_ids, value represents list of tuple(item_id, score).
        """
        cur_metrics = self._offline_metrics(
            recommendations, self._ground_truth, self._train, self._base_recommendations
        )
        for metric, value in cur_metrics.items():
            self.results.at[name, metric] = value

    # pylint: disable=not-an-iterable
    def compare(self, name: str) -> pd.DataFrame:
        """
        Show results as a percentage difference to record ``name``.

        :param name: name of the baseline record
        :return: results table in a percentage format
        """
        if name not in self.results.index:
            raise ValueError(f"No results for model {name}")
        columns = [column for column in self.results.columns if column[-1].isdigit()]
        data_frame = self.results[columns].copy()
        baseline = data_frame.loc[name]
        for idx in data_frame.index:
            if idx != name:
                diff = data_frame.loc[idx] / baseline - 1
                data_frame.loc[idx] = [str(round(v * 100, 2)) + "%" for v in diff]
            else:
                data_frame.loc[name] = ["–"] * len(baseline)
        return data_frame
