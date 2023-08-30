from typing import Any, Dict, Optional

import pandas as pd

from replay.data import AnyDataFrame, IntOrList, NumType
from replay.utils.spark_utils import convert2spark
from replay.metrics.base_metric import (
    get_enriched_recommendations,
    Metric,
    NCISMetric,
    RecOnlyMetric,
)


# pylint: disable=too-few-public-methods
class Experiment:
    """
    This class calculates and stores metric values.
    Initialize it with test data and a dictionary mapping metrics to their depth cut-offs.

    Results are available with ``pandas_df`` attribute.

    Example:

    >>> import pandas as pd
    >>> from replay.metrics import Coverage, NDCG, Precision, Surprisal
    >>> log = pd.DataFrame({"user_idx": [2, 2, 2, 1], "item_idx": [1, 2, 3, 3], "relevance": [5, 5, 5, 5]})
    >>> test = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [1, 2, 3], "relevance": [5, 3, 4]})
    >>> pred = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [4, 1, 3], "relevance": [5, 4, 5]})
    >>> recs = pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [1, 4, 5], "relevance": [5, 4, 5]})
    >>> ex = Experiment(test, {NDCG(): [2, 3], Surprisal(log): 3})
    >>> ex.add_result("baseline", recs)
    >>> ex.add_result("baseline_gt_users", recs, ground_truth_users=pd.DataFrame({"user_idx": [1, 3]}))
    >>> ex.add_result("model", pred)
    >>> ex.results
                         NDCG@2    NDCG@3  Surprisal@3
    baseline           0.386853  0.296082     1.000000
    baseline_gt_users  0.193426  0.148041     0.500000
    model              0.386853  0.530721     0.666667
    >>> ex.compare("baseline")
                       NDCG@2  NDCG@3 Surprisal@3
    baseline                –       –           –
    baseline_gt_users  -50.0%  -50.0%      -50.0%
    model                0.0%  79.25%     -33.33%
    >>> ex = Experiment(test, {Precision(): [3]}, calc_median=True, calc_conf_interval=0.95)
    >>> ex.add_result("baseline", recs)
    >>> ex.add_result("model", pred)
    >>> ex.results
              Precision@3  Precision@3_median  Precision@3_0.95_conf_interval
    baseline     0.333333            0.333333                             0.0
    model        0.666667            0.666667                             0.0
    >>> ex = Experiment(test, {Coverage(log): 3}, calc_median=True, calc_conf_interval=0.95)
    >>> ex.add_result("baseline", recs)
    >>> ex.add_result("model", pred)
    >>> ex.results
              Coverage@3  Coverage@3_median  Coverage@3_0.95_conf_interval
    baseline         1.0                1.0                            0.0
    model            1.0                1.0                            0.0
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test: Any,
        metrics: Dict[Metric, IntOrList],
        calc_median: bool = False,
        calc_conf_interval: Optional[float] = None,
    ):
        """
        :param test: test DataFrame
        :param metrics: Dictionary of metrics to calculate.
            Key -- metric, value -- ``int`` or a list of ints.
        :param calc_median: flag to calculate median value across users
        :param calc_conf_interval: quantile value for the calculation of the confidence interval.
            Resulting value is the half of confidence interval.
        """
        self.test = convert2spark(test)
        self.results = pd.DataFrame()
        self.metrics = metrics
        self.calc_median = calc_median
        self.calc_conf_interval = calc_conf_interval

    def add_result(
        self,
        name: str,
        pred: AnyDataFrame,
        ground_truth_users: Optional[AnyDataFrame] = None,
    ) -> None:
        """
        Calculate metrics for predictions

        :param name: name of the run to store in the resulting DataFrame
        :param pred: model recommendations
        :param ground_truth_users: list of users to consider in metric calculation.
            if None, only the users from ground_truth are considered.
        """
        max_k = 0
        for current_k in self.metrics.values():
            max_k = max(
                (*current_k, max_k)
                if isinstance(current_k, list)
                else (current_k, max_k)
            )

        recs = get_enriched_recommendations(
            pred, self.test, max_k, ground_truth_users
        ).cache()
        for metric, k_list in sorted(
            self.metrics.items(), key=lambda x: str(x[0])
        ):
            enriched = None
            if isinstance(metric, (RecOnlyMetric, NCISMetric)):
                enriched = metric._get_enriched_recommendations(
                    pred, self.test, max_k, ground_truth_users
                )
            values, median, conf_interval = self._calculate(
                metric, enriched or recs, k_list
            )

            if isinstance(k_list, int):
                self._add_metric(  # type: ignore
                    name,
                    metric,
                    k_list,
                    values,  # type: ignore
                    median,  # type: ignore
                    conf_interval,  # type: ignore
                )
            else:
                for k, val in sorted(values.items(), key=lambda x: x[0]):
                    self._add_metric(
                        name,
                        metric,
                        k,
                        val,
                        None if median is None else median[k],
                        None if conf_interval is None else conf_interval[k],
                    )
        recs.unpersist()

    def _calculate(self, metric, enriched, k_list):
        median = None
        conf_interval = None
        values = metric._mean(enriched, k_list)
        if self.calc_median:
            median = metric._median(enriched, k_list)
        if self.calc_conf_interval is not None:
            conf_interval = metric._conf_interval(
                enriched, k_list, self.calc_conf_interval
            )
        return values, median, conf_interval

    # pylint: disable=too-many-arguments
    def _add_metric(
        self,
        name: str,
        metric: Metric,
        k: int,
        value: NumType,
        median: Optional[NumType],
        conf_interval: Optional[NumType],
    ):
        """
        Add metric for a specific k

        :param name: name to save results
        :param metric: metric object
        :param k: length of the recommendation list
        :param value: metric value
        :param median: median value
        :param conf_interval: confidence interval value
        """
        self.results.at[name, f"{metric}@{k}"] = value  # type: ignore
        if median is not None:
            self.results.at[
                name, f"{metric}@{k}_median"
            ] = median  # type: ignore
        if conf_interval is not None:
            self.results.at[
                name, f"{metric}@{k}_{self.calc_conf_interval}_conf_interval"
            ] = conf_interval

    # pylint: disable=not-an-iterable
    def compare(self, name: str) -> pd.DataFrame:
        """
        Show results as a percentage difference to record ``name``.

        :param name: name of the baseline record
        :return: results table in a percentage format
        """
        if name not in self.results.index:
            raise ValueError(f"No results for model {name}")
        columns = [
            column for column in self.results.columns if column[-1].isdigit()
        ]
        data_frame = self.results[columns].copy()
        baseline = data_frame.loc[name]
        for idx in data_frame.index:
            if idx != name:
                diff = data_frame.loc[idx] / baseline - 1
                data_frame.loc[idx] = [
                    str(round(v * 100, 2)) + "%" for v in diff
                ]
            else:
                data_frame.loc[name] = ["–"] * len(baseline)
        return data_frame
