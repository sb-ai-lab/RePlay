"""
Most metrics require dataframe with recommendations
and dataframe with ground truth values —
which objects each user interacted with.
All input dataframes must have the same type.

- recommendations (Union[pandas.DataFrame, spark.DataFrame, Dict]):
    If the recommendations is instance of dict then key represents user_id, value represents tuple(item_id, score).
    If the recommendations is instance of Spark or Pandas dataframe
    then the names of the corresponding columns should be passed through the constructor of metric.
- ground_truth (Union[pandas.DataFrame, spark.DataFrame]):
    If ground_truth is instance of dict then key represents user_id, value represents item_id.
    If the recommendations is instance of Spark or Pandas dataframe
    then the names of the corresponding columns must match the recommendations.

Metric is calculated for all users, presented in ``ground_truth``
for accurate metric calculation in case when the recommender system generated
recommendation not for all users.  It is assumed, that all users,
we want to calculate metric for, have positive interactions.

Every metric is calculated using top ``K`` items for each user.
It is also possible to calculate metrics
using multiple values for ``K`` simultaneously.

Make sure your recommendations do not contain user-item duplicates
as duplicates could lead to the wrong calculation results.

- k (Union[Iterable[int], int]):
    a single number or a list, specifying the
    truncation length for recommendation list for each user

By default, metrics are averaged by users - ``replay.metrics.Mean``
but you can alternatively use ``replay.metrics.Median``.
You can get the median value of the confidence interval -
``replay.metrics.ConfidenceInterval`` for a given ``alpha``.
To calculate the metric value for each user separately in most metrics there is a parameter ``per user``.

To write your own aggregation kernel,
you need to inherit from the ``replay.metrics.CalculationDescriptor`` and redefine two methods (``spark``, ``cpu``).

For each metric, a formula for its calculation is given, because this is
important for the correct comparison of algorithms, as mentioned in our
`article <https://arxiv.org/abs/2206.12858>`_.
"""

from .base_metric import Metric
from .categorical_diversity import CategoricalDiversity
from .coverage import Coverage
from .descriptors import CalculationDescriptor, ConfidenceInterval, Mean, Median, PerUser
from .experiment import Experiment
from .hitrate import HitRate
from .map import MAP
from .mrr import MRR
from .ndcg import NDCG
from .novelty import Novelty
from .offline_metrics import OfflineMetrics
from .precision import Precision
from .recall import Recall
from .rocauc import RocAuc
from .surprisal import Surprisal
from .unexpectedness import Unexpectedness
