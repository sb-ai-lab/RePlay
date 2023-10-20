"""
Most metrics require dataframe with recommendations
and dataframe with ground truth values —
which objects each user interacted with.

- recommendations (Union[pandas.DataFrame, spark.DataFrame]):
    predictions of a recommender system,
    DataFrame with columns ``[user_id, item_id, relevance]``
- ground_truth (Union[pandas.DataFrame, spark.DataFrame]):
    test data, DataFrame with columns
    ``[user_id, item_id, timestamp, relevance]``

Metric is calculated for all users, presented in ``ground_truth``
for accurate metric calculation in case when the recommender system generated
recommendation not for all users.  It is assumed, that all users,
we want to calculate metric for, have positive interactions.

But if we have users, who observed the recommendations, but have not responded,
those users will be ignored and metric will be overestimated.
For such case we propose additional optional parameter ``ground_truth_users``,
the dataframe with all users, which should be considered during the metric calculation.

- ground_truth_users (Optional[Union[pandas.DataFrame, spark.DataFrame]]):
    full list of users to calculate metric for, DataFrame with ``user_id`` column

Every metric is calculated using top ``K`` items for each user.
It is also possible to calculate metrics
using multiple values for ``K`` simultaneously.
In this case the result will be a dictionary and not a number.

Make sure your recommendations do not contain user-item duplicates
as duplicates could lead to the wrong calculation results.

- k (Union[Iterable[int], int]):
    a single number or a list, specifying the
    truncation length for recommendation list for each user

By default, metrics are averaged by users,
but you can alternatively use method ``metric.median``.
Also, you can get the lower bound
of ``conf_interval`` for a given ``alpha``.

Diversity metrics require extra parameters on initialization stage,
but do not use ``ground_truth`` parameter.

For each metric, a formula for its calculation is given, because this is
important for the correct comparison of algorithms, as mentioned in our
`article <https://arxiv.org/abs/2206.12858>`_.
"""
from .base_metric import Metric
from .coverage import Coverage
from .experiment import Experiment
from .hitrate import HitRate
from .map import MAP
from .mrr import MRR
from .ncis_precision import NCISPrecision
from .ndcg import NDCG
from .precision import Precision
from .recall import Recall
from .rocauc import RocAuc
from .surprisal import Surprisal
from .unexpectedness import Unexpectedness

__all__ = [
    "Metric",
    "Coverage",
    "Experiment",
    "HitRate",
    "MAP",
    "MRR",
    "NCISPrecision",
    "NDCG",
    "Precision",
    "Recall",
    "RocAuc",
    "Surprisal",
    "Unexpectedness",
]
