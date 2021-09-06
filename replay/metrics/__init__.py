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

Every metric is calculated using top ``K`` items for each user.
It is also possible to calculate metrics
using multiple values for ``K`` simultaneously.
In this case the result will be a dictionary and not a number.

- k (Union[Iterable[int], int]):
    a single number or a list, specifying the
    truncation length for recommendation list for each user

By default metrics are averaged by users,
but you can alternatively use method ``metric.median``.
Also you can get the lower bound
of ``conf_interval`` for a given ``alpha``.

Diversity metrics require extra parameters on initialization stage,
but do not use ``ground_truth`` parameter.
"""
from replay.metrics.base_metric import Metric
from replay.metrics.coverage import Coverage
from replay.metrics.hitrate import HitRate
from replay.metrics.map import MAP
from replay.metrics.mrr import MRR
from replay.metrics.ndcg import NDCG
from replay.metrics.precision import Precision
from replay.metrics.recall import Recall
from replay.metrics.rocauc import RocAuc
from replay.metrics.surprisal import Surprisal
from replay.metrics.unexpectedness import Unexpectedness
