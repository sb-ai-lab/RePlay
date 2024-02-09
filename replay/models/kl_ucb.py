import math

from typing import Optional
from .ucb import UCB
from replay.utils import PYSPARK_AVAILABLE
from scipy.optimize import root_scalar

if PYSPARK_AVAILABLE:
    from pyspark.sql.types import DoubleType
    from pyspark.sql.functions import udf


class KLUCB(UCB):
    """
    Bernoulli `bandit model
    <https://en.wikipedia.org/wiki/Multi-armed_bandit>`_. Same to :class:`UCB`
    computes item relevance as an upper confidence bound of true fraction of
    positive interactions.

    In a nutshell, KL-UCB сonsiders the data as the history of interactions
    with items. The interaction may be either positive or negative. For each
    item the model computes empirical frequency of positive interactions
    and estimates the true frequency with an upper confidence bound. The higher
    the bound for an item is the more relevant it is presumed.

    The upper bound below is what differs from the classical UCB. It is
    computed according to the `original article
    <https://arxiv.org/pdf/1102.2490.pdf>`_ where is proven to produce
    assymptotically better results.

    .. math::
        u_i = \\max q \\in [0,1] :
        n_i \\cdot \\operatorname{KL}\\left(\\frac{p_i}{n_i}, q \\right)
        \\leqslant \\log(n) + c \\log(\\log(n)),

    where

    :math:`u_i` -- upper bound for item :math:`i`,

    :math:`c` -- exploration coeficient,

    :math:`n` -- number of interactions in log,

    :math:`n_i` -- number of interactions with item :math:`i`,

    :math:`p_i` -- number of positive interactions with item :math:`i`,

    and

    .. math::
        \\operatorname{KL}(p, q)
        = p \\log\\frac{p}{q} + (1-p)\\log\\frac{1-p}{1-q}

    is the KL-divergence of Bernoulli distribution with parameter :math:`p`
    from Bernoulli distribution with parameter :math:`q`.

    Being a bit trickier though, the bound shares with UCB the same
    exploration-exploitation tradeoff dilemma. You may increase the `c`
    coefficient in order to shift the tradeoff towards exploration or decrease
    it to set the model to be more sceptical of items with small volume of
    collected statistics. The authors of the `article
    <https://arxiv.org/pdf/1102.2490.pdf>`_ though claim `c = 0` to be of the
    best choice in practice.


    As any other RePlay model, KL-UCB takes a log to fit on as a ``DataFrame``
    with columns ``[user_idx, item_idx, timestamp, relevance]``. Following the
    procedure above, KL-UCB would see each row as a record of an interaction
    with ``item_idx`` with positive (relevance = 1) or negative (relevance = 0)
    outcome. ``user_idx`` and ``timestamp`` are ignored i.e. the model treats
    log as non-personalized - item scores are same for all users.

    If ``relevance`` column is not of 0/1 initially, then you have to decide
    what kind of relevance has to be considered as positive and convert
    ``relevance`` to binary format during preprocessing.

    To provide a prediction, KL-UCB would sample a set of recommended items for
    each user with probabilites proportional to obtained relevances.

    >>> import pandas as pd
    >>> from replay.data.dataset import Dataset, FeatureSchema, FeatureInfo, FeatureHint, FeatureType
    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = pd.DataFrame({"user_id": [1, 2, 3, 3], "item_id": [1, 2, 1, 2], "rating": [1, 0, 0, 0]})
    >>> interactions = convert2spark(data_frame)
    >>> feature_schema = FeatureSchema(
    ...     [
    ...         FeatureInfo(
    ...             column="user_id",
    ...             feature_type=FeatureType.CATEGORICAL,
    ...             feature_hint=FeatureHint.QUERY_ID,
    ...         ),
    ...         FeatureInfo(
    ...             column="item_id",
    ...             feature_type=FeatureType.CATEGORICAL,
    ...             feature_hint=FeatureHint.ITEM_ID,
    ...         ),
    ...         FeatureInfo(
    ...             column="rating",
    ...             feature_type=FeatureType.NUMERICAL,
    ...             feature_hint=FeatureHint.RATING,
    ...         ),
    ...     ]
    ... )
    >>> dataset = Dataset(feature_schema, interactions)
    >>> model = KLUCB()
    >>> model.fit(dataset)
    >>> model.predict(dataset, k=2, queries=[1,2,3,4], items=[1,2,3]
    ... ).toPandas().sort_values(["user_id","rating","item_id"],
    ... ascending=[True,False,True]).reset_index(drop=True)
        user_id   item_id     rating
    0         1         3   1.000000
    1         1	        2   0.750000
    2         2	        3   1.000000
    3         2	        1   0.933013
    4         3	        3   1.000000
    5         4	        3   1.000000
    6         4	        1   0.933013

    """

    def __init__(
        self,
        exploration_coef: float = 0.0,
        sample: bool = False,
        seed: Optional[int] = None,
    ):
        """
        :param exploration_coef: exploration coefficient
        :param sample: flag to choose recommendation strategy.
            If True, items are sampled with a probability proportional
            to the calculated predicted relevance.
            Could be changed after model training by setting the `sample`
            attribute.
        :param seed: random seed. Provides reproducibility if fixed
        """

        super().__init__(exploration_coef, sample, seed)

    def _calc_item_popularity(self):

        right_hand_side = math.log(self.full_count) \
            + self.coef * math.log(math.log(self.full_count))
        eps = 1e-12

        def bernoulli_kl(proba_p, proba_q):  # pragma: no cover
            return proba_p * math.log(proba_p / proba_q) +\
                (1 - proba_p) * math.log((1 - proba_p) / (1 - proba_q))

        @udf(returnType=DoubleType())
        def get_ucb(pos, total):  # pragma: no cover
            proba = pos / total

            if proba == 0:
                ucb = root_scalar(
                    f=lambda qq: math.log(1 / (1 - qq)) - right_hand_side,
                    bracket=[0, 1 - eps],
                    method='brentq').root
                return ucb

            if proba == 1:
                ucb = root_scalar(
                    f=lambda qq: math.log(1 / qq) - right_hand_side,
                    bracket=[0 + eps, 1],
                    method='brentq').root
                return ucb

            ucb = root_scalar(
                f=lambda q: total * bernoulli_kl(proba, q) - right_hand_side,
                bracket=[proba, 1 - eps],
                method='brentq').root
            return ucb

        items_counts = self.items_counts_aggr.withColumn(
            self.rating_column, get_ucb("pos", "total")
        )

        self.item_popularity = items_counts.drop("pos", "total")

        self.item_popularity.cache().count()
        self.fill = 1 + math.sqrt(self.coef * math.log(self.full_count))
