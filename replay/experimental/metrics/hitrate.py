from .base_metric import Metric


class HitRate(Metric):
    """
    Percentage of users that have at least one
        correctly recommended item among top-k.

    .. math::
        HitRate@K(i) = \\max_{j \\in [1..K]}\\mathbb{1}_{r_{ij}}

    .. math::
        HitRate@K = \\frac {\\sum_{i=1}^{N}HitRate@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function stating that user :math:`i` interacted with item :math:`j`

    """

    _scala_udf_name = "getHitRateMetricValue"

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        for i in pred[:k]:
            if i in ground_truth:
                return 1
        return 0
