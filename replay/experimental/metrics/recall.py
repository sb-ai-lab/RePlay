from .base_metric import Metric


class Recall(Metric):
    """
    Mean percentage of relevant items, that was shown among top ``K`` recommendations.

    .. math::
        Recall@K(i) = \\frac {\\sum_{j=1}^{K}\\mathbb{1}_{r_{ij}}}{|Rel_i|}

    .. math::
        Recall@K = \\frac {\\sum_{i=1}^{N}Recall@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing that user :math:`i` interacted with item :math:`j`

    :math:`|Rel_i|` -- the number of relevant items for user :math:`i`
    """

    _scala_udf_name = "getRecallMetricValue"

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        if len(ground_truth) == 0:
            return 0.0
        return len(set(pred[:k]) & set(ground_truth)) / len(ground_truth)
