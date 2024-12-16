from .base_metric import Metric


class Precision(Metric):
    """
    Mean percentage of relevant items among top ``K`` recommendations.

    .. math::
        Precision@K(i) = \\frac {\\sum_{j=1}^{K}\\mathbb{1}_{r_{ij}}}{K}

    .. math::
        Precision@K = \\frac {\\sum_{i=1}^{N}Precision@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing that user :math:`i` interacted with item :math:`j`"""

    _scala_udf_name = "getPrecisionMetricValue"

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        if len(pred) == 0:
            return 0
        return len(set(pred[:k]) & set(ground_truth)) / k
