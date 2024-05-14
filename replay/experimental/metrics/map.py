from .base_metric import Metric


class MAP(Metric):
    """
    Mean Average Precision -- average the ``Precision`` at relevant positions for each user,
        and then calculate the mean across all users.

    .. math::
        &AP@K(i) = \\frac 1K \\sum_{j=1}^{K}\\mathbb{1}_{r_{ij}}Precision@j(i)

        &MAP@K = \\frac {\\sum_{i=1}^{N}AP@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing if user :math:`i` interacted with item :math:`j`
    """

    _scala_udf_name = "getMAPMetricValue"

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        length = min(k, len(pred))
        if len(ground_truth) == 0 or len(pred) == 0:
            return 0
        tp_cum = 0
        result = 0
        for i in range(length):
            if pred[i] in ground_truth:
                tp_cum += 1
                result += tp_cum / (i + 1)
        return result / k
