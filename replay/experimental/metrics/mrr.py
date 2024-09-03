from .base_metric import Metric


class MRR(Metric):
    """
    Mean Reciprocal Rank --
    Reciprocal Rank is the inverse position of the first relevant item among top-k recommendations,
    :math:`\\frac {1}{rank_i}`. This value is averaged by all users.
    """

    _scala_udf_name = "getMRRMetricValue"

    @staticmethod
    def _get_metric_value_by_user(k, pred, ground_truth) -> float:
        for i in range(min(k, len(pred))):
            if pred[i] in ground_truth:
                return 1 / (1 + i)
        return 0
