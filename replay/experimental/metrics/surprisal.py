from replay.experimental.metrics.base_metric import ScalaRecOnlyMetric
from replay.metrics import Surprisal


# pylint: disable=too-few-public-methods
class ScalaSurprisal(Surprisal, ScalaRecOnlyMetric):
    """
    Measures how many surprising rare items are present in recommendations.

    .. math::
        \\textit{Self-Information}(j)= -\log_2 \\frac {u_j}{N}

    :math:`u_j` -- number of users that interacted with item :math:`j`.
    Cold items are treated as if they were rated by 1 user.
    That is, if they appear in recommendations it will be completely unexpected.

    Metric is normalized.

    Surprisal for item :math:`j` is

    .. math::
        Surprisal(j)= \\frac {\\textit{Self-Information}(j)}{log_2 N}

    Recommendation list surprisal is the average surprisal of items in it.

    .. math::
        Surprisal@K(i) = \\frac {\sum_{j=1}^{K}Surprisal(j)} {K}

    Final metric is averaged by users.

    .. math::
        Surprisal@K = \\frac {\sum_{i=1}^{N}Surprisal@K(i)}{N}
    """

    _scala_udf_name = "getSurprisalMetricValue"
