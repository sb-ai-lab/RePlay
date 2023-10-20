from replay.experimental.metrics.base_metric import ScalaRecOnlyMetric
from replay.metrics import Unexpectedness


# pylint: disable=too-few-public-methods
class ScalaUnexpectedness(Unexpectedness, ScalaRecOnlyMetric):
    """
    Fraction of recommended items that are not present in some baseline recommendations.
    """

    _scala_udf_name = "getUnexpectednessMetricValue"
