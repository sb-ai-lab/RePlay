from replay.metrics import Unexpectedness

from .base_metric import ScalaRecOnlyMetric


# pylint: disable=too-few-public-methods
class ScalaUnexpectedness(Unexpectedness, ScalaRecOnlyMetric):
    """
    Fraction of recommended items that are not present in some baseline recommendations.
    """

    _scala_udf_name = "getUnexpectednessMetricValue"
