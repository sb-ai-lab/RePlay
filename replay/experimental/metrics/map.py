from replay.experimental.metrics.base_metric import ScalaMetric


# pylint: disable=too-few-public-methods
class ScalaMAP(ScalaMetric):
    """
    Mean Average Precision -- average the ``Precision`` at relevant positions for each user,
        and then calculate the mean across all users.

    .. math::
        &AP@K(i) = \\frac 1K \sum_{j=1}^{K}\mathbb{1}_{r_{ij}}Precision@j(i)

        &MAP@K = \\frac {\sum_{i=1}^{N}AP@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function showing if user :math:`i` interacted with item :math:`j`
    """

    _scala_udf_name = "getMAPMetricValue"
