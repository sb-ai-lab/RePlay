from replay.experimental.metrics.base_metric import ScalaMetric


# pylint: disable=too-few-public-methods
class ScalaRocAuc(ScalaMetric):
    """
    Receiver Operating Characteristic/Area Under the Curve is the aggregated performance measure,
    that depends only on the order of recommended items.
    It can be interpreted as the fraction of object pairs (object of class 1, object of class 0)
    that were correctly ordered by the model.
    The bigger the value of AUC, the better the classification model.

    .. math::
        ROCAUC@K(i) = \\frac {\sum_{s=1}^{K}\sum_{t=1}^{K}
        \mathbb{1}_{r_{si}<r_{ti}}
        \mathbb{1}_{gt_{si}<gt_{ti}}}
        {\sum_{s=1}^{K}\sum_{t=1}^{K} \mathbb{1}_{gt_{si}<gt_{tj}}}

    :math:`\\mathbb{1}_{r_{si}<r_{ti}}` -- indicator function showing that recommendation score for
    user :math:`i` for item :math:`s` is bigger than for item :math:`t`

    :math:`\mathbb{1}_{gt_{si}<gt_{ti}}` --  indicator function showing that
    user :math:`i` values item :math:`s` more than item :math:`t`.

    Metric is averaged by all users.

    .. math::
        ROCAUC@K = \\frac {\sum_{i=1}^{N}ROCAUC@K(i)}{N}
    """

    _scala_udf_name = "getRocAucMetricValue"
