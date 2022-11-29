import numpy as np

from replay.metrics.base_metric import NCISMetric


# pylint: disable=too-few-public-methods
class NCISPrecision(NCISMetric):
    """
    Share of relevant items among top ``K`` recommendations with NCIS weighting.

    .. math::
        Precision@K(i) = \\frac {\sum_{j=1}^{K}\mathbb{1}_{r_{ij} w_{ij}}}{\sum_{j=1}^{K} w_{ij}}

    .. math::
        Precision@K = \\frac {\sum_{i=1}^{N}Precision@K(i)}{N}

    :math:`\\mathbb{1}_{r_{ij}}` -- indicator function
        showing that user :math:`i` interacted with item :math:`j`
    :math:`w_{ij}` -- NCIS weight, calculated as ratio of current policy score on previous
        policy score with clipping and optional activation over policy scores (relevance).
        Source: arxiv.org/abs/1801.07030
    """

    @staticmethod
    def _get_metric_value_by_user(k, *args):
        pred, ground_truth, pred_weights = args
        if len(pred) == 0:
            return 0
        mask = np.isin(pred[:k], ground_truth)
        return sum(np.array(pred_weights)[mask]) / sum(pred_weights[:k])
