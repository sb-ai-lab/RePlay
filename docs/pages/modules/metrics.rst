.. _metrics:

Metrics
========
.. automodule:: replay.metrics

You can also
:ref:`add new metrics <new-metric>`.


Metric call API
---------------
.. autoclass:: replay.metrics.Metric
   :special-members: __call__


.. _hit-rate:

HitRate
--------
.. autoclass:: replay.metrics.HitRate

Precision
---------
.. autoclass:: replay.metrics.Precision

MAP
---
.. autoclass:: replay.metrics.MAP

Recall
------
.. autoclass:: replay.metrics.Recall

ROC-AUC
-------
.. autoclass:: replay.metrics.RocAuc

MRR
----
.. autoclass:: replay.metrics.MRR

NDCG
-----
.. autoclass:: replay.metrics.NDCG

Surprisal
----------
.. autoclass:: replay.metrics.Surprisal
   :special-members: __init__

Unexpectedness
---------------
.. autoclass:: replay.metrics.Unexpectedness
   :special-members: __init__

Coverage
---------
.. autoclass:: replay.metrics.Coverage
   :special-members: __init__

NCIS metrics
------------

RePlay implements Normalized Capped Importance Sampling for metric calculation with ``NCISMetric`` class.
This method is mostly applied to RL-based recommendation systems to perform counterfactual evaluation, but could be
used for any kind of recommender systems. See an article
`Offline A/B testing for Recommender Systems <http://arxiv.org/abs/1801.07030>`_ for details.

*Reward* (metric value for a user-item pair) is weighed by
the ratio of *current policy score* (current relevance) on *previous policy score* (historical relevance).

The *weight* is clipped by the *threshold* and put into interval :math:`[\frac{1}{threshold}, threshold]`.
Activation function (e.g. softmax, sigmoid) could be applied to the scores before weights calculation.

Normalization weight for recommended item is calculated as follows:

.. math::
    w_{ui} = \frac{f(\pi^t_ui, pi^t_u)}{f(\pi^p_ui, pi^p_u)}

Where:

:math:`\pi^t_{ui}` - current policy value (predicted relevance) of the user-item interaction

:math:`\pi^p_{ui}` - previous policy value (historical relevance) of the user-item interaction.
Only values for user-item pairs present in current recommendations are used for calculation.

:math:`\pi_u` - all predicted /historical policy values for selected user :math:`u`

:math:`f(\pi_{ui}, \pi_u)` - activation function applied to policy values (optional)

:math:`w_{ui}` - weight of user-item interaction for normalized metric calculation before clipping


Calculated weights are clipped as follows:

.. math::
    \hat{w_{ui}} = min(max(\frac{1}{threshold}, w_{ui}), threshold)

Normalization metric value for a user is calculated as follows:

.. math::
    R_u = \frac{r_{ui} \hat{w_{ui}}}{\sum_{i}\hat{w_{ui}}}

Where:

:math:`r_ui` - metric value (reward) for user-item interaction

:math:`R_u` - metric value (reward) for user :math:`u`

Weight calculation is implemented in ``_get_enriched_recommendations`` method.

.. autoclass:: replay.metrics.base_metric.NCISMetric
   :special-members: __init__, _get_enriched_recommendations

NCISPrecision metric
^^^^^^^^^^^^^^^^^^^^^
The NCISPrecision is currently only one NCIS metric implemented in RePlay.

.. autoclass:: replay.metrics.NCISPrecision
   :special-members: _get_metric_value_by_user

.. _new-metric:

----------------------

Custom Metric
----------------------
Your metric should be inherited from ``Metric`` or ``NCISMetric`` class and implement following methods:

- **__init__**
- **_get_enriched_recommendations**
- **_get_metric_value_by_user**

``get_enriched_recommendations`` is already implemented, but you can change it if it is required for your metric.
``_get_metric_value_by_user`` is required for every metric because this is where the actual calculations happen.

.. autofunction:: replay.metrics.base_metric.get_enriched_recommendations

.. autoclass:: replay.metrics.base_metric.Metric
   :special-members: _get_metric_value_by_user

.. autoclass:: replay.metrics.base_metric.RecOnlyMetric

----------------------

Compare Results
----------------------

.. autoclass:: replay.metrics.experiment.Experiment
   :members:
   :special-members: __init__
