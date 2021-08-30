.. _metrics:

Metrics
========
.. automodule:: replay.metrics

You can also
:ref:`add new metrics <new-metric>`.

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

.. _new-metric:

----------------------

Custom Metric
----------------------
Your metric should be inherited from ``Metric`` class and implement following methods:

- **__init__**
- **_get_enriched_recommendations**
- **_get_metric_value_by_user**

``get_enriched_recommendations`` is already implemented, but you can change it if it is required for your metric.
``_get_metric_value_by_user`` is required for every metric because this is where the actual calculations happen.

.. autofunction:: replay.metrics.base_metric.get_enriched_recommendations

.. autoclass:: replay.metrics.base_metric.Metric
   :special-members: _get_metric_value_by_user

.. autoclass:: replay.metrics.base_metric.RecOnlyMetric
