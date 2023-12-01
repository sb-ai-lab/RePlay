Metrics
==================================

.. automodule:: replay.metrics

You can also
:ref:`add new metrics <new-metric>`.


Precision
----------------------
.. autoclass:: replay.metrics.Precision
    :members: __init__, __call__


.. _recall:

Recall
----------------------
.. autoclass:: replay.metrics.Recall
    :members: __init__, __call__


MAP
----------------------
.. autoclass:: replay.metrics.MAP
    :members: __init__, __call__


MRR
----------------------
.. autoclass:: replay.metrics.MRR
    :members: __init__, __call__


NDCG
----------------------
.. autoclass:: replay.metrics.NDCG
    :members: __init__, __call__


HitRate
----------------------
.. autoclass:: replay.metrics.HitRate
    :members: __init__, __call__


RocAuc
----------------------
.. autoclass:: replay.metrics.RocAuc
    :members: __init__, __call__


Unexpectedness
----------------------
.. autoclass:: replay.metrics.Unexpectedness
    :members: __init__, __call__


Coverage
----------------------
.. autoclass:: replay.metrics.Coverage
    :members: __init__, __call__


CategoricalDiversity
----------------------
.. autoclass:: replay.metrics.CategoricalDiversity
    :members: __init__, __call__


Novelty
----------------------
.. autoclass:: replay.metrics.Novelty
    :members: __init__, __call__


Surprisal
----------------------
.. autoclass:: replay.metrics.Surprisal


OfflineMetrics
----------------------
.. autoclass:: replay.metrics.OfflineMetrics
    :members: __init__, __call__


Compare Results
----------------------
.. autoclass:: replay.metrics.Experiment
   :members: __init__, add_result, compare


----------------------

.. _new-metric:

Custom Metric
----------------------
Your metric should be inherited from ``Metric`` class and implement following methods:

- **__init__**
- **_get_metric_value_by_user**

``_get_metric_value_by_user`` is required for every metric because this is where the actual calculations happen.
For a better understanding, see already implemented metrics, for example :ref:`Recall <recall>`.

.. autoclass:: replay.metrics.base_metric.Metric
   :special-members: _get_metric_value_by_user