.. _metrics:

Метрики
=======
.. automodule:: replay.metrics

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

Своя метрика
----------------------
Для добавления необходимо унаследоваться от класса ``Metric`` и реализовать/переопределить следующие методы

- **__init__**
- **_get_enriched_recommendations**
- **_get_metric_value_by_user**

``_get_enriched_recommendations`` уже реализован, и его стоит переопределять только в случае необходимости.
Последний метод необходимо реализовать для всех метрик, так как в нём происходит основное вычисление метрики.

.. autoclass:: replay.metrics.base_metric.Metric
   :special-members: _get_enriched_recommendations, _get_metric_value_by_user
