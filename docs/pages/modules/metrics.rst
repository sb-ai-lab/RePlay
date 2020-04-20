.. _metrics:

Метрики
=======
.. automodule:: sponge_bob_magic.metrics

.. _hit-rate:

HitRate
--------
.. autoclass:: sponge_bob_magic.metrics.HitRate

Precision
---------
.. autoclass:: sponge_bob_magic.metrics.Precision

MAP
---
.. autoclass:: sponge_bob_magic.metrics.MAP

Recall
------
.. autoclass:: sponge_bob_magic.metrics.Recall

ROC-AUC
-------
.. autoclass:: sponge_bob_magic.metrics.RocAuc

MRR
----
.. autoclass:: sponge_bob_magic.metrics.MRR

NDCG
-----
.. autoclass:: sponge_bob_magic.metrics.NDCG

Surprisal
----------
.. autoclass:: sponge_bob_magic.metrics.Surprisal
   :special-members: __init__

Unexpectedness
---------------
.. autoclass:: sponge_bob_magic.metrics.Unexpectedness
   :special-members: __init__

Coverage
---------
.. autoclass:: sponge_bob_magic.metrics.Coverage
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

.. autoclass:: sponge_bob_magic.metrics.base_metric.Metric
   :special-members: _get_enriched_recommendations, _get_metric_value_by_user

