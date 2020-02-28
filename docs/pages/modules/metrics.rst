Метрики
=======
.. automodule:: sponge_bob_magic.metrics

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

.. _new-metric:

Своя метрика
----------------------
Для добавления необходимо унаследоваться от класса ``Metric`` и реализовать/переопределить следующие методы

- **__init__**
- **_get_enriched_recommendations**
- **_get_metric_value_by_user**

Первые два метода уже реализованы, их стоит переопределять только в случае необходимости.
Последний метод необходимо реализовать для всех метрик.

.. autoclass:: sponge_bob_magic.metrics.Metric
   :special-members: _get_enriched_recommendations, _get_metric_value_by_user

