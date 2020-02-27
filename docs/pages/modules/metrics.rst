Метрики
=======
.. automodule:: sponge_bob_magic.metrics

HitRate
--------
.. _HitRate:
.. autoclass:: sponge_bob_magic.metrics.HitRate

Precision
---------
.. _Precision:
.. autoclass:: sponge_bob_magic.metrics.Precision

MAP
---
.. _MAP:
.. autoclass:: sponge_bob_magic.metrics.MAP

Recall
------
.. _Recall:
.. autoclass:: sponge_bob_magic.metrics.Recall

nDCG
-----
.. _NDCG:
.. autoclass:: sponge_bob_magic.metrics.NDCG

Surprisal
----------
.. _Surprisal:
.. autoclass:: sponge_bob_magic.metrics.Surprisal
   :special-members: __init__

.. _new-metric:

Своя метрика
----------------------
Для добавления необходимо унаследоваться от класса ``Metric`` и реализовать/переопределить следующшие методы

- **__init__**
- **_get_enriched_recommendations**
- **_get_metric_value_by_user**

Первые два метода уже реализованы, их стоит переопределять только в случае необходимости.
Последний метод необходимо реализовать для всех метрик.

.. autoclass:: sponge_bob_magic.metrics.Metric
   :special-members: _get_enriched_recommendations, _get_metric_value_by_user

