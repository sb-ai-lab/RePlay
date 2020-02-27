metrics
========
.. automodule:: sponge_bob_magic.metrics

.. _HitRate:
.. autoclass:: sponge_bob_magic.metrics.HitRate

.. _Precision:
.. autoclass:: sponge_bob_magic.metrics.Precision

.. _MAP:
.. autoclass:: sponge_bob_magic.metrics.MAP

.. _Recall:
.. autoclass:: sponge_bob_magic.metrics.Recall

.. _NDCG:
.. autoclass:: sponge_bob_magic.metrics.NDCG

.. _Surprisal:
.. autoclass:: sponge_bob_magic.metrics.Surprisal
   :special-members: __init__

.. _new-metric:

Создание новой метрики
======================
Для создания новой метрики достаточно унаследоваться от класса ``Metric`` и реализовать/переопределить следующшие методы

- **__init__**
- **_get_enriched_recommendations**
- **_get_metric_value_by_user**

Первые два метода уже реализованы, их стоит переопределять только в случае необходимости.
Последний метод необходимо реализовать для всех метрик.

.. autoclass:: sponge_bob_magic.metrics.Metric
   :special-members: _get_enriched_recommendations, _get_metric_value_by_user

