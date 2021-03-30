.. _metrics:

Метрики
=======
.. automodule:: replay.metrics

.. csv-table::
   :header: "Метрика", "Описание", "Тип метрики", "Дополнительные параметры"
   :widths: 1, 2, 1, 2
   :width: 10em

    "HitRate@K", "Доля пользователей, для которых хотя бы одна рекомендация из первых K релевантна", "Классификация", ""
    "Precision@K", "Доля релевантных рекомендаций среди первых K элементов выдачи", "Классификация", ""
    "Mean Average Precision, MAP@K", "Средний Precision по всем j <= K элементам выдачи", "Ранжирование", ""
    "Recall@K", "Доля объектов из тестовой выборки, вошедших в первые K рекомендаций алгоритма", "Классификация", ""
    "ROC-AUC@K", "Доля правильно упорядоченных (релевантный / нерелевантный) пар объектов среди рекомендованных", "Ранжирование", ""
    "Mean Reciprocal Rank, MRR@K", "Средняя обратная позиция первой релевантной рекомендации из K элементов выдачи", "Ранжирование", ""
    "Normalized Discounted Cumulative Gain, NDCG@K", "Метрика ранжирования, учитывающая позиции релевантных объектов среди первых K элементов выдачи", "Ранжирование", ""
    "Surprisal@K", "Степень 'редкости' (непопулярности) рекомендуемых объектов", "Разнообразие", "Лог взаимодействия для определения степени редкости объектов"
    "Unexpectedness@K", "Доля рекомендуемых объектов, которые не содержатся в рекомендациях базового алгоритма", "Разнообразие", "Базовый алгоритм и лог взаимодействия для обучения или рекомендации базового алгоритма"
    "Coverage@K", "Доля объектов, которые встречаются в полученных рекомендациях", "Разнообразие", "Лог взаимодействия, содержащий все доступные объекты"

Если реализованных метрик недостаточно, библиотека поддерживает возможность
:ref:`добавления своих метрик <new-metric>`.

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

Своя метрика
----------------------
Для добавления необходимо унаследоваться от класса ``Metric`` и реализовать/переопределить следующие методы

- **__init__**
- **_get_enriched_recommendations**
- **_get_metric_value_by_user**

``get_enriched_recommendations`` уже реализован в виде функции, и его стоит переопределять методом только в случае необходимости.
Последний метод необходимо реализовать для всех метрик, так как в нём происходит основное вычисление метрики.

.. autofunction:: replay.metrics.base_metric.get_enriched_recommendations

.. autoclass:: replay.metrics.base_metric.Metric
   :special-members: _get_metric_value_by_user

.. autoclass:: replay.metrics.base_metric.RecOnlyMetric
