Сценарии
==========
.. automodule:: replay.scenarios

Дополнение рекомендаций
---------------------------
.. autofunction:: replay.utils.fallback

Интеграция с `optuna`
----------------------
Нативные модели можно оптимизировать с помощью optuna встроенным методом

.. autoclass:: replay.models.base_rec.BaseRecommender
    :members: optimize

Двухуровневый сценарий
----------------------
.. autoclass:: replay.scenarios.TwoStagesScenario
   :special-members: __init__
   :members: get_recs
