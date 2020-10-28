Сценарии
==========
.. automodule:: replay.scenarios

Основной сценарий
-----------------
.. autoclass:: replay.scenarios.MainScenario
   :special-members: __init__
   :members:

Дополнение рекомендаций до необходимой длинны можно использовать и вне сценария.

.. autofunction:: replay.utils.fallback

Двухуровневый сценарий
----------------------
.. autoclass:: replay.scenarios.TwoStagesScenario
   :special-members: __init__
   :members: get_recs

Интеграция с `optuna`
----------------------
.. autoclass:: replay.scenarios.MainObjective
   :special-members: __call__
