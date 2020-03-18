"""
Сценарий --- сущность, объединяющая в себе основные этапы создания рекомендательной системы:

* :ref:`разбиение данных на обучающую и валидационную выборки <splitters>`
* подбор гипер-параметров с помощью `optuna <https://optuna.org/>`_
* :ref:`расчёт метрик качества для полученных моделей-кандидатов <metrics>`
* обучение на всём объёме данных с подобранными гипер-параметрами и отгрузка рекомендаций (batch production)

Перед использованием сценария необходимо :ref:`перевести свои данные во внутренний формат библиотеки <data-preparator>`.
"""
from sponge_bob_magic.scenarios.main_objective import MainObjective
from sponge_bob_magic.scenarios.main_scenario import MainScenario
