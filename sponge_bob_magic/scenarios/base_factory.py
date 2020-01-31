"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from abc import ABC, abstractmethod

from sponge_bob_magic.scenarios.base_scenario import Scenario


class ScenarioFactory(ABC):
    """ Базовый класс фабрики. """

    @abstractmethod
    def get(self,  **kwargs) -> Scenario:
        """
        Основной метод, который должен быть имплементирован наследниками.
        Возвращает инициализированный параметрами сценарий.

        :param kwargs: параметры, которыми инициализируется сценарий
        :return: инициализированный объект класса Scenario
        """
