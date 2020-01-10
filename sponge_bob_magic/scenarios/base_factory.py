from abc import ABC, abstractmethod

from pyspark.sql import SparkSession

from sponge_bob_magic.scenarios.base_scenario import Scenario


class ScenarioFactory(ABC):
    """ Базовый класс фабрики. """
    
    def __init__(self, spark: SparkSession, **kwargs):
        """
        Инициализирует фабрику нужными параметрами и сохраняет спарк-сессию.

        :param spark: инициализированная спарк-сессия
        :param kwargs: дополнительные параметры классов-наследников
        """
        self.spark = spark

    @abstractmethod
    def get(self) -> Scenario:
        """
        Основной метод, который должен быть имплементирован наследниками.
        Возвращает инициализированный сценарий.

        :return: инициализированный объект класса Scenario
        """
