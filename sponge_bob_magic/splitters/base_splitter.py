"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from abc import abstractmethod, ABC
from typing import Tuple

from pyspark.sql import DataFrame, SparkSession

SplitterReturnType = Tuple[DataFrame, DataFrame, DataFrame]


class Splitter(ABC):
    """ Базовый класс для разбиения выборки на обучающую и тестовую. """

    def __init__(
            self,
            spark: SparkSession,
            drop_cold_items: bool = False,
            drop_cold_users: bool = False,
            **kwargs
    ):
        """
        :param spark: инициализированная спарк-сессия
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
           которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
           которых нет в обучающей
        """
        self.drop_cold_users = drop_cold_users
        self.drop_cold_items = drop_cold_items
        self.spark = spark

    @staticmethod
    def _filter_zero_relevance(dataframe: DataFrame) -> DataFrame:
        """
        Удаляет записи с нулевой релевантностью (нужно для тестовой выборки).

        :param dataframe: входной набор данных стандартного формата
        :returns: набор данных той же структуры, но без записей с нулевой
        релевантностью
        """
        return dataframe.filter("relevance > 0.0")

    @staticmethod
    def _drop_cold_items_and_users(
            train: DataFrame,
            test: DataFrame,
            drop_cold_items: bool,
            drop_cold_users: bool
    ) -> DataFrame:
        """
        Удаляет из тестовой выборки холодных users и холодные items.

        :param train: обучающая выборка, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, context, relevance]`
        :param test: тестовая выборка как train
        :param drop_cold_items: если True, удаляет холодные items
        :param drop_cold_users: если True, удаляет холодные users
        :return: тестовая выборка без холодных users / items
        """
        if drop_cold_items:
            test = test.join(
                train.select("item_id").distinct(),
                how="inner",
                on="item_id"
            )

        if drop_cold_users:
            test = test.join(
                train.select("user_id").distinct(),
                how="inner",
                on="user_id"
            )
        return test

    @abstractmethod
    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        """
        Внутренний метод, который должны имплеменитровать классы-наследники.

        Разбивает лог действий пользователей на обучающую и тестовую выборки.

        :param log: лог взаимодействия, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, context, relevance]`
        :returns: тройка спарк-датафреймов структуры, аналогичной входной,
            `train, predict_input, test`, где `train` - обучающая выборка,
            `predict_input` - выборка, которая известна на момент предсказания,
            `test` - тестовая выборка
        """

    def split(
            self,
            log: DataFrame,
    ) -> SplitterReturnType:
        """
        Разбивает лог действий пользователей на обучающую и тестовую выборки.

        :param log: лог взаимодействия, спарк-датафрейм с колонками
           `[timestamp, user_id, item_id, context, relevance]`
        :returns: тройка спарк-датафреймов структуры, аналогичной входной,
            `train, predict_input, test`, где `train` - обучающая выборка,
            `predict_input` - выборка, которая известна на момент предсказания,
            `test` - тестовая выборка
        """
        train, predict_input, test = self._core_split(log)

        test = self._drop_cold_items_and_users(
            train, test,
            self.drop_cold_items, self.drop_cold_users
        )

        return (train,
                self._filter_zero_relevance(predict_input),
                self._filter_zero_relevance(test))
