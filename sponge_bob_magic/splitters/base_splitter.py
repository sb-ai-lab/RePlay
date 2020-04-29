"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from abc import ABC, abstractmethod
from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from sponge_bob_magic.converter import convert, get_type

SplitterReturnType = Tuple[DataFrame, DataFrame]


# pylint: disable=too-few-public-methods
class Splitter(ABC):
    """ Базовый класс для разбиения выборки на обучающую и тестовую. """

    def __init__(
        self, drop_cold_items: bool, drop_cold_users: bool,
    ):
        """
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
           которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
           которых нет в обучающей
        """
        self.drop_cold_users = drop_cold_users
        self.drop_cold_items = drop_cold_items

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
        drop_cold_users: bool,
    ) -> DataFrame:
        """
        Удаляет из тестовой выборки холодных users и холодные items.

        :param train: обучающая выборка, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, relevance]`
        :param test: тестовая выборка как train
        :param drop_cold_items: если True, удаляет холодные items
        :param drop_cold_users: если True, удаляет холодные users
        :return: тестовая выборка без холодных users / items
        """
        if drop_cold_items:
            train_tmp = train.select(
                sf.col("item_id").alias("item")
            ).distinct()
            test = test.join(train_tmp, train_tmp.item == test.item_id).drop(
                "item"
            )

        if drop_cold_users:
            train_tmp = train.select(
                sf.col("user_id").alias("user")
            ).distinct()
            test = test.join(train_tmp, train_tmp.user == test.user_id).drop(
                "user"
            )
        return test

    @abstractmethod
    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        """
        Внутренний метод, который должны имплеменитровать классы-наследники.

        Разбивает лог действий пользователей на обучающую и тестовую выборки.

        :param log: лог взаимодействия, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, relevance]`
        :returns: спарк-датафреймы структуры, аналогичной входной,
            `train, test`, где `train` - обучающая выборка,
            `test` - тестовая выборка
        """

    def split(self, log: DataFrame,) -> SplitterReturnType:
        """
        Разбивает лог действий пользователей на обучающую и тестовую выборки.

        :param log: лог взаимодействия, спарк-датафрейм с колонками
           ``[timestamp, user_id, item_id, relevance]``
        :returns: спарк-датафреймы структуры, аналогичной входной,
            ``train, test``, где ``train`` - обучающая выборка,
            ``test`` - тестовая выборка
        """
        type_in = get_type(log)
        train, test = self._core_split(convert(log))
        test = self._drop_cold_items_and_users(
            train, test, self.drop_cold_items, self.drop_cold_users
        )
        return convert(train, self._filter_zero_relevance(test), to=type_in)
