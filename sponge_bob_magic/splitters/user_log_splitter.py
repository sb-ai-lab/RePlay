"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from abc import abstractmethod

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, SparkSession, Window

from sponge_bob_magic.splitters.base_splitter import (Splitter,
                                                      SplitterReturnType)


class UserLogSplitter(Splitter):
    def __init__(self, spark: SparkSession,
                 test_size: float or int = 0.3,
                 seed: int = 1234):
        """
        :param seed: сид для разбиения
        :param test_size: размер тестовой выборки; если от 0 до 1, то в тест
            попадает данная доля объектов у каждого пользователя: если число
            большее 1, то в тест попадает заданное число объектов у каждого
            пользователя
        :param spark: инициализированная спарк-сессия
        """
        super().__init__(spark)

        self.test_size = test_size
        self.seed = seed

    @abstractmethod
    def _split_proportion(self, log: DataFrame) -> SplitterReturnType:
        """
        Внутренний метод, который должны имплементировать классы-наследники.

        Разбивает лог действий пользователей на обучающую и тестовую
        выборки так, чтобы в тестовой выборке было фиксированная доля
        объектов для каждого пользователя. Способ разбиения определяется
        классом-наследником.

        :param log: лог взаимодействия, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, context, relevance]`
        :return: тройка спарк-датафреймов структуры, аналогичной входной
            `train, predict_input, test`
        """
        pass

    @abstractmethod
    def _split_quantity(self, log: DataFrame) -> SplitterReturnType:
        """
        Внутренний метод, который должны имплементировать классы-наследники.

        Разбивает лог действий пользователей на обучающую и тестовую
        выборки так, чтобы в тестовой выборке было фиксированное количество
        объектов для каждого пользователя. Способ разбиения определяется
        классом-наследником.

        :param log: лог взаимодействия, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, context, relevance]`
        :return: тройка спарк-датафреймов структуры, аналогичной входной
            `train, predict_input, test`
        """
        pass

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        if 0 <= self.test_size <= 1:
            train, predict_input, test = self._split_proportion(log)
        elif 1 <= self.test_size:
            train, predict_input, test = self._split_quantity(log)
        else:
            raise ValueError(
                "Значение `test_size` должно быть в диапазоне [0, 1] или "
                f"быть числом больше 1; сейчас test_size={self.test_size}"
            )

        return train, predict_input, test


class RandomUserLogSplitter(UserLogSplitter):
    """ Класс для деления лога каждого пользователя случайно. """

    @staticmethod
    def _add_random_partition(dataframe: DataFrame, seed: int) -> DataFrame:
        """
        Добавляет в датафрейм колонку случайных чисел `rand` и колонку
        порядкового номера пользователя `row_num` на основе этого случайного
        порядка. Пользователи должны лежать в колонке `user_id`.

        :param dataframe: спарк-датафрейм с обязательной колонкой `user_id`
        :returns: датафрейм с добавленными колонками
        """
        dataframe = dataframe.withColumn("rand", sf.rand(seed))
        dataframe = dataframe.withColumn(
            "row_num",
            sf.row_number().over(Window
                                 .partitionBy("user_id")
                                 .orderBy("rand"))
        ).cache()
        return dataframe

    def _split_quantity(self, log: DataFrame) -> SplitterReturnType:
        res = self._add_random_partition(log, self.seed)

        train = (res
                 .filter(res.row_num > self.test_size)
                 .drop("rand", "row_num"))
        test = (res
                .filter(res.row_num <= self.test_size)
                .drop("rand", "row_num"))

        return train, train, test

    def _split_proportion(self, log: DataFrame) -> SplitterReturnType:
        counts = log.groupBy("user_id").count()
        res = self._add_random_partition(log, self.seed)

        res = res.join(counts, on="user_id", how="left")
        res = res.withColumn(
            "frac",
            sf.col("row_num") / sf.col("count")
        ).cache()

        train = (res
                 .filter(res.frac > self.test_size)
                 .drop("rand", "row_num", "count", "frac"))
        test = (res
                .filter(res.frac <= self.test_size)
                .drop("rand", "row_num", "count", "frac"))
        return train, train, test


class ByTimeUserLogSplitter(UserLogSplitter):
    """ Класс для деления лога каждого пользователя по времени. """

    @staticmethod
    def _add_time_partition(dataframe: DataFrame) -> DataFrame:
        """
        Добавляет в лог столбец порядкового номера пользователя `row_num`
        на основе порядка времени в колонке `timestamp`. Пользователи
        должны лежать в колонке `user_id`.

        :param dataframe: спарк-датафрейм с обязательными колонками
            `[timestamp, user_id]`
        :returns: датафрейм с добавленной колонкой
        """
        res = dataframe.withColumn(
            "row_num",
            sf.row_number().over(Window
                                 .partitionBy("user_id")
                                 .orderBy(sf.col("timestamp")
                                          .desc()))
        ).cache()
        return res

    def _split_quantity(self, log: DataFrame) -> SplitterReturnType:
        res = self._add_time_partition(log)
        train = res.filter(res.row_num > self.test_size).drop("row_num")
        test = res.filter(res.row_num <= self.test_size).drop("row_num")

        return train, train, test

    def _split_proportion(self, log: DataFrame) -> SplitterReturnType:
        counts = log.groupBy("user_id").count()
        res = self._add_time_partition(log)

        res = res.join(counts, on="user_id", how="left")

        res = res.withColumn(
            "frac",
            sf.col("row_num") / sf.col("count")
        ).cache()

        train = (res
                 .filter(res.frac > self.test_size)
                 .drop("row_num", "count", "frac"))

        test = (res
                .filter(res.frac <= self.test_size)
                .drop("row_num", "count", "frac"))

        return train, train, test
