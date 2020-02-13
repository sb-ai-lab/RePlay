"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.

В скрипте собраны классы для разбиения лога взаимодействий пользователей и
объектов на тестовую и обучающие выборки так, что делится лог каждого
пользователя по отдельности.
Способы разбиения - по времени и случайно.
"""
from abc import abstractmethod
from typing import Optional, Union

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, Window

from sponge_bob_magic.splitters.base_splitter import (Splitter,
                                                      SplitterReturnType)


class UserLogSplitter(Splitter):
    """ Абстрактный класс для деления лога каждого пользователя. """

    def __init__(
            self,
            drop_cold_items: bool,
            drop_cold_users: bool,
            item_test_size: Union[float, int] = 1,
            user_test_size: Optional[Union[float, int]] = None,
            seed: int = None):
        """
        :param drop_cold_items: исключать ли из тестовой выборки объекты,
           которых нет в обучающей
        :param drop_cold_users: исключать ли из тестовой выборки пользователей,
           которых нет в обучающей
        :param item_test_size: размер тестовой выборки; если от 0 до 1, то в
            тест попадает данная доля объектов у каждого пользователя: если
            число большее 1, то в тест попадает заданное число объектов у
            каждого пользователя
        :param user_test_size: аналогично item_test_size, но не сколько
            объектов от каждого пользователя включать в тест, а сколько самих
            пользователей (доля либо количество); если None, то берутся все
            пользователи
        :param seed: сид для разбиения
        """
        super().__init__(drop_cold_users, drop_cold_items)
        self.item_test_size = item_test_size
        self.user_test_size = user_test_size
        self.seed = seed

    def get_test_users(
            self,
            log: DataFrame,
    ) -> DataFrame:
        """
        отобрать тестовых пользователей

        :param log: стандартный лог взаимодействий
        :return: Spark DataFrame с одной колонкой `user_id`
        """
        all_users = log.select("user_id").distinct()
        user_count = all_users.count()
        if self.user_test_size is not None:
            value_error = False
            if isinstance(self.user_test_size, int):
                if (
                        self.user_test_size >= 1 and
                        self.user_test_size < user_count
                ):
                    fraction = self.user_test_size / user_count
                else:
                    value_error = True
            else:
                if self.user_test_size < 1 and self.user_test_size > 0:
                    fraction = self.user_test_size
                else:
                    value_error = True
            if value_error:
                raise ValueError(f"""
                Недопустимое значение параметра
                user_test_size: {self.user_test_size}
                """)
            test_users = all_users.sample(
                fraction=fraction, seed=self.seed, withReplacement=False)
        else:
            test_users = all_users
        return test_users

    @abstractmethod
    def _split_proportion(self, log: DataFrame) -> SplitterReturnType:
        """
        Внутренний метод, который должны имплементировать классы-наследники.

        Разбивает лог действий пользователей на обучающую и тестовую
        выборки так, чтобы в тестовой выборке была фиксированная доля
        объектов для каждого пользователя. Способ разбиения определяется
        классом-наследником.

        :param log: лог взаимодействия, спарк-датафрейм с колонками
            `[timestamp, user_id, item_id, context, relevance]`
        :return: тройка спарк-датафреймов структуры, аналогичной входной
            `train, predict_input, test`
        """

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

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        if 0 <= self.item_test_size < 1.0:
            train, predict_input, test = self._split_proportion(log)
        elif self.item_test_size >= 1 and isinstance(self.item_test_size, int):
            train, predict_input, test = self._split_quantity(log)
        else:
            raise ValueError(
                "Значение `test_size` должно быть в диапазоне [0, 1] или "
                "быть целым числом больше 1; "
                f"сейчас test_size={self.item_test_size}"
            )

        return train, predict_input, test


class RandomUserLogSplitter(UserLogSplitter):
    """ Класс для деления лога каждого пользователя случайно. """

    @staticmethod
    def _add_random_partition(dataframe: DataFrame, seed: int = None) -> DataFrame:
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
        test_users = self.get_test_users(log).withColumn(
            "test_user", sf.lit(1)
        )
        res = self._add_random_partition(
            log.join(test_users, how="left", on="user_id"),
            self.seed
        )
        train = res.filter(f"""
            row_num > {self.item_test_size} OR
            test_user IS NULL
        """).drop("rand", "row_num", "test_user")
        test = res.filter(f"""
            row_num <= {self.item_test_size} AND
            test_user IS NOT NULL
        """).drop("rand", "row_num", "test_user")
        return train, train, test

    def _split_proportion(self, log: DataFrame) -> SplitterReturnType:
        counts = log.groupBy("user_id").count()
        test_users = self.get_test_users(log).withColumn(
            "test_user", sf.lit(1)
        )
        res = self._add_random_partition(
            log.join(test_users, how="left", on="user_id"),
            self.seed
        )
        res = res.join(counts, on="user_id", how="left")
        res = res.withColumn(
            "frac",
            sf.col("row_num") / sf.col("count")
        ).cache()
        train = res.filter(f"""
            frac > {self.item_test_size} OR
            test_user IS NULL
        """).drop("rand", "row_num", "count", "frac", "test_user")
        test = res.filter(f"""
            frac <= {self.item_test_size} AND
            test_user IS NOT NULL
        """).drop("rand", "row_num", "count", "frac", "test_user")
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
        test_users = self.get_test_users(log).withColumn(
            "test_user", sf.lit(1)
        )
        res = self._add_time_partition(
            log.join(test_users, how="left", on="user_id")
        )
        train = res.filter(f"""
            row_num > {self.item_test_size} OR
            test_user IS NULL
        """).drop("row_num", "test_user")
        test = res.filter(f"""
            row_num <= {self.item_test_size} AND
            test_user IS NOT NULL
        """).drop("row_num", "test_user")
        return train, train, test

    def _split_proportion(self, log: DataFrame) -> SplitterReturnType:
        test_users = self.get_test_users(log).withColumn(
            "test_user", sf.lit(1)
        )
        res = self._add_time_partition(
            log.join(test_users, how="left", on="user_id")
        )
        counts = log.groupBy("user_id").count()
        res = res.join(counts, on="user_id", how="left")
        res = res.withColumn(
            "frac",
            sf.col("row_num") / sf.col("count")
        ).cache()
        train = res.filter(f"""
            frac > {self.item_test_size} OR
            test_user IS NULL
        """).drop("row_num", "count", "frac", "test_user")
        test = res.filter(f"""
            frac <= {self.item_test_size} AND
            test_user IS NOT NULL
        """).drop("row_num", "count", "frac", "test_user")
        return train, train, test
