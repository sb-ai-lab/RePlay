from abc import ABC, abstractmethod
from typing import Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.constants import AnyDataFrame
from replay.utils import convert2spark

SplitterReturnType = Tuple[DataFrame, DataFrame]


# pylint: disable=too-few-public-methods
class Splitter(ABC):
    """Base class"""

    def __init__(
        self,
        drop_cold_items: bool,
        drop_cold_users: bool,
        drop_zero_rel_in_test: bool,
    ):
        """
        :param drop_cold_items: flag to remove items that are not in train data
        :param drop_cold_users: flag to remove users that are not in train data
        :param drop_zero_rel_in_test: flag to remove entries with relevance <= 0
            from the test part of the dataset
        """
        self.drop_cold_users = drop_cold_users
        self.drop_cold_items = drop_cold_items
        self.drop_zero_rel_in_test = drop_zero_rel_in_test

    def _filter_zero_relevance(self, dataframe: DataFrame) -> DataFrame:
        """
        Removes records with zero relevance if required by
        `drop_zero_rel_in_test` initialization parameter

        :param dataframe: input DataFrame
        :returns: filtered DataFrame
        """
        if self.drop_zero_rel_in_test:
            return dataframe.filter("relevance > 0.0")
        return dataframe

    @staticmethod
    def _drop_cold_items_and_users(
        train: DataFrame,
        test: DataFrame,
        drop_cold_items: bool,
        drop_cold_users: bool,
    ) -> DataFrame:
        """
        Removes cold users and items from the test data

        :param train: train DataFrame `[timestamp, user_id, item_id, relevance]`
        :param test: DataFrame like train
        :param drop_cold_items: flag to remove cold items
        :param drop_cold_users: flag to remove cold users
        :return: filtered DataFrame
        """
        if drop_cold_items:
            train_tmp = train.select(
                sf.col("item_idx").alias("item")
            ).distinct()
            test = test.join(train_tmp, train_tmp.item == test.item_idx).drop(
                "item"
            )

        if drop_cold_users:
            train_tmp = train.select(
                sf.col("user_idx").alias("user")
            ).distinct()
            test = test.join(train_tmp, train_tmp.user == test.user_idx).drop(
                "user"
            )
        return test

    @abstractmethod
    def _core_split(self, log: DataFrame) -> SplitterReturnType:
        """
        This method implements split strategy

        :param log: input DataFrame `[timestamp, user_id, item_id, relevance]`
        :returns: `train` and `test DataFrames
        """

    def split(self, log: AnyDataFrame) -> SplitterReturnType:
        """
        Splits input DataFrame into train and test

        :param log: input DataFrame ``[timestamp, user_id, item_id, relevance]``
        :returns: `train` and `test` DataFrame
        """
        train, test = self._core_split(convert2spark(log))  # type: ignore
        train.cache()
        train.count()
        test = self._drop_cold_items_and_users(
            train, test, self.drop_cold_items, self.drop_cold_users
        )
        test = self._filter_zero_relevance(test).cache()
        test.count()
        return train, test
