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
    ):
        """
        :param drop_cold_items: flag to remove items that are not in train data
        :param drop_cold_users: flag to remove users that are not in train data
        """
        self.drop_cold_users = drop_cold_users
        self.drop_cold_items = drop_cold_items

    @staticmethod
    def _filter_zero_relevance(dataframe: DataFrame) -> DataFrame:
        """
        Removes records with zero relevance

        :param dataframe: input DataFrame
        :returns: filtered DataFrame
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
        Removes cold users and items from the test data

        :param train: train DataFrame `[timestamp, user_id, item_id, relevance]`
        :param test: DataFrame like train
        :param drop_cold_items: flag to remove cold items
        :param drop_cold_users: flag to remove cold users
        :return: filtered DataFrame
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
        test = self._drop_cold_items_and_users(
            train, test, self.drop_cold_items, self.drop_cold_users
        )
        return train.cache(), self._filter_zero_relevance(test).cache()
