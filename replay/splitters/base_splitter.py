from abc import ABC, abstractmethod
from typing import Optional, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as sf

from replay.data import AnyDataFrame
from replay.utils.spark_utils import convert2spark

SplitterReturnType = Tuple[DataFrame, DataFrame]


# pylint: disable=too-few-public-methods
class Splitter(ABC):
    """Base class"""

    _init_arg_names = [
        "drop_cold_users",
        "drop_cold_items",
        "drop_zero_rel_in_test",
        "user_col",
        "item_col",
        "date_col",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        drop_cold_items: bool,
        drop_cold_users: bool,
        drop_zero_rel_in_test: bool,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx",
        date_col: Optional[str] = "timestamp",
    ):
        """
        :param drop_cold_items: flag to remove items that are not in train data
        :param drop_cold_users: flag to remove users that are not in train data
        :param drop_zero_rel_in_test: flag to remove entries with relevance <= 0
            from the test part of the dataset
        :param user_col: user id column name
        :param item_col: item id column name
        :param date_col: timestamp column name
        """
        self.drop_cold_users = drop_cold_users
        self.drop_cold_items = drop_cold_items
        self.drop_zero_rel_in_test = drop_zero_rel_in_test
        self.user_col = user_col
        self.item_col = item_col
        self.date_col = date_col

    @property
    def _init_args(self):
        return {name: getattr(self, name) for name in self._init_arg_names}

    def __str__(self):
        return type(self).__name__

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

    # pylint: disable=too-many-arguments
    @staticmethod
    def _drop_cold_items_and_users(
        train: DataFrame,
        test: DataFrame,
        drop_cold_items: bool,
        drop_cold_users: bool,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx"
    ) -> DataFrame:
        """
        Removes cold users and items from the test data

        :param train: train DataFrame `[timestamp, user_id, item_id, relevance]`
        :param test: DataFrame like train
        :param drop_cold_items: flag to remove cold items
        :param drop_cold_users: flag to remove cold users
        :param user_col: user id column name
        :param item_col: item id column name
        :return: filtered DataFrame
        """
        if drop_cold_items:
            train_tmp = train.select(
                sf.col(item_col).alias("_item_id_inner")
            ).distinct()
            test = test.join(train_tmp, sf.col(item_col) == sf.col("_item_id_inner")).drop(
                "_item_id_inner"
            )

        if drop_cold_users:
            train_tmp = train.select(
                sf.col(user_col).alias("_user_id_inner")
            ).distinct()
            test = test.join(train_tmp, sf.col(user_col) == sf.col("_user_id_inner")).drop(
                "_user_id_inner"
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
            train, test, self.drop_cold_items, self.drop_cold_users, self.user_col, self.item_col
        )
        test = self._filter_zero_relevance(test).cache()
        test.count()
        return train, test
