from typing import Optional, Union

from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
import pyspark.sql.functions as sf

from replay.data import AnyDataFrame
from replay.splitters.base_splitter import (
    Splitter,
    SplitterReturnType,
)


# pylint: disable=too-few-public-methods, duplicate-code
class ColdUserRandomSplitter(Splitter):
    """
    Test set consists of all actions of randomly chosen users.
    """

    # для использования в тестах

    _init_arg_names = [
        "test_size",
        "drop_cold_users",
        "drop_cold_items",
        "seed",
        "query_column",
        "item_column",
        "timestamp_column",
        "rating_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
        query_column: str = "user_id",
        item_column: Optional[str] = "item_id",
        timestamp_column: Optional[str] = "timestamp",
        rating_column: Optional[str] = "relevance",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: fraction of users to be in test
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        :param query_column: query id column name
        :param item_column: item id column name
        :param timestamp_column: timestamp column name
        :param rating_column: rating column name
        :param session_id_column: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            query_column=query_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            rating_column=rating_column,
            session_id_column=session_id_column,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.seed = seed
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must be 0 to 1")
        self.test_size = test_size

    def _get_order_of_sort(self) -> list:   # pragma: no cover
        pass

    def _core_split_pandas(
        self,
        interactions: PandasDataFrame,
        threshold: float
    ) -> Union[PandasDataFrame, PandasDataFrame]:
        index_name = interactions.index.name
        df = interactions.reset_index()
        users = PandasDataFrame(df[self.query_column].unique(), columns=[self.query_column])
        train_users = users.sample(frac=(1 - threshold), random_state=self.seed)
        test_users = users.drop(train_users.index)

        train = df.merge(train_users, on=self.query_column, how="inner")
        test = df.merge(test_users, on=self.query_column, how="inner")
        train.set_index("index", inplace=True)
        test.set_index("index", inplace=True)

        train.index.name = index_name
        test.index.name = index_name

        if self.session_id_column:
            interactions["is_test"] = False
            interactions.loc[test.index, "is_test"] = True
            interactions = self._recalculate_with_session_id_column(interactions)
            train = interactions[~interactions["is_test"]].drop(columns=["is_test"])
            test = interactions[interactions["is_test"]].drop(columns=["is_test"])
            interactions = interactions.drop(columns=["is_test"])

        return train, test

    def _core_split_spark(
        self,
        interactions: SparkDataFrame,
        threshold: float
    ) -> Union[SparkDataFrame, SparkDataFrame]:
        users = interactions.select(self.query_column).distinct()
        train_users, test_users = users.randomSplit(
            [1 - threshold, threshold],
            seed=self.seed,
        )
        train = interactions.join(train_users, on=self.query_column, how="inner")
        test = interactions.join(test_users, on=self.query_column, how="inner")

        if self.session_id_column:
            test = test.withColumn("is_test", sf.lit(True))
            interactions = interactions.join(test, on=interactions.schema.names, how="left").na.fill({"is_test": False})
            interactions = self._recalculate_with_session_id_column(interactions)
            train = interactions.filter(~sf.col("is_test")).drop("is_test")
            test = interactions.filter(sf.col("is_test")).drop("is_test")

        return train, test

    def _core_split(self, interactions: AnyDataFrame) -> SplitterReturnType:
        split_method = self._core_split_spark
        if isinstance(interactions, PandasDataFrame):
            split_method = self._core_split_pandas

        return split_method(interactions, self.test_size)
