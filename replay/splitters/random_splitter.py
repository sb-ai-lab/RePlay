from typing import Optional, Union

from .base_splitter import Splitter, SplitterReturnType
from replay.utils import DataFrameLike, PandasDataFrame, SparkDataFrame


# pylint: disable=too-few-public-methods, duplicate-code
class RandomSplitter(Splitter):
    """Assign records into train and test at random."""

    _init_arg_names = [
        "test_size",
        "drop_cold_users",
        "drop_cold_items",
        "seed",
        "query_column",
        "item_column",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
        query_column: str = "query_id",
        item_column: str = "item_id"
    ):
        """
        :param test_size: test size 0 to 1
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        :param query_column: Name of query interaction column
        :param item_column: Name of item interaction column
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            query_column=query_column,
            item_column=item_column
        )
        self.seed = seed
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must between 0 and 1")
        self.test_size = test_size

    def _random_split_spark(
        self,
        interactions: SparkDataFrame,
        threshold: float
    ) -> Union[SparkDataFrame, SparkDataFrame]:
        train, test = interactions.randomSplit(
            [1 - threshold, threshold], self.seed
        )
        return train, test

    def _random_split_pandas(
        self,
        interactions: PandasDataFrame,
        threshold: float
    ) -> Union[PandasDataFrame, PandasDataFrame]:
        train = interactions.sample(frac=(1 - threshold), random_state=self.seed)
        test = interactions.drop(train.index)
        return train, test

    def _core_split(self, interactions: DataFrameLike) -> SplitterReturnType:
        split_method = self._random_split_spark
        if isinstance(interactions, PandasDataFrame):
            split_method = self._random_split_pandas
        return split_method(interactions, self.test_size)
