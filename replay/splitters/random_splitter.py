from typing import Optional, Tuple

from replay.utils import DataFrameLike, PandasDataFrame, PolarsDataFrame, SparkDataFrame

from .base_splitter import Splitter, SplitterReturnType


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

    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
        query_column: str = "query_id",
        item_column: str = "item_id",
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
            item_column=item_column,
        )
        self.seed = seed
        if test_size < 0 or test_size > 1:
            msg = "test_size must between 0 and 1"
            raise ValueError(msg)
        self.test_size = test_size

    def _random_split_spark(
        self, interactions: SparkDataFrame, threshold: float
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        train, test = interactions.randomSplit([1 - threshold, threshold], self.seed)
        return train, test

    def _random_split_pandas(
        self, interactions: PandasDataFrame, threshold: float
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        train = interactions.sample(frac=(1 - threshold), random_state=self.seed)
        test = interactions.drop(train.index)
        return train, test

    def _random_split_polars(
        self, interactions: PolarsDataFrame, threshold: float
    ) -> Tuple[PolarsDataFrame, PolarsDataFrame]:
        train_size = int(len(interactions) * (1 - threshold)) + 1
        shuffled_interactions = interactions.sample(fraction=1, shuffle=True, seed=self.seed)
        train = shuffled_interactions[:train_size]
        test = shuffled_interactions[train_size:]
        return train, test

    def _core_split(self, interactions: DataFrameLike) -> SplitterReturnType:
        if isinstance(interactions, SparkDataFrame):
            return self._random_split_spark(interactions, self.test_size)
        if isinstance(interactions, PandasDataFrame):
            return self._random_split_pandas(interactions, self.test_size)
        if isinstance(interactions, PolarsDataFrame):
            return self._random_split_polars(interactions, self.test_size)

        msg = f"{self} is not implemented for {type(interactions)}"
        raise NotImplementedError(msg)
