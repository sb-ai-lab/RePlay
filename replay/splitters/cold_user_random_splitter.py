from typing import Optional, Union
import polars as pl

from .base_splitter import Splitter, SplitterReturnType
from replay.utils import (
    PYSPARK_AVAILABLE,
    DataFrameLike,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf


# pylint: disable=too-few-public-methods, duplicate-code
class ColdUserRandomSplitter(Splitter):
    """
    Test set consists of all actions of randomly chosen users.
    """

    _init_arg_names = [
        "test_size",
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
        seed: Optional[int] = None,
        query_column: str = "query_id",
        item_column: Optional[str] = "item_id",
    ):
        """
        :param test_size: fraction of users to be in test
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        :param query_column: query id column name
        :param item_column: item id column name
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            query_column=query_column,
            item_column=item_column,
        )
        self.seed = seed
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must between 0 and 1")
        self.test_size = test_size

    def _core_split_pandas(
        self,
        interactions: PandasDataFrame,
        threshold: float
    ) -> Union[PandasDataFrame, PandasDataFrame]:
        users = PandasDataFrame(interactions[self.query_column].unique(), columns=[self.query_column])
        train_users = users.sample(frac=(1 - threshold), random_state=self.seed)
        train_users["is_test"] = False

        interactions = interactions.merge(train_users, on=self.query_column, how="left")
        interactions["is_test"].fillna(True, inplace=True)

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
        train_users, _ = users.randomSplit(
            [1 - threshold, threshold],
            seed=self.seed,
        )
        interactions = interactions.join(
            train_users.withColumn("is_test", sf.lit(False)),
            on=self.query_column,
            how="left"
        ).na.fill({"is_test": True})

        train = interactions.filter(~sf.col("is_test")).drop("is_test")
        test = interactions.filter(sf.col("is_test")).drop("is_test")

        return train, test

    def _core_split_polars(
        self,
        interactions: PolarsDataFrame,
        threshold: float
    ) -> Union[PolarsDataFrame, PolarsDataFrame]:
        train_users = (
            interactions
            .select(self.query_column)
            .unique()
            .sample(fraction=(1 - threshold), seed=self.seed)
            .with_columns(pl.lit(False).alias("is_test"))
        )

        interactions = (
            interactions
            .join(
                train_users,
                on=self.query_column, how="left")
            .fill_null(True)
        )

        train = interactions.filter(~pl.col("is_test")).drop("is_test")  # pylint: disable=invalid-unary-operand-type
        test = interactions.filter(pl.col("is_test")).drop("is_test")
        return train, test

    def _core_split(self, interactions: DataFrameLike) -> SplitterReturnType:
        if isinstance(interactions, SparkDataFrame):
            return self._core_split_spark(interactions, self.test_size)
        if isinstance(interactions, PandasDataFrame):
            return self._core_split_pandas(interactions, self.test_size)
        if isinstance(interactions, PolarsDataFrame):
            return self._core_split_polars(interactions, self.test_size)

        raise NotImplementedError(f"{self} is not implemented for {type(interactions)}")
