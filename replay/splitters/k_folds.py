from typing import Optional, Literal

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame as SparkDataFrame, Window

from replay.data import AnyDataFrame
from replay.splitters.base_splitter import Splitter, SplitterReturnType


StrategyName = Literal["query"]


# pylint: disable=too-few-public-methods
class KFolds(Splitter):
    """
    Splits interactions inside each query into folds at random.
    """
    _init_arg_names = [
        "n_folds",
        "strategy",
        "drop_cold_users",
        "drop_cold_items",
        "seed",
        "query_column",
        "item_column",
        "timestamp_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        n_folds: Optional[int] = 5,
        strategy: Optional[StrategyName] = "query",
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
        query_column: str = "query_id",
        item_column: Optional[str] = "item_id",
        timestamp_column: Optional[str] = "timestamp",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param n_folds: number of folds.
        :param strategy: splitting strategy. Only query variant is available atm.
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        :param query_column: query id column name
        :param item_column: item id column name
        :param timestamp_column: timestamp column name
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
            session_id_column=session_id_column,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.n_folds = n_folds
        if strategy not in {"query"}:
            raise ValueError(f"Wrong splitter parameter: {strategy}")
        self.strategy = strategy
        self.seed = seed

    def _core_split(self, interactions: AnyDataFrame) -> SplitterReturnType:
        if self.strategy == "query":
            if isinstance(interactions, SparkDataFrame):
                dataframe = interactions.withColumn("_rand", sf.rand(self.seed))
                dataframe = dataframe.withColumn(
                    "fold",
                    sf.row_number().over(
                        Window.partitionBy(self.query_column).orderBy("_rand")
                    )
                    % self.n_folds,
                ).drop("_rand")
                for i in range(self.n_folds):
                    dataframe = dataframe.withColumn(
                        "is_test",
                        sf.when(sf.col("fold") != i, True).otherwise(False)
                    )
                    if self.session_id_column:
                        dataframe = self._recalculate_with_session_id_column(dataframe)

                    train = dataframe.filter(~sf.col("is_test")).drop("is_test", "fold")
                    test = dataframe.filter(sf.col("is_test")).drop("is_test", "fold")
                    yield train, test
            else:
                dataframe = interactions.sample(frac=1, random_state=self.seed).sort_values(self.query_column)
                dataframe["fold"] = (dataframe.groupby(self.query_column, sort=False).cumcount() + 1) % self.n_folds
                for i in range(self.n_folds):
                    dataframe["is_test"] = dataframe["fold"] == i
                    if self.session_id_column:
                        dataframe = self._recalculate_with_session_id_column(dataframe)

                    train = dataframe[~dataframe["is_test"]].drop(columns=["is_test", "fold"])
                    test = dataframe[dataframe["is_test"]].drop(columns=["is_test", "fold"])
                    dataframe.drop(columns=["is_test"])
                    yield train, test
