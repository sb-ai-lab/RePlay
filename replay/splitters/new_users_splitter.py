from typing import Optional, Union
import polars as pl

from .base_splitter import Splitter, SplitterReturnType
from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, PolarsDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf
    from pyspark.sql import Window


# pylint: disable=too-few-public-methods, duplicate-code
class NewUsersSplitter(Splitter):
    """
    Only new users will be assigned to test set.
    Splits interactions by timestamp so that test has `test_size` fraction of most recent users.


    >>> from replay.splitters import NewUsersSplitter
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"query_id": [1,1,2,2,3,4],
    ...    "item_id": [1,2,3,1,2,3],
    ...    "relevance": [1,2,3,4,5,6],
    ...    "timestamp": [20,40,20,30,10,40]})
    >>> data_frame
       query_id   item_id  relevance  timestamp
    0         1         1          1         20
    1         1         2          2         40
    2         2         3          3         20
    3         2         1          4         30
    4         3         2          5         10
    5         4         3          6         40
    >>> train, test = NewUsersSplitter(test_size=0.1).split(data_frame)
    >>> train
      query_id  item_id  relevance  timestamp
    0        1        1          1         20
    2        2        3          3         20
    3        2        1          4         30
    4        3        2          5         10
    <BLANKLINE>
    >>> test
      query_id  item_id  relevance  timestamp
    0        4        3          6         40
    <BLANKLINE>

    Train DataFrame can be drastically reduced even with moderate
    `test_size` if the amount of new users is small.

    >>> train, test = NewUsersSplitter(test_size=0.3).split(data_frame)
    >>> train
      query_id  item_id  relevance  timestamp
    4        3        2          5         10
    <BLANKLINE>
    """

    _init_arg_names = [
        "test_size",
        "drop_cold_items",
        "query_column",
        "item_column",
        "timestamp_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: float,
        drop_cold_items: bool = False,
        query_column: str = "query_id",
        item_column: Optional[str] = "item_id",
        timestamp_column: Optional[str] = "timestamp",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param test_size: test size 0 to 1
        :param drop_cold_items: flag to drop cold items from test
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
            query_column=query_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            session_id_column=session_id_column,
            session_id_processing_strategy=session_id_processing_strategy
        )
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must between 0 and 1")
        self.test_size = test_size

    def _core_split_pandas(
        self,
        interactions: PandasDataFrame,
        threshold: float
    ) -> Union[PandasDataFrame, PandasDataFrame]:
        start_date_by_user = interactions.groupby(self.query_column).agg(
            _start_dt_by_user=(self.timestamp_column, "min")
        ).reset_index()
        test_start_date = (
            start_date_by_user
            .groupby("_start_dt_by_user")
            .agg(_num_users_by_start_date=(self.query_column, "count")).reset_index()
            .sort_values(by="_start_dt_by_user", ascending=False)
        )
        test_start_date["_cum_num_users_to_dt"] = test_start_date["_num_users_by_start_date"].cumsum()
        test_start_date["total"] = sum(test_start_date["_num_users_by_start_date"])
        test_start_date = test_start_date[
            test_start_date["_cum_num_users_to_dt"] >= threshold * test_start_date["total"]
        ]
        test_start = test_start_date["_start_dt_by_user"].max()

        train = interactions[interactions[self.timestamp_column] < test_start]
        test = interactions.merge(
            start_date_by_user[start_date_by_user["_start_dt_by_user"] >= test_start],
            how="inner",
            on=self.query_column
        ).drop(columns=["_start_dt_by_user"])

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
        start_date_by_user = interactions.groupby(self.query_column).agg(
            sf.min(self.timestamp_column).alias("_start_dt_by_user")
        )
        test_start_date = (
            start_date_by_user.groupby("_start_dt_by_user")
            .agg(sf.count(self.query_column).alias("_num_users_by_start_date"))
            .select(
                "_start_dt_by_user",
                sf.sum("_num_users_by_start_date")
                .over(Window.orderBy(sf.desc("_start_dt_by_user")))
                .alias("_cum_num_users_to_dt"),
                sf.sum("_num_users_by_start_date").over(Window.orderBy(sf.lit(1))).alias("total"),
            )
            .filter(sf.col("_cum_num_users_to_dt") >= sf.col("total") * threshold)
            .agg(sf.max("_start_dt_by_user"))
            .head()[0]
        )

        train = interactions.filter(sf.col(self.timestamp_column) < test_start_date)
        test = interactions.join(
            start_date_by_user.filter(sf.col("_start_dt_by_user") >= test_start_date),
            how="inner",
            on=self.query_column,
        ).drop("_start_dt_by_user")

        if self.session_id_column:
            test = test.withColumn("is_test", sf.lit(True))
            interactions = interactions.join(test, on=interactions.schema.names, how="left").na.fill({"is_test": False})
            interactions = self._recalculate_with_session_id_column(interactions)
            train = interactions.filter(~sf.col("is_test")).drop("is_test")
            test = interactions.filter(sf.col("is_test")).drop("is_test")

        return train, test

    def _core_split_polars(
        self,
        interactions: PolarsDataFrame,
        threshold: float
    ) -> Union[PolarsDataFrame, PolarsDataFrame]:
        start_date_by_user = (
            interactions
            .group_by(self.query_column).agg(
                pl.col(self.timestamp_column).min()
                .alias("_start_dt_by_user")
            )
        )
        test_start_date = (
            start_date_by_user
            .group_by("_start_dt_by_user").agg(
                pl.col(self.query_column).count()
                .alias("_num_users_by_start_date")
            )
            .sort("_start_dt_by_user", descending=True)
            .with_columns(
                pl.col("_num_users_by_start_date").cum_sum()
                .alias("cum_sum_users"),
            )
            .filter(
                pl.col("cum_sum_users") >= pl.col("cum_sum_users").max() * threshold
            )
            ["_start_dt_by_user"]
            .max()
        )

        train = interactions.filter(pl.col(self.timestamp_column) < test_start_date)
        test = interactions.join(
            start_date_by_user.filter(pl.col("_start_dt_by_user") >= test_start_date),
            on=self.query_column,
            how="inner"
        ).drop("_start_dt_by_user")

        if self.session_id_column:
            interactions = interactions.with_columns(
                pl.when(
                    pl.col(self.timestamp_column) < test_start_date
                )
                .then(False)
                .otherwise(True)
                .alias("is_test")
            )
            interactions = self._recalculate_with_session_id_column(interactions)
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
