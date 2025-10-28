from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np
import polars as pl

from replay.utils import (
    PYSPARK_AVAILABLE,
    DataFrameLike,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)

from .base_splitter import Splitter

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf
    from pyspark.sql import Window


class RandomTargetNextNSplitter(Splitter):
    """
    Split interactions by a random cut per user.

    For each user, a random index r (0-based) is sampled uniformly from
    [0, user_events_count - 1]. All events with rank < r go to the input
    (train) subset; the next N events [r, r+N) form the target (test)
    subset. Events with rank >= r + N are dropped. Filtering of cold
    users/items is applied by the base ``Splitter`` in ``split``.

    Example
    -------
    >>> import pandas as pd
    >>> columns = ["query_id", "item_id", "timestamp"]
    >>> data = [
    ...     (1, 1, "2020-01-01"),
    ...     (1, 2, "2020-01-02"),
    ...     (1, 3, "2020-01-03"),
    ...     (2, 10, "2020-01-01"),
    ...     (2, 20, "2020-01-02"),
    ...     (2, 30, "2020-01-03"),
    ... ]
    >>> dataset = pd.DataFrame(data, columns=columns)
    >>> dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], format="%Y-%m-%d")
    >>> splitter = RandomTargetNextNSplitter(N=1, seed=42)
    >>> train_input, test_target = splitter.split(dataset)
    """

    _init_arg_names = [
        "N",
        "seed",
        "drop_cold_users",
        "drop_cold_items",
        "query_column",
        "item_column",
        "timestamp_column",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    def __init__(
        self,
        N: int = 1,
        seed: Optional[int] = None,
        query_column: str = "query_id",
        drop_cold_users: bool = False,
        drop_cold_items: bool = False,
        item_column: str = "item_id",
        timestamp_column: str = "timestamp",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        Parameters
        ----------
        N : int, default 1
            Number of consecutive target events after the random cut position r (per user).
        seed : int | None, default None
            Random seed used to sample r for each user.
        query_column : str, default "query_id"
            User (query) column name.
        drop_cold_users : bool, default False
            Remove from test users that are not present in train (handled in base Splitter).
        drop_cold_items : bool, default False
            Remove from test items that are not present in train (handled in base Splitter).
        item_column : str, default "item_id"
            Item column name.
        timestamp_column : str, default "timestamp"
            Timestamp column name.
        session_id_column : str | None, default None
            Session id column that must not be split. If a session is split,
            the whole session moves to train or test according to
            ``session_id_processing_strategy``.
        session_id_processing_strategy : {"train", "test"}, default "test"
            Where to move split sessions: to train or to test.
        """
        super().__init__(
            drop_cold_users=drop_cold_users,
            drop_cold_items=drop_cold_items,
            query_column=query_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            session_id_column=session_id_column,
            session_id_processing_strategy=session_id_processing_strategy,
        )
        self.N = int(N)
        if self.N < 1:
            raise ValueError("N must be >= 1")
        self.seed = seed

    def _sample_cuts(self, counts: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        return rng.randint(0, counts)

    def _partial_split_pandas(
        self,
        interactions: PandasDataFrame,
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        df = interactions.sort_values([self.query_column, self.timestamp_column])
        df["_rn"] = df.groupby(self.query_column, sort=False).cumcount()

        counts = df.groupby(self.query_column, sort=False)["_rn"].max() + 1
        r_values = self._sample_cuts(counts.values)
        r_map = dict(zip(counts.index, r_values))
        df["_r"] = df[self.query_column].map(r_map)

        input_mask = df["_rn"] < df["_r"]
        target_mask = (df["_rn"] >= df["_r"]) & (df["_rn"] < df["_r"] + self.N)

        input_df = df.loc[input_mask, interactions.columns]
        target_df = df.loc[target_mask, interactions.columns]

        return input_df, target_df

    def _partial_split_polars(
        self,
        interactions: PolarsDataFrame,
    ) -> Tuple[PolarsDataFrame, PolarsDataFrame]:
        df = interactions.sort([self.query_column, self.timestamp_column]).with_columns(
            pl.cum_count().over(self.query_column).alias("_rn")
        )

        counts = df.group_by(self.query_column).agg(pl.len().alias("_cnt"))
        users = counts.get_column(self.query_column).to_list()
        cnts = counts.get_column("_cnt").to_numpy()

        r_values = self._sample_cuts(cnts)
        r_map = pl.DataFrame({self.query_column: users, "_r": r_values})

        df = df.join(r_map, on=self.query_column, how="left")

        input_mask = pl.col("_rn") < pl.col("_r")
        target_mask = (pl.col("_rn") >= pl.col("_r")) & (
            pl.col("_rn") < pl.col("_r") + self.N
        )

        input_df = df.filter(input_mask).select(interactions.columns)
        target_df = df.filter(target_mask).select(interactions.columns)

        return input_df, target_df

    def _partial_split_spark(
        self,
        interactions: SparkDataFrame,
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        w = Window.partitionBy(self.query_column).orderBy(sf.col(self.timestamp_column))
        df = interactions.withColumn("_rn", sf.row_number().over(w) - sf.lit(1))

        counts = (
            interactions.groupBy(self.query_column)
            .count()
            .withColumnRenamed("count", "_cnt")
        )

        r_per_user = counts.withColumn(
            "_r", sf.floor(sf.rand(self.seed) * sf.col("_cnt")).cast("long")
        ).select(self.query_column, "_r")

        df = df.join(r_per_user, on=self.query_column, how="left")

        input_cond = sf.col("_rn") < sf.col("_r")
        target_cond = (sf.col("_rn") >= sf.col("_r")) & (
            sf.col("_rn") < sf.col("_r") + sf.lit(self.N)
        )

        input_df = df.where(input_cond).select(*interactions.columns)
        target_df = df.where(target_cond).select(*interactions.columns)

        return input_df, target_df

    def _partial_split(
        self, interactions: DataFrameLike
    ) -> Tuple[DataFrameLike, DataFrameLike]:
        if isinstance(interactions, PandasDataFrame):
            return self._partial_split_pandas(interactions)
        if isinstance(interactions, PolarsDataFrame):
            return self._partial_split_polars(interactions)
        if isinstance(interactions, SparkDataFrame):
            return self._partial_split_spark(interactions)
        raise NotImplementedError(f"{self} is not implemented for {type(interactions)}")

    def _core_split(
        self, interactions: DataFrameLike
    ) -> Tuple[DataFrameLike, DataFrameLike]:
        return self._partial_split(interactions)
