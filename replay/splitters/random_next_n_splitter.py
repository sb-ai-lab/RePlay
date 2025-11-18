from typing import Optional, Tuple

import numpy as np
import pandas as pd
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


class RandomNextNSplitter(Splitter):
    """
    Split interactions by a random position in the user sequence.
    For each user, a random cut index is sampled and the target part consists of
    the next ``N`` interactions starting from this cut; the train part contains
    all interactions before the cut. Interactions after the target window are
    discarded.

    >>> from datetime import datetime
    >>> import pandas as pd
    >>> columns = ["query_id", "item_id", "timestamp"]
    >>> data = [
    ...     (1, 1, "01-01-2020"),
    ...     (1, 2, "02-01-2020"),
    ...     (1, 3, "03-01-2020"),
    ...     (2, 1, "06-01-2020"),
    ...     (2, 2, "07-01-2020"),
    ...     (2, 3, "08-01-2020"),
    ... ]
    >>> dataset = pd.DataFrame(data, columns=columns)
    >>> dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], format="%d-%m-%Y")
    >>> splitter = RandomNextNSplitter(
    ...     N=2,
    ...     divide_column="query_id",
    ...     seed=42,
    ...     query_column="query_id",
    ...     item_column="item_id",
    ... )
    >>> train, test = splitter.split(dataset)
    """

    _init_arg_names = [
        "N",
        "divide_column",
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
        N: Optional[int] = 1,  # noqa: N803
        divide_column: str = "query_id",
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
        :param N: Optional window size. If None, the test set contains all interactions
            from the cut to the end; otherwise the next ``N`` interactions. Must be >= 1.
            Default: 1.
        :param divide_column: Name of the column used to group interactions
            for random cut sampling, default: ``query_id``.
        :param seed: Random seed used to sample cut indices, default: ``None``.
        :param query_column: Name of query interaction column.
        :param drop_cold_users: Drop users from test DataFrame which are not in
            the train DataFrame, default: ``False``.
        :param drop_cold_items: Drop items from test DataFrame which are not in
            the train DataFrame, default: ``False``.
        :param item_column: Name of item interaction column.
            If ``drop_cold_items`` is ``False``, then you can omit this
            parameter. Default: ``item_id``.
        :param timestamp_column: Name of time column, default: ``timestamp``.
        :param session_id_column: Name of session id column whose values cannot
            be split between train/test, default: ``None``.
        :param session_id_processing_strategy: Strategy to process a session if
            it crosses the boundary: ``train`` or ``test``. ``train`` means the
            whole session goes to train, ``test`` — the whole session goes to
            test. Default: ``test``.
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
        self.N = N
        if self.N is not None and self.N < 1:
            msg = "N must be >= 1"
            raise ValueError(msg)
        self.divide_column = divide_column
        self.seed = seed

    def _sample_cuts(self, counts: np.ndarray) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        return rng.randint(0, counts)

    def _partial_split_pandas(
        self,
        interactions: PandasDataFrame,
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        df = interactions.sort_values([self.divide_column, self.timestamp_column])
        df["_event_rank"] = df.groupby(self.divide_column, sort=False).cumcount()

        counts = df.groupby(self.divide_column, sort=False).size()
        cuts = pd.Series(self._sample_cuts(counts.values), index=counts.index)
        df["_cut_index"] = df[self.divide_column].map(cuts)

        if self.N is not None:
            df = df[df["_event_rank"] < df["_cut_index"] + self.N]

        df["is_test"] = df["_event_rank"] >= df["_cut_index"]
        if self.session_id_column:
            df = self._recalculate_with_session_id_column(df)

        train = df[~df["is_test"]][interactions.columns]
        test = df[df["is_test"]][interactions.columns]

        return train, test

    def _partial_split_polars(
        self,
        interactions: PolarsDataFrame,
    ) -> Tuple[PolarsDataFrame, PolarsDataFrame]:
        df = interactions.sort([self.divide_column, self.timestamp_column]).with_columns(
            (pl.col(self.divide_column).cum_count().over(self.divide_column) - 1).alias("_event_rank")
        )

        counts = df.group_by(self.divide_column).len()
        r_values = self._sample_cuts(counts["len"].to_numpy())
        cuts_df = pl.DataFrame(
            {
                self.divide_column: counts[self.divide_column],
                "_cut_index": r_values,
            }
        )
        df = df.join(cuts_df, on=self.divide_column, how="left")

        if self.N is not None:
            df = df.filter(pl.col("_event_rank") < pl.col("_cut_index") + self.N)

        df = df.with_columns((pl.col("_event_rank") >= pl.col("_cut_index")).alias("is_test"))
        if self.session_id_column:
            df = self._recalculate_with_session_id_column(df)

        train = df.filter(~pl.col("is_test")).select(interactions.columns)
        test = df.filter(pl.col("is_test")).select(interactions.columns)

        return train, test

    def _partial_split_spark(
        self,
        interactions: SparkDataFrame,
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        w = Window.partitionBy(self.divide_column).orderBy(self.timestamp_column)
        df = interactions.withColumn("_event_rank", sf.row_number().over(w) - sf.lit(1))

        counts = df.groupBy(self.divide_column).agg(sf.count(sf.lit(1)).alias("_count"))
        seed_lit = sf.lit(self.seed) if self.seed is not None else sf.lit(0)
        cuts = counts.select(
            self.divide_column,
            sf.pmod(
                sf.xxhash64(sf.col(self.divide_column), seed_lit).cast("long"),
                sf.col("_count").cast("long"),
            ).cast("long").alias("_cut_index"),
        )

        df = df.join(cuts, on=self.divide_column, how="left")

        if self.N is not None:
            df = df.filter(sf.col("_event_rank") < sf.col("_cut_index") + sf.lit(self.N))

        df = df.withColumn("is_test", sf.col("_event_rank") >= sf.col("_cut_index"))
        if self.session_id_column:
            df = self._recalculate_with_session_id_column(df)

        train = df.filter(~sf.col("is_test")).select(interactions.columns)
        test = df.filter(sf.col("is_test")).select(interactions.columns)

        return train, test

    def _partial_split(self, interactions: DataFrameLike) -> Tuple[DataFrameLike, DataFrameLike]:
        if isinstance(interactions, PandasDataFrame):
            return self._partial_split_pandas(interactions)
        if isinstance(interactions, PolarsDataFrame):
            return self._partial_split_polars(interactions)
        if isinstance(interactions, SparkDataFrame):
            return self._partial_split_spark(interactions)
        msg = f"{self} is not implemented for {type(interactions)}"
        raise NotImplementedError(msg)

    def _core_split(self, interactions: DataFrameLike) -> Tuple[DataFrameLike, DataFrameLike]:
        return self._partial_split(interactions)
