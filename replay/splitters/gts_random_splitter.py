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
from .time_splitter import TimeSplitter

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf
    from pyspark.sql import Window


class GlobalTemporalSplitter(Splitter):
    """
    Split interactions by a global time threshold and then form a per-user
    holdout prefix by selecting a random target position inside the test part.

    The split happens in two steps:
    1) Global time split (delegated to `TimeSplitter`): all interactions with
       `timestamp >= time_threshold` go to the initial test, the rest to train.
    2) For each user in the initial test, a random target rank `r` is sampled
       in the range [1..num_user_events]. The final test (holdout) contains
       only the user's interactions with rank `<= r` (i.e. all interactions up
       to and including the random target). Interactions after the target are
       dropped.

    Example
    -------
    >>> from datetime import datetime
    >>> import pandas as pd
    >>> data = pd.DataFrame(
    ...     {
    ...         "query_id": [1, 1, 1, 2, 2, 2],
    ...         "item_id": [1, 2, 3, 1, 2, 3],
    ...         "timestamp": [1, 2, 3, 3, 2, 1],
    ...     }
    ... )
    >>> splitter = GlobalTemporalSplitter(
    ...     time_threshold=datetime.fromtimestamp(2),
    ...     seed=42,
    ...     query_column="query_id",
    ...     item_column="item_id",
    ...     timestamp_column="timestamp",
    ... )
    >>> train, test = splitter.split(data)

    Notes
    -----
    - Cold users/items can be filtered automatically after the split via the
      base `Splitter.split()` pipeline depending on `drop_cold_users` and
      `drop_cold_items` flags.
    - The `seed` parameter makes the random target selection reproducible.

    Parameters
    ----------
    time_threshold : datetime | str | float
        Threshold used by the underlying `TimeSplitter`. If float in (0, 1],
        it acts as a ratio for the test size; if str, parsed with
        `time_column_format`.
    query_column : str, default "query_id"
        Name of user column.
    drop_cold_users : bool, default False
        Drop test interactions for users absent in train.
    drop_cold_items : bool, default False
        Drop test interactions for items absent in train.
    item_column : str, default "item_id"
        Name of item column.
    timestamp_column : str, default "timestamp"
        Name of timestamp column.
    seed : int | None, default None
        Random seed used for per-user target rank sampling.
    session_id_column : str | None, default None
        Session id column; if provided, session consistency rules are applied
        by the base class utilities.
    session_id_processing_strategy : {"train", "test"}, default "test"
        Where to move split sessions when session consistency is enforced.
    time_column_format : str, default "%Y-%m-%d %H:%M:%S"
        Format for parsing string thresholds.
    """
    
    _init_arg_names = [
        "time_threshold",
        "drop_cold_users",
        "drop_cold_items",
        "query_column",
        "item_column",
        "timestamp_column",
        "seed",
        "session_id_column",
        "session_id_processing_strategy",
        "time_column_format",
    ]

    def __init__(
        self,
        time_threshold: Union[datetime, str, float],
        query_column: str = "query_id",
        drop_cold_users: bool = False,
        drop_cold_items: bool = False,
        item_column: str = "item_id",
        timestamp_column: str = "timestamp",
        seed: Optional[int] = None,
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
        time_column_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        super().__init__(
            drop_cold_users=drop_cold_users,
            drop_cold_items=drop_cold_items,
            query_column=query_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
            session_id_column=session_id_column,
            session_id_processing_strategy=session_id_processing_strategy,
        )
        self.seed = seed
        self._precision = 3
        self.time_column_format = time_column_format
        if isinstance(time_threshold, float) and (time_threshold < 0 or time_threshold > 1):
            msg = "time_threshold must be between 0 and 1"
            raise ValueError(msg)
        self.time_threshold = time_threshold
    
    def _select_random_holdout_pandas(
        self,
        test: PandasDataFrame,
    ) -> PandasDataFrame:
        holdout = test.sort_values([self.query_column, self.timestamp_column])
        holdout["_rn"] = holdout.groupby(self.query_column, sort=False).cumcount() + 1
        
        sizes = holdout.groupby(self.query_column, sort=False)[self.query_column].size()
        rng = np.random.RandomState(self.seed)
        target_rank= rng.randint(1, sizes.values + 1)
        r_map = dict(zip(sizes.index, target_rank))
        holdout["_r"] = holdout[self.query_column].map(r_map)
        
        holdout = holdout[holdout["_rn"] <= holdout["_r"]]
        holdout = holdout.drop(columns=["_rn", "_r"])
        return holdout
    
    def _select_random_holdout_polars(
        self,
        test: PolarsDataFrame,
    ) -> PolarsDataFrame:
        holdout = test.sort(self.timestamp_column).with_columns(
            pl.cum_count().over(self.query_column).alias("_rn")
        )
        counts = holdout.group_by(self.query_column).agg(pl.len().alias("_cnt"))
        users = counts.get_column(self.query_column).to_list()
        sizes = counts.get_column("_cnt").to_numpy()
        rng = np.random.RandomState(self.seed)
        target_ranks = rng.randint(1, sizes + 1)
        r_map = pl.DataFrame({self.query_column: users, "_r": target_ranks})
        
        holdout = holdout.join(r_map, on=self.query_column, how="left")
        holdout = holdout.filter(pl.col("_rn") <= pl.col("_r")).select(test.columns)
        return holdout
         
    def _select_random_holdout_spark(
        self,
        test: SparkDataFrame,
    ) -> SparkDataFrame:
        w = Window.partitionBy(self.query_column).orderBy(sf.col(self.timestamp_column))
        holdout = test.withColumn("_rn", sf.row_number().over(w))
        
        r_per_user = (
            test.groupBy(self.query_column)
            .count()
            .withColumnRenamed("count", "_cnt")
            .withColumn("_r", sf.floor(sf.rand(self.seed) * sf.col("_cnt")) + sf.lit(1))
            .select(self.query_column, "_r")
        )
        
        holdout = (
            holdout.join(r_per_user, on=self.query_column, how="left")
            .filter(sf.col("_rn") <= sf.col("_r"))
            .select(test.columns)
        )
        return holdout
        
    def _partial_split(
        self, interactions: DataFrameLike, time_threshold: Union[datetime, str, int, float]
    ) -> Tuple[DataFrameLike, DataFrameLike]:
        ts = TimeSplitter(
            time_threshold=time_threshold,
            query_column=self.query_column,
            drop_cold_users=self.drop_cold_users,
            drop_cold_items=self.drop_cold_items,
            item_column=self.item_column,
            timestamp_column=self.timestamp_column,
            session_id_column=self.session_id_column,
            session_id_processing_strategy=self.session_id_processing_strategy,
            time_column_format=self.time_column_format,
        )
        train, test =  ts._core_split(interactions)
        
        if isinstance(test, PandasDataFrame):
            test = self._select_random_holdout_pandas(test)
        elif isinstance(test, PolarsDataFrame):
            test = self._select_random_holdout_polars(test)
        elif isinstance(test, SparkDataFrame):
            test = self._select_random_holdout_spark(test)
        else:
            msg = f"{self} is not implemented for {type(interactions)}"
            raise NotImplementedError(msg)
            
        return train, test

    def _core_split(self, interactions: DataFrameLike) -> Tuple[DataFrameLike]:
        return self._partial_split(interactions, self.time_threshold)
