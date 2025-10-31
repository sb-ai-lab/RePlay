from typing import Optional, Tuple

import numpy as np

from replay.utils import (
    DataFrameLike,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)

from .base_splitter import Splitter


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

        counts = df.groupby(self.divide_column, sort=False)["_event_rank"].max() + 1
        r_values = self._sample_cuts(counts.values)
        r_map = dict(zip(counts.index, r_values))
        df["_cut_index"] = df[self.divide_column].map(r_map)

        if self.N is not None:
            df = df[df["_event_rank"] < df["_cut_index"] + self.N]

        df["is_test"] = df["_event_rank"] >= df["_cut_index"]
        if self.session_id_column:
            df = self._recalculate_with_session_id_column(df)

        train = df[~df["is_test"]][interactions.columns]
        test = df[df["is_test"]][interactions.columns]

        return train, test

    def _partial_split(self, interactions: DataFrameLike) -> Tuple[DataFrameLike, DataFrameLike]:
        if isinstance(interactions, PandasDataFrame):
            return self._partial_split_pandas(interactions)
        if isinstance(interactions, PolarsDataFrame):
            msg = f"{self} is not implemented for {type(interactions)}"
            raise NotImplementedError(msg)
        if isinstance(interactions, SparkDataFrame):
            msg = f"{self} is not implemented for {type(interactions)}"
            raise NotImplementedError(msg)
        msg = f"{self} is not implemented for {type(interactions)}"
        raise NotImplementedError(msg)

    def _core_split(self, interactions: DataFrameLike) -> Tuple[DataFrameLike, DataFrameLike]:
        return self._partial_split(interactions)
