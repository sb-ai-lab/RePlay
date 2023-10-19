from datetime import datetime
from typing import List, Optional, Tuple, Union

from pandas import DataFrame as PandasDataFrame
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame as SparkDataFrame

from replay.data import AnyDataFrame
from replay.splitters.base_splitter import Splitter

# pylint: disable=too-few-public-methods
class TimeSplitter(Splitter):
    """
    Split interactions by time.

    >>> from datetime import datetime
    >>> import pandas as pd
    >>> columns = ["user_id", "item_id", "timestamp"]
    >>> data = [
    ...     (1, 1, "01-01-2020"),
    ...     (1, 2, "02-01-2020"),
    ...     (1, 3, "03-01-2020"),
    ...     (1, 4, "04-01-2020"),
    ...     (1, 5, "05-01-2020"),
    ...     (2, 1, "06-01-2020"),
    ...     (2, 2, "07-01-2020"),
    ...     (2, 3, "08-01-2020"),
    ...     (2, 9, "09-01-2020"),
    ...     (2, 10, "10-01-2020"),
    ...     (3, 1, "01-01-2020"),
    ...     (3, 5, "02-01-2020"),
    ...     (3, 3, "03-01-2020"),
    ...     (3, 1, "04-01-2020"),
    ...     (3, 2, "05-01-2020"),
    ... ]
    >>> dataset = pd.DataFrame(data, columns=columns)
    >>> dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], format="%d-%m-%Y")
    >>> dataset
        user_id  item_id  timestamp
    0         1        1 2020-01-01
    1         1        2 2020-01-02
    2         1        3 2020-01-03
    3         1        4 2020-01-04
    4         1        5 2020-01-05
    5         2        1 2020-01-06
    6         2        2 2020-01-07
    7         2        3 2020-01-08
    8         2        9 2020-01-09
    9         2       10 2020-01-10
    10        3        1 2020-01-01
    11        3        5 2020-01-02
    12        3        3 2020-01-03
    13        3        1 2020-01-04
    14        3        2 2020-01-05
    >>> train, test = TimeSplitter(
    ...     time_threshold=[datetime.strptime("2020-01-04", "%Y-%M-%d")]
    ... ).split(dataset)
    >>> train
        user_id  item_id  timestamp
    0         1        1 2020-01-01
    1         1        2 2020-01-02
    2         1        3 2020-01-03
    3         1        4 2020-01-04
    10        3        1 2020-01-01
    11        3        5 2020-01-02
    12        3        3 2020-01-03
    13        3        1 2020-01-04
    >>> test
        user_id  item_id  timestamp
    4         1        5 2020-01-05
    5         2        1 2020-01-06
    6         2        2 2020-01-07
    7         2        3 2020-01-08
    8         2        9 2020-01-09
    9         2       10 2020-01-10
    14        3        2 2020-01-05
    <BLANKLINE>
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        time_threshold: List[Union[datetime, str, int]],
        user_col: str = "user_id",
        drop_cold_users: bool = False,
        drop_cold_items: bool = False,
        drop_zero_rel_in_test: bool = False,
        item_col: str = "item_id",
        timestamp_col: str = "timestamp",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        Args:
            time_threshold (array[datetime, str, int]): Array of test threshold
            user_col (str): Name of user interaction column.
            drop_cold_users (bool): Drop users from test DataFrame
                which are not in train DataFrame, default: False.
            drop_cold_items (bool): Drop items from test DataFrame
                which are not in train DataFrame, default: False.
            drop_zero_rel_in_test (bool): Flag to remove entries with relevance <= 0
                from the test part of the dataset.
                Default: ``False``.
            item_col (str): Name of item interaction column.
                If ``drop_cold_items`` is ``False``, then you can omit this parameter.
                Default: ``item_id``.
            timestamp_col (str): Name of time column,
                default: ``timestamp``.
            session_id_col (str, optional): Name of session id column, which values can not be split,
                default: ``None``.
            session_id_processing_strategy (str): strategy of processing session if it is split,
                Values: ``train, test``, train: whole split session goes to train. test: same but to test.
                default: ``test``.
        """
        super().__init__(
            drop_cold_users=drop_cold_users,
            drop_cold_items=drop_cold_items,
            user_col=user_col,
            item_col=item_col,
            drop_zero_rel_in_test=drop_zero_rel_in_test,
            timestamp_col=timestamp_col,
            session_id_col=session_id_col,
            session_id_processing_strategy=session_id_processing_strategy,
        )
        self.time_threshold = time_threshold
        self.user_column = user_col

    def _get_order_of_sort(self) -> list:
        return [self.user_column, self.timestamp_col]

    def _partial_split(
        self, interactions: AnyDataFrame, threshold: Union[datetime, str, int]
    ) -> Tuple[AnyDataFrame, AnyDataFrame]:
        if isinstance(interactions, SparkDataFrame):
            return self._partial_split_spark(interactions, threshold)

        return self._partial_split_pandas(interactions, threshold)

    def _partial_split_pandas(
        self, interactions: PandasDataFrame, threshold: Union[datetime, str, int]
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        res = interactions.copy(deep=True)
        res["is_test"] = res[self.timestamp_col] >= threshold
        if self.session_id_col:
            res = self._recalculate_with_session_id_column(res)

        train = res[~res["is_test"]].drop(columns=["is_test"])
        test = res[res["is_test"]].drop(columns=["is_test"])

        return train, test

    def _partial_split_spark(
        self, interactions: SparkDataFrame, threshold: Union[datetime, str, int]
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        res = interactions.withColumn("is_test", sf.col(self.timestamp_col) >= sf.lit(threshold))
        if self.session_id_col:
            res = self._recalculate_with_session_id_column(res)
        train = res.filter("is_test == 0").drop("is_test")
        test = res.filter("is_test").drop("is_test")

        return train, test

    def _core_split(self, log: AnyDataFrame) -> List[AnyDataFrame]:
        res = []
        for threshold in self.time_threshold:
            train, log = self._partial_split(log, threshold)
            res.append(train)
        res.append(log)
        return res
