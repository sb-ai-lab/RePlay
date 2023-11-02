from datetime import datetime
from typing import List, Optional, Tuple, Union

from pandas import DataFrame as PandasDataFrame
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame as SparkDataFrame, Window

from replay.data import AnyDataFrame
from replay.splitters.base_splitter import Splitter


# pylint: disable=too-few-public-methods
class TimeSplitter(Splitter):
    """
    Split interactions by time.

    >>> from datetime import datetime
    >>> import pandas as pd
    >>> columns = ["query_id", "item_id", "timestamp"]
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
       query_id  item_id  timestamp
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
    ...     time_threshold=datetime.strptime("2020-01-04", "%Y-%M-%d")
    ... ).split(dataset)
    >>> train
       query_id  item_id  timestamp
    0         1        1 2020-01-01
    1         1        2 2020-01-02
    2         1        3 2020-01-03
    3         1        4 2020-01-04
    10        3        1 2020-01-01
    11        3        5 2020-01-02
    12        3        3 2020-01-03
    13        3        1 2020-01-04
    >>> test
       query_id  item_id  timestamp
    4         1        5 2020-01-05
    5         2        1 2020-01-06
    6         2        2 2020-01-07
    7         2        3 2020-01-08
    8         2        9 2020-01-09
    9         2       10 2020-01-10
    14        3        2 2020-01-05
    <BLANKLINE>
    """
    _init_arg_names = [
        "time_threshold",
        "drop_cold_users",
        "drop_cold_items",
        "query_column",
        "item_column",
        "timestamp_column",
        "session_id_column",
        "session_id_processing_strategy",
        "time_column_format",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        time_threshold: Union[datetime, str, int, float],
        query_column: str = "query_id",
        drop_cold_users: bool = False,
        drop_cold_items: bool = False,
        item_column: str = "item_id",
        timestamp_column: str = "timestamp",
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
        time_column_format: str = "%Y-%m-%d %H:%M:%S",
    ):
        """
        :param time_threshold: Array of test threshold.
        :param query_column: Name of user interaction column.
        :param drop_cold_users: Drop users from test DataFrame.
            which are not in train DataFrame, default: False.
        :param drop_cold_items: Drop items from test DataFrame
            which are not in train DataFrame, default: False.
        :param item_column: Name of item interaction column.
            If ``drop_cold_items`` is ``False``, then you can omit this parameter.
            Default: ``item_id``.
        :param timestamp_column: Name of time column,
            Default: ``timestamp``.
        :param session_id_column: Name of session id column, which values can not be split,
            default: ``None``.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
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
        self._precision = 3
        self.time_column_format = time_column_format
        if isinstance(time_threshold, float) and (time_threshold < 0 or time_threshold > 1):
            raise ValueError("test_size must between 0 and 1")
        self.time_threshold = time_threshold

    def _partial_split(
        self, interactions: AnyDataFrame, threshold: Union[datetime, str, int]
    ) -> Tuple[AnyDataFrame, AnyDataFrame]:
        if isinstance(threshold, str):
            threshold = datetime.strptime(threshold, self.time_column_format)
        if isinstance(interactions, SparkDataFrame):
            return self._partial_split_spark(interactions, threshold)

        return self._partial_split_pandas(interactions, threshold)

    def _partial_split_pandas(
        self, interactions: PandasDataFrame, threshold: Union[datetime, str, int]
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        res = interactions.copy(deep=True)
        if isinstance(threshold, float):
            res.sort_values(self.timestamp_column, inplace=True)
            test_start_ind = int(res.shape[0] * (1 - threshold))
            test_start = res.iloc[test_start_ind][self.timestamp_column]
            res["is_test"] = res[self.timestamp_column] >= test_start
        else:
            res["is_test"] = res[self.timestamp_column] >= threshold

        if self.session_id_column:
            res = self._recalculate_with_session_id_column(res)

        train = res[~res["is_test"]].drop(columns=["is_test"])
        test = res[res["is_test"]].drop(columns=["is_test"])

        return train, test

    def _partial_split_spark(
        self, interactions: SparkDataFrame, threshold: Union[datetime, str, int]
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        if isinstance(threshold, float):
            dates = interactions.select(self.timestamp_column).withColumn(
                "_row_number_by_ts", sf.row_number().over(Window.orderBy(self.timestamp_column))
            )
            test_start = int(dates.count() * (1 - threshold)) + 1
            test_start = (
                dates.filter(sf.col("_row_number_by_ts") == test_start)
                .select(self.timestamp_column)
                .collect()[0][0]
            )
            res = interactions.withColumn("is_test", sf.col(self.timestamp_column) >= test_start)
        else:
            res = interactions.withColumn("is_test", sf.col(self.timestamp_column) >= threshold)

        if self.session_id_column:
            res = self._recalculate_with_session_id_column(res)
        train = res.filter("is_test == 0").drop("is_test")
        test = res.filter("is_test").drop("is_test")

        return train, test

    def _core_split(self, interactions: AnyDataFrame) -> List[AnyDataFrame]:
        return self._partial_split(interactions, self.time_threshold)
