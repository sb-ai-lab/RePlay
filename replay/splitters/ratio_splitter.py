from typing import List, Optional, Tuple

from replay.splitters.base_splitter import Splitter
from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf
    from pyspark.sql import Window
    from pyspark.sql.types import IntegerType


# pylint: disable=too-few-public-methods, too-many-instance-attributes
class RatioSplitter(Splitter):
    """
    Split interactions into train and test by ratio. Split is made for each user separately.

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
    >>> splitter = RatioSplitter(
    ...     test_size=0.5,
    ...     divide_column="query_id",
    ...     query_column="query_id",
    ...     item_column="item_id"
    ... )
    >>> train, test = splitter.split(dataset)
    >>> train
       query_id  item_id  timestamp
    0         1        1 2020-01-01
    1         1        2 2020-01-02
    5         2        1 2020-01-06
    6         2        2 2020-01-07
    10        3        1 2020-01-01
    11        3        5 2020-01-02
    >>> test
       query_id  item_id  timestamp
    2         1        3 2020-01-03
    3         1        4 2020-01-04
    4         1        5 2020-01-05
    7         2        3 2020-01-08
    8         2        9 2020-01-09
    9         2       10 2020-01-10
    12        3        3 2020-01-03
    13        3        1 2020-01-04
    14        3        2 2020-01-05
    <BLANKLINE>
    """
    _init_arg_names = [
        "test_size",
        "divide_column",
        "drop_cold_users",
        "drop_cold_items",
        "query_column",
        "item_column",
        "timestamp_column",
        "min_interactions_per_group",
        "split_by_fraqtions",
        "session_id_column",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        test_size: float,
        divide_column: str = "query_id",
        drop_cold_users: bool = False,
        drop_cold_items: bool = False,
        query_column: str = "query_id",
        item_column: str = "item_id",
        timestamp_column: str = "timestamp",
        min_interactions_per_group: Optional[int] = None,
        split_by_fraqtions: bool = True,
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param ratio: test size, must be in :math:`(0, 1)`.
        :param divide_column: Name of column for dividing
            in dataframe, default: ``query_id``.
        :param drop_cold_users: Drop users from test DataFrame.
            which are not in train DataFrame, default: False.
        :param drop_cold_items: Drop items from test DataFrame
            which are not in train DataFrame, default: False.
        :param query_column: Name of query interaction column.
            If ``drop_cold_users`` is ``False``, then you can omit this parameter.
            Default: ``query_id``.
        :param item_column: Name of item interaction column.
            If ``drop_cold_items`` is ``False``, then you can omit this parameter.
            Default: ``item_id``.
        :param timestamp_column: Name of time column,
            Default: ``timestamp``.
        :param min_interactions_per_group: minimal required interactions per group to make first split.
            if value is less than min_interactions_per_group, than whole group goes to train.
            If not set, than any amount of interactions will be split.
            default: ``None``.
        :param split_by_fraqtions: the variable that is responsible for using the split by fractions.
            Split by fractions means that each line is marked with its fraq (line number / number of lines)
            and only those lines with a fraq > test_ratio get into the test.
            Split not by fractions means that the number of rows in the train is calculated by rounding the formula:
            the total number of rows minus the number of rows multiplied by the test ratio.
            The difference between these two methods is that due to rounding in the second method,
            1 more interaction in each group (1 item for each user) falls into the train.
            When split by fractions, these items fall into the test.
            default: ``True``.
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
        self.divide_column = divide_column
        self._precision = 3
        self.min_interactions_per_group = min_interactions_per_group
        self.split_by_fraqtions = split_by_fraqtions
        if test_size < 0 or test_size > 1:
            raise ValueError("test_size must between 0 and 1")
        self.test_size = test_size

    def _add_time_partition(self, interactions: DataFrameLike) -> DataFrameLike:
        if isinstance(interactions, SparkDataFrame):
            return self._add_time_partition_to_spark(interactions)

        return self._add_time_partition_to_pandas(interactions)

    def _add_time_partition_to_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        res = interactions.copy(deep=True)
        res.sort_values(by=[self.divide_column, self.timestamp_column], inplace=True)
        res["row_num"] = res.groupby(self.divide_column, sort=False).cumcount() + 1

        return res

    def _add_time_partition_to_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        res = interactions.withColumn(
            "row_num",
            sf.row_number().over(Window.partitionBy(self.divide_column).orderBy(sf.col(self.timestamp_column))),
        )

        return res

    def _partial_split_fraqtions(
        self, interactions: DataFrameLike, ratio: float
    ) -> Tuple[DataFrameLike, DataFrameLike]:
        res = self._add_time_partition(interactions)
        train_size = round(1 - ratio, self._precision)

        if isinstance(res, SparkDataFrame):
            return self._partial_split_fraqtions_spark(res, train_size)

        return self._partial_split_fraqtions_pandas(res, train_size)

    def _partial_split_fraqtions_pandas(
        self, interactions: PandasDataFrame, train_size: float
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        interactions["count"] = interactions.groupby(self.divide_column, sort=False)[self.divide_column].transform(len)
        interactions["frac"] = (interactions["row_num"] / interactions["count"]).round(self._precision)
        if self.min_interactions_per_group is not None:
            interactions["frac"].where(interactions["count"] >= self.min_interactions_per_group, 0, inplace=True)

        interactions["is_test"] = interactions["frac"] > train_size
        if self.session_id_column:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions[~interactions["is_test"]].drop(columns=["row_num", "count", "frac", "is_test"])
        test = interactions[interactions["is_test"]].drop(columns=["row_num", "count", "frac", "is_test"])

        return train, test

    def _partial_split_fraqtions_spark(
        self, interactions: SparkDataFrame, train_size: float
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        interactions = interactions.withColumn(
            "count", sf.count(self.timestamp_column).over(Window.partitionBy(self.divide_column))
        )
        if self.min_interactions_per_group is not None:
            interactions = interactions.withColumn(
                "frac",
                sf.when(
                    sf.col("count") >= self.min_interactions_per_group,
                    sf.round(sf.col("row_num") / sf.col("count"), self._precision),
                ).otherwise(sf.lit(0)),
            )
        else:
            interactions = interactions.withColumn(
                "frac", sf.round(sf.col("row_num") / sf.col("count"), self._precision)
            )

        interactions = interactions.withColumn("is_test", sf.col("frac") > train_size)
        if self.session_id_column:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions.filter("is_test == 0").drop("row_num", "count", "frac", "is_test")
        test = interactions.filter("is_test").drop("row_num", "count", "frac", "is_test")

        return train, test

    def _partial_split(self, interactions: DataFrameLike, ratio: float) -> Tuple[DataFrameLike, DataFrameLike]:
        res = self._add_time_partition(interactions)
        if isinstance(res, SparkDataFrame):
            return self._partial_split_spark(res, ratio)

        return self._partial_split_pandas(res, ratio)

    def _partial_split_pandas(
        self, interactions: PandasDataFrame, ratio: float
    ) -> Tuple[PandasDataFrame, PandasDataFrame]:
        interactions["count"] = interactions.groupby(self.divide_column, sort=False)[self.divide_column].transform(len)
        interactions["train_size"] = interactions["count"] - (interactions["count"] * ratio).astype(int)
        if self.min_interactions_per_group is not None:
            interactions["train_size"].where(
                interactions["count"] >= self.min_interactions_per_group, interactions["count"], inplace=True
            )
        else:
            interactions.loc[
                (interactions["count"] * ratio > 0)
                & (interactions["count"] * ratio < 1)
                & (interactions["train_size"] > 1),
                "train_size",
            ] = (
                interactions["train_size"] - 1
            )  # pylint: disable=C0325

        interactions["is_test"] = interactions["row_num"] > interactions["train_size"]
        if self.session_id_column:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions[~interactions["is_test"]].drop(columns=["row_num", "count", "train_size", "is_test"])
        test = interactions[interactions["is_test"]].drop(columns=["row_num", "count", "train_size", "is_test"])

        return train, test

    def _partial_split_spark(self, interactions: SparkDataFrame, ratio: float) -> Tuple[SparkDataFrame, SparkDataFrame]:
        interactions = interactions.withColumn(
            "count", sf.count(self.timestamp_column).over(Window.partitionBy(self.divide_column))
        )
        if self.min_interactions_per_group is not None:
            interactions = interactions.withColumn(
                "train_size",
                sf.when(
                    sf.col("count") >= self.min_interactions_per_group,
                    sf.col("count") - (sf.col("count") * ratio).cast(IntegerType()),
                ).otherwise(sf.col("count")),
            )
        else:
            interactions = interactions.withColumn(
                "train_size", sf.col("count") - (sf.col("count") * ratio).cast(IntegerType())
            ).withColumn(
                "train_size",
                sf.when(
                    (sf.col("count") * ratio > 0) & (sf.col("count") * ratio < 1) & (sf.col("train_size") > 1),
                    sf.col("train_size") - 1,
                ).otherwise(sf.col("train_size")),
            )
        interactions = interactions.withColumn("is_test", sf.col("row_num") > sf.col("train_size"))
        if self.session_id_column:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions.filter("is_test == 0").drop("row_num", "count", "train_size", "is_test")
        test = interactions.filter("is_test").drop("row_num", "count", "train_size", "is_test")

        return train, test

    # pylint: disable=invalid-name
    def _core_split(self, interactions: DataFrameLike) -> List[DataFrameLike]:
        if self.split_by_fraqtions:
            return self._partial_split_fraqtions(interactions, self.test_size)
        else:
            return self._partial_split(interactions, self.test_size)
