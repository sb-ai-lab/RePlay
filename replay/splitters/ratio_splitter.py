from typing import List, Optional, Tuple

from pandas import DataFrame as PandasDataFrame
import pyspark.sql.functions as sf
from pyspark.sql import DataFrame as SparkDataFrame, Window
from pyspark.sql.types import IntegerType

from replay.data import AnyDataFrame
from replay.splitters.base_splitter import Splitter


# pylint: disable=too-few-public-methods, too-many-instance-attributes
class RatioSplitter(Splitter):
    """
    Split interactions into train and test by ratio.

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
    >>> train, test = RatioSplitter(ratio=[0.5]).split(dataset)
    >>> train
        user_id  item_id  timestamp
    0         1        1 2020-01-01
    1         1        2 2020-01-02
    5         2        1 2020-01-06
    6         2        2 2020-01-07
    10        3        1 2020-01-01
    11        3        5 2020-01-02
    >>> test
        user_id  item_id  timestamp
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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        ratio: List[float],
        divide_column: str = "user_id",
        drop_cold_users: bool = False,
        drop_cold_items: bool = False,
        drop_zero_rel_in_test: bool = False,
        user_column: str = "user_id",
        item_column: str = "item_id",
        time_column: str = "timestamp",
        min_interactions_per_group: Optional[int] = None,
        split_by_fraqtions: bool = True,
        session_id_column: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        Args:
            ratio (array of float): Array of test size. Sum must be in :math:`(0, 1)`.
            divide_column (str): Name of column for dividing
                in dataframe, default: ``user_id``.
            drop_cold_users (bool): Drop users from test DataFrame
                which are not in train DataFrame, default: False.
            drop_cold_items (bool): Drop items from test DataFrame
                which are not in train DataFrame, default: False.
            drop_zero_rel_in_test (bool): Flag to remove entries with relevance <= 0
                from the test part of the dataset.
                Default: ``False``.
            user_column (str): Name of user interaction column.
                If ``drop_cold_users`` is ``False``, then you can omit this parameter.
                Default: ``user_id``.
            item_column (str): Name of item interaction column.
                If ``drop_cold_items`` is ``False``, then you can omit this parameter.
                Default: ``item_id``.
            time_column (str): Name of time column,
                default: ``timestamp``.
            min_interactions_per_group (int, optional): minimal required interactions per group to make first split.
                if value is less than min_interactions_per_group, than whole group goes to train.
                If not set, than any amount of interactions will be split.
                default: ``None``.
            split_by_fraqtions (bool): the variable that is responsible for using the split by fractions.
                Split by fractions means that each line is marked with its fraq (line number / number of lines)
                and only those lines with a fraq > test_ratio get into the test.
                Split not by fractions means that the number of rows in the train is calculated by rounding the formula:
                the total number of rows minus the number of rows multiplied by the test ratio.
                The difference between these two methods is that due to rounding in the second method,
                1 more interaction in each group (1 item for each user) falls into the train.
                When split by fractions, these items fall into the test.
                default: ``True``.
            session_id_column (str, optional): Name of session id column, which values can not be split,
                default: ``None``.
            session_id_processing_strategy (str): strategy of processing session if it is split,
                Values: ``train, test``, train: whole split session goes to train. test: same but to test.
                default: ``test``.
        """
        super().__init__(
            drop_cold_users=drop_cold_users,
            drop_cold_items=drop_cold_items,
            drop_zero_rel_in_test=drop_zero_rel_in_test,
            user_col=user_column,
            item_col=item_column,
            timestamp_col=time_column,
            session_id_col=session_id_column,
            session_id_processing_strategy=session_id_processing_strategy,
        )
        self.ratio = list(reversed(ratio))
        self.divide_column = divide_column
        self._precision = 3
        self.min_interactions_per_group = min_interactions_per_group
        self.split_by_fraqtions = split_by_fraqtions
        self._sanity_check()

    def _get_order_of_sort(self) -> list:
        return [self.divide_column, self.timestamp_col]

    def _sanity_check(self) -> None:
        sum_ratio = round(sum(self.ratio), self._precision)
        if sum_ratio <= 0 or sum_ratio >= 1:
            raise ValueError(f"sum of `ratio` list must be in (0, 1); sum={sum_ratio}")

    def _add_time_partition(self, interactions: AnyDataFrame) -> AnyDataFrame:
        if isinstance(interactions, SparkDataFrame):
            return self._add_time_partition_to_spark(interactions)

        return self._add_time_partition_to_pandas(interactions)

    def _add_time_partition_to_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        res = interactions.copy(deep=True)
        res.sort_values(by=[self.divide_column, self.timestamp_col], inplace=True)
        res["row_num"] = res.groupby(self.divide_column, sort=False).cumcount() + 1

        return res

    def _add_time_partition_to_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        res = interactions.withColumn(
            "row_num",
            sf.row_number().over(Window.partitionBy(self.divide_column).orderBy(sf.col(self.timestamp_col))),
        )

        return res

    def _partial_split_fraqtions(
        self, interactions: AnyDataFrame, ratio: float
    ) -> Tuple[AnyDataFrame, AnyDataFrame]:
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
        if self.session_id_col:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions[~interactions["is_test"]].drop(columns=["row_num", "count", "frac", "is_test"])
        test = interactions[interactions["is_test"]].drop(columns=["row_num", "count", "frac", "is_test"])

        return train, test

    def _partial_split_fraqtions_spark(
        self, interactions: SparkDataFrame, train_size: float
    ) -> Tuple[SparkDataFrame, SparkDataFrame]:
        interactions = interactions.withColumn(
            "count", sf.count(self.timestamp_col).over(Window.partitionBy(self.divide_column))
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
        if self.session_id_col:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions.filter("is_test == 0").drop("row_num", "count", "frac", "is_test")
        test = interactions.filter("is_test").drop("row_num", "count", "frac", "is_test")

        return train, test

    def _partial_split(self, interactions: AnyDataFrame, ratio: float) -> Tuple[AnyDataFrame, AnyDataFrame]:
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
        if self.session_id_col:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions[~interactions["is_test"]].drop(columns=["row_num", "count", "train_size", "is_test"])
        test = interactions[interactions["is_test"]].drop(columns=["row_num", "count", "train_size", "is_test"])

        return train, test

    def _partial_split_spark(self, interactions: SparkDataFrame, ratio: float) -> Tuple[SparkDataFrame, SparkDataFrame]:
        interactions = interactions.withColumn(
            "count", sf.count(self.timestamp_col).over(Window.partitionBy(self.divide_column))
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
        if self.session_id_col:
            interactions = self._recalculate_with_session_id_column(interactions)

        train = interactions.filter("is_test == 0").drop("row_num", "count", "train_size", "is_test")
        test = interactions.filter("is_test").drop("row_num", "count", "train_size", "is_test")

        return train, test

    # pylint: disable=invalid-name
    def _core_split(self, log: AnyDataFrame) -> List[AnyDataFrame]:
        sum_ratio = round(sum(self.ratio), self._precision)
        if self.split_by_fraqtions:
            train, test = self._partial_split_fraqtions(log, sum_ratio)
        else:
            train, test = self._partial_split(log, sum_ratio)

        self.min_interactions_per_group = None  # Needed only at first split
        res = []
        for r in self.ratio:
            if self.split_by_fraqtions:
                test, test1 = self._partial_split_fraqtions(test, round(r / sum_ratio, self._precision))
            else:
                test, test1 = self._partial_split(test, round(r / sum_ratio, self._precision))
            res.append(test1)
            sum_ratio -= r
        return [train] + list(reversed(res))