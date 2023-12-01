"""
This splitter split data by two columns.
"""
from typing import Optional, Union

from replay.splitters.base_splitter import Splitter, SplitterReturnType
from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf
    from pyspark.sql import Window


# pylint: disable=too-few-public-methods
class TwoStageSplitter(Splitter):
    """
    Split data by two columns.
    First step: takes `first_divide_size` distinct values of `first_divide_column` to the test split.
    Second step: takes `second_divide_size` of `second_divide_column` among the data
    provided after first step to the test split.

    Example:

    >>> from replay.utils.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> from replay.splitters import TwoStageSplitter
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"query_id": [1,1,1,2,2,2],
    ...    "item_id": [1,2,3,1,2,3],
    ...    "relevance": [1,2,3,4,5,6],
    ...    "timestamp": [1,2,3,3,2,1]})
    >>> data_frame
       query_id  item_id  relevance  timestamp
    0         1         1          1          1
    1         1         2          2          2
    2         1         3          3          3
    3         2         1          4          3
    4         2         2          5          2
    5         2         3          6          1
    >>> train, test = TwoStageSplitter(first_divide_size=1, second_divide_size=2, seed=42).split(data_frame)
    >>> test
       query_id  item_id  relevance  timestamp
    3         2         1          4          3
    4         2         2          5          2

    >>> train, test = TwoStageSplitter(first_divide_size=0.5, second_divide_size=2, seed=42).split(data_frame)
    >>> test
       query_id  item_id  relevance  timestamp
    3         2         1          4          3
    4         2         2          5          2

    >>> train, test = TwoStageSplitter(first_divide_size=0.5, second_divide_size=0.7, seed=42).split(data_frame)
    >>> test
       query_id  item_id  relevance  timestamp
    3         2         1          4          3
    4         2         2          5          2
    """

    _init_arg_names = [
        "first_divide_size",
        "second_divide_size",
        "first_divide_column",
        "second_divide_column",
        "shuffle",
        "drop_cold_users",
        "drop_cold_items",
        "seed",
        "query_column",
        "item_column",
        "timestamp_column",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        first_divide_size: Union[float, int],
        second_divide_size: Union[float, int],
        first_divide_column: str = "query_id",
        second_divide_column: str = "item_id",
        shuffle=False,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
        query_column: str = "query_id",
        item_column: Optional[str] = "item_id",
        timestamp_column: Optional[str] = "timestamp",
    ):
        """
        :param second_divide_size: fraction or a number of items per user
        :param first_divide_size: similar to ``item_test_size``,
            but corresponds to the number of users.
            ``None`` is all available users.
        :param shuffle: take random items and not last based on ``timestamp``.
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        :param query_column: query id column name
        :param item_column: item id column name
        :param timestamp_column: timestamp column name
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            query_column=query_column,
            item_column=item_column,
            timestamp_column=timestamp_column,
        )
        self.first_divide_column = first_divide_column
        self.second_divide_column = second_divide_column
        self.first_divide_size = first_divide_size
        self.second_divide_size = second_divide_size
        self.shuffle = shuffle
        self.seed = seed

    def _get_test_values(
        self,
        interactions: DataFrameLike,
    ) -> DataFrameLike:
        """
        :param interactions: input DataFrame
        :return: Spark DataFrame with single column `first_divide_column`
        """
        if isinstance(interactions, SparkDataFrame):
            all_values = interactions.select(self.first_divide_column).distinct()
            user_count = all_values.count()
        else:
            all_values = PandasDataFrame(
                interactions[self.first_divide_column].unique(), columns=[self.first_divide_column]
            )
            user_count = len(all_values)

        value_error = False
        if isinstance(self.first_divide_size, int):
            if 1 <= self.first_divide_size < user_count:
                test_user_count = self.first_divide_size
            else:
                value_error = True
        else:
            if 1 > self.first_divide_size > 0:
                test_user_count = user_count * self.first_divide_size
            else:
                value_error = True
        if value_error:
            raise ValueError(
                f"""
            Invalid value for user_test_size: {self.first_divide_size}
            """
            )
        if isinstance(interactions, SparkDataFrame):
            test_users = (
                all_values.withColumn("_rand", sf.rand(self.seed))
                .withColumn(
                    "_row_num", sf.row_number().over(Window.orderBy("_rand"))
                )
                .filter(f"_row_num <= {test_user_count}")
                .drop("_rand", "_row_num")
            )
        else:
            test_users = all_values.sample(n=int(test_user_count), random_state=self.seed)

        return test_users

    def _split_proportion_spark(self, interactions: SparkDataFrame) -> Union[SparkDataFrame, SparkDataFrame]:
        counts = interactions.groupBy(self.first_divide_column).count()
        test_users = self._get_test_values(interactions).withColumn(
            "is_test", sf.lit(True)
        )
        if self.shuffle:
            res = self._add_random_partition_spark(
                interactions.join(test_users, how="left", on=self.first_divide_column)
            )
        else:
            res = self._add_time_partition_spark(
                interactions.join(test_users, how="left", on=self.first_divide_column),
                query_column=self.query_column,
            )

        res = res.join(counts, on=self.first_divide_column, how="left")
        res = res.withColumn("_frac", sf.col("_row_num") / sf.col("count"))
        res = res.na.fill({"is_test": False})

        train = res.filter(
            f"""
                    _frac > {self.second_divide_size} OR
                    NOT is_test
                """
        ).drop("_rand", "_row_num", "count", "_frac", "is_test")
        test = res.filter(
            f"""
                    _frac <= {self.second_divide_size} AND
                    is_test
                """
        ).drop("_rand", "_row_num", "count", "_frac", "is_test")

        return train, test

    def _split_proportion_pandas(self, interactions: PandasDataFrame) -> Union[PandasDataFrame, PandasDataFrame]:
        counts = interactions.groupby(self.first_divide_column).agg(
            count=(self.first_divide_column, "count")
        ).reset_index()
        test_users = self._get_test_values(interactions)
        test_users["is_test"] = True
        if self.shuffle:
            res = self._add_random_partition_pandas(
                interactions.merge(test_users, how="left", on=self.first_divide_column)
            )
        else:
            res = self._add_time_partition_pandas(
                interactions.merge(test_users, how="left", on=self.first_divide_column),
                query_column=self.query_column,
            )
        res["is_test"].fillna(False, inplace=True)
        res = res.merge(counts, on=self.first_divide_column, how="left")
        res["_frac"] = res["_row_num"] / res["count"]
        train = res[(res["_frac"] > self.second_divide_size) | (~res["is_test"])].drop(
            columns=["_row_num", "count", "_frac", "is_test"]
        )
        test = res[(res["_frac"] <= self.second_divide_size) & (res["is_test"])].drop(
            columns=["_row_num", "count", "_frac", "is_test"]
        )

        return train, test

    def _split_proportion(self, interactions: DataFrameLike) -> SplitterReturnType:
        """
        Proportionate split

        :param interactions: input DataFrame `[self.first_divide_column, self.item_column, self.date_column, relevance]`
        :return: train and test DataFrames
        """
        if isinstance(interactions, SparkDataFrame):
            return self._split_proportion_spark(interactions)
        else:
            return self._split_proportion_pandas(interactions)

    def _split_quantity_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        test_users = self._get_test_values(interactions).withColumn(
            "is_test", sf.lit(True)
        )
        if self.shuffle:
            res = self._add_random_partition_spark(
                interactions.join(test_users, how="left", on=self.first_divide_column)
            )
        else:
            res = self._add_time_partition_spark(
                interactions.join(test_users, how="left", on=self.first_divide_column),
                query_column=self.query_column,
            )
        res = res.na.fill({"is_test": False})
        train = res.filter(
            f"""
                    _row_num > {self.second_divide_size} OR
                    NOT is_test
                """
        ).drop("_rand", "_row_num", "is_test")
        test = res.filter(
            f"""
                    _row_num <= {self.second_divide_size} AND
                    is_test
                """
        ).drop("_rand", "_row_num", "is_test")

        return train, test

    def _split_quantity_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        test_users = self._get_test_values(interactions)
        test_users["is_test"] = True
        if self.shuffle:
            res = self._add_random_partition_pandas(
                interactions.merge(test_users, how="left", on=self.first_divide_column)
            )
        else:
            res = self._add_time_partition_pandas(
                interactions.merge(test_users, how="left", on=self.first_divide_column),
                query_column=self.query_column,
            )
        res["is_test"].fillna(False, inplace=True)
        train = res[(res["_row_num"] > self.second_divide_size) | (~res["is_test"])].drop(
            columns=["_row_num", "is_test"]
        )
        test = res[(res["_row_num"] <= self.second_divide_size) & (res["is_test"])].drop(
            columns=["_row_num", "is_test"]
        )

        return train, test

    def _split_quantity(self, interactions: DataFrameLike) -> SplitterReturnType:
        """
        Split by quantity

        :param interactions: input DataFrame `[self.first_divide_column, self.item_column, self.date_column, relevance]`
        :return: train and test DataFrames
        """
        if isinstance(interactions, SparkDataFrame):
            return self._split_quantity_spark(interactions)
        else:
            return self._split_quantity_pandas(interactions)

    def _core_split(self, interactions: DataFrameLike) -> SplitterReturnType:
        if 0 <= self.second_divide_size < 1.0:
            train, test = self._split_proportion(interactions)
        elif self.second_divide_size >= 1 and isinstance(self.second_divide_size, int):
            train, test = self._split_quantity(interactions)
        else:
            raise ValueError(
                "`test_size` value must be [0, 1) or "
                "a positive integer; "
                f"test_size={self.second_divide_size}"
            )

        return train, test

    def _add_random_partition_spark(self, dataframe: SparkDataFrame) -> SparkDataFrame:
        """
        Adds `_rand` column and a user index column `_row_num` based on `_rand`.

        :param dataframe: input DataFrame with `query_id` column
        :returns: processed DataFrame
        """
        dataframe = dataframe.withColumn("_rand", sf.rand(self.seed))
        dataframe = dataframe.withColumn(
            "_row_num",
            sf.row_number().over(
                Window.partitionBy(self.first_divide_column).orderBy("_rand")
            ),
        )
        return dataframe

    def _add_random_partition_pandas(self, dataframe: PandasDataFrame) -> PandasDataFrame:
        res = dataframe.sample(frac=1, random_state=self.seed).sort_values(self.first_divide_column)
        res["_row_num"] = res.groupby(self.first_divide_column, sort=False).cumcount() + 1

        return res

    @staticmethod
    def _add_time_partition_spark(
            dataframe: SparkDataFrame,
            query_column: str = "query_id",
            date_column: str = "timestamp",
    ) -> SparkDataFrame:
        """
        Adds user index `_row_num` based on `timestamp`.

        :param dataframe: input DataFrame with `[timestamp, query_id]`
        :param query_column: user id column name
        :param date_column: timestamp column name
        :returns: processed DataFrame
        """
        res = dataframe.withColumn(
            "_row_num",
            sf.row_number().over(
                Window.partitionBy(query_column).orderBy(
                    sf.col(date_column).desc()
                )
            ),
        )
        return res

    @staticmethod
    def _add_time_partition_pandas(
            dataframe: PandasDataFrame,
            query_column: str = "query_id",
            date_column: str = "timestamp",
    ) -> PandasDataFrame:
        res = dataframe.copy(deep=True)
        res.sort_values([query_column, date_column], ascending=[True, False], inplace=True)
        res["_row_num"] = res.groupby(query_column, sort=False).cumcount() + 1
        return res
