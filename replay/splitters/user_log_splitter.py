"""
This splitter split data for each user separately
"""
from typing import Optional, Union

import pyspark.sql.functions as sf
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame, Window

from replay.data import AnyDataFrame
from replay.splitters.base_splitter import Splitter, SplitterReturnType


# pylint: disable=too-few-public-methods
class UserSplitter(Splitter):
    """
    Split data inside each user's history separately.

    Example:

    >>> from replay.utils.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> from replay.splitters import UserSplitter
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_idx": [1,1,1,2,2,2],
    ...    "item_idx": [1,2,3,1,2,3],
    ...    "relevance": [1,2,3,4,5,6],
    ...    "timestamp": [1,2,3,3,2,1]})
    >>> data_frame
       user_idx  item_idx  relevance  timestamp
    0         1         1          1          1
    1         1         2          2          2
    2         1         3          3          3
    3         2         1          4          3
    4         2         2          5          2
    5         2         3          6          1

    >>> from replay.utils.spark_utils import convert2spark
    >>> data_frame = convert2spark(data_frame)

    By default, test is one last item for each user

    >>> UserSplitter(seed=80083).split(data_frame)[-1].toPandas()
       user_idx  item_idx  relevance  timestamp
    0         1         3          3          3
    1         2         1          4          3

    Random records can be retrieved with ``shuffle``:

    >>> UserSplitter(shuffle=True, seed=80083).split(data_frame)[-1].toPandas()
       user_idx  item_idx  relevance  timestamp
    0         1         2          2          2
    1         2         3          6          1

    You can specify the number of items for each user:

    >>> UserSplitter(item_test_size=3, shuffle=True, seed=80083).split(data_frame)[-1].toPandas()
       user_idx  item_idx  relevance  timestamp
    0         1         2          2          2
    1         1         3          3          3
    2         1         1          1          1
    3         2         3          6          1
    4         2         2          5          2
    5         2         1          4          3

    Or a fraction:

    >>> UserSplitter(item_test_size=0.67, shuffle=True, seed=80083).split(data_frame)[-1].toPandas()
       user_idx  item_idx  relevance  timestamp
    0         1         2          2          2
    1         1         3          3          3
    2         2         3          6          1
    3         2         2          5          2

    `user_test_size` allows to put exact number of users into test set

    >>> UserSplitter(user_test_size=1, item_test_size=2, seed=42).split(data_frame)[-1].toPandas().user_idx.nunique()
    1

    >>> UserSplitter(user_test_size=0.5, item_test_size=2, seed=42).split(data_frame)[-1].toPandas().user_idx.nunique()
    1

    """

    _init_arg_names = [
        "item_test_size",
        "user_test_size",
        "shuffle",
        "drop_cold_users",
        "drop_cold_items",
        "seed",
        "user_col",
        "item_col",
        "timestamp_col",
        "session_id_col",
        "session_id_processing_strategy",
    ]

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        item_test_size: Union[float, int] = 1,
        user_test_size: Optional[Union[float, int]] = None,
        shuffle=False,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
        user_col: str = "user_idx",
        item_col: Optional[str] = "item_idx",
        timestamp_col: Optional[str] = "timestamp",
        rating_col: Optional[str] = "relevance",
        session_id_col: Optional[str] = None,
        session_id_processing_strategy: str = "test",
    ):
        """
        :param item_test_size: fraction or a number of items per user
        :param user_test_size: similar to ``item_test_size``,
            but corresponds to the number of users.
            ``None`` is all available users.
        :param shuffle: take random items and not last based on ``timestamp``.
        :param drop_cold_items: flag to drop cold items from test
        :param drop_cold_users: flag to drop cold users from test
        :param seed: random seed
        :param user_col: user id column name
        :param item_col: item id column name
        :param timestamp_col: timestamp column name
        :param rating_col: rating column name
        :param session_id_col: name of session id column, which values can not be split.
        :param session_id_processing_strategy: strategy of processing session if it is split,
            values: ``train, test``, train: whole split session goes to train. test: same but to test.
            default: ``test``.
        """
        super().__init__(
            drop_cold_items=drop_cold_items,
            drop_cold_users=drop_cold_users,
            user_col=user_col,
            item_col=item_col,
            timestamp_col=timestamp_col,
            rating_col=rating_col,
            session_id_col=session_id_col,
            session_id_processing_strategy=session_id_processing_strategy
        )
        self.item_test_size = item_test_size
        self.user_test_size = user_test_size
        self.shuffle = shuffle
        self.seed = seed

    def _get_order_of_sort(self) -> list:   # pragma: no cover
        pass

    def _get_test_users(
        self,
        log: AnyDataFrame,
    ) -> AnyDataFrame:
        """
        :param log: input DataFrame
        :return: Spark DataFrame with single column `user_id`
        """
        if isinstance(log, SparkDataFrame):
            all_users = log.select(self.user_col).distinct()
            user_count = all_users.count()
        else:
            all_users = PandasDataFrame(log[self.user_col].unique(), columns=[self.user_col])
            user_count = len(all_users)

        if self.user_test_size is not None:
            value_error = False
            if isinstance(self.user_test_size, int):
                if 1 <= self.user_test_size < user_count:
                    test_user_count = self.user_test_size
                else:
                    value_error = True
            else:
                if 1 > self.user_test_size > 0:
                    test_user_count = user_count * self.user_test_size
                else:
                    value_error = True
            if value_error:
                raise ValueError(
                    f"""
                Invalid value for user_test_size: {self.user_test_size}
                """
                )
            if isinstance(log, SparkDataFrame):
                test_users = (
                    all_users.withColumn("_rand", sf.rand(self.seed))
                    .withColumn(
                        "_row_num", sf.row_number().over(Window.orderBy("_rand"))
                    )
                    .filter(f"_row_num <= {test_user_count}")
                    .drop("_rand", "_row_num")
                )
            else:
                test_users = all_users.sample(n=int(test_user_count), random_state=self.seed)
        else:
            test_users = all_users

        return test_users

    def _split_proportion_spark(self, log: SparkDataFrame) -> Union[SparkDataFrame, SparkDataFrame]:
        counts = log.groupBy(self.user_col).count()
        test_users = self._get_test_users(log).withColumn(
            "is_test", sf.lit(True)
        )
        if self.shuffle:
            res = self._add_random_partition_spark(
                log.join(test_users, how="left", on=self.user_col)
            )
        else:
            res = self._add_time_partition_spark(
                log.join(test_users, how="left", on=self.user_col)
            )

        res = res.join(counts, on=self.user_col, how="left")
        res = res.withColumn("_frac", sf.col("_row_num") / sf.col("count"))
        res = res.na.fill({"is_test": False})
        if self.session_id_col:
            res = self._recalculate_with_session_id_column(res)

        train = res.filter(
            f"""
                    _frac > {self.item_test_size} OR
                    NOT is_test
                """
        ).drop("_rand", "_row_num", "count", "_frac", "is_test")
        test = res.filter(
            f"""
                    _frac <= {self.item_test_size} AND
                    is_test
                """
        ).drop("_rand", "_row_num", "count", "_frac", "is_test")

        return train, test

    def _split_proportion_pandas(self, log: PandasDataFrame) -> Union[PandasDataFrame, PandasDataFrame]:
        counts = log.groupby(self.user_col).agg(count=(self.user_col, "count")).reset_index()
        test_users = self._get_test_users(log)
        test_users["is_test"] = True
        if self.shuffle:
            res = self._add_random_partition_pandas(
                log.merge(test_users, how="left", on=self.user_col)
            )
        else:
            res = self._add_time_partition_pandas(
                log.merge(test_users, how="left", on=self.user_col)
            )
        res["is_test"].fillna(False, inplace=True)
        res = res.merge(counts, on=self.user_col, how="left")
        if self.session_id_col:
            res = self._recalculate_with_session_id_column(res)
        res["_frac"] = res["_row_num"] / res["count"]
        train = res[(res["_frac"] > self.item_test_size) | (~res["is_test"])].drop(
            columns=["_row_num", "count", "_frac", "is_test"]
        )
        test = res[(res["_frac"] <= self.item_test_size) & (res["is_test"])].drop(
            columns=["_row_num", "count", "_frac", "is_test"]
        )

        return train, test

    def _split_proportion(self, log: AnyDataFrame) -> SplitterReturnType:
        """
        Proportionate split

        :param log: input DataFrame `[self.user_col, self.item_col, self.date_col, relevance]`
        :return: train and test DataFrames
        """
        if isinstance(log, SparkDataFrame):
            return self._split_proportion_spark(log)
        else:
            return self._split_proportion_pandas(log)

    def _split_quantity_spark(self, log: SparkDataFrame) -> SparkDataFrame:
        test_users = self._get_test_users(log).withColumn(
            "is_test", sf.lit(True)
        )
        if self.shuffle:
            res = self._add_random_partition_spark(
                log.join(test_users, how="left", on=self.user_col)
            )
        else:
            res = self._add_time_partition_spark(
                log.join(test_users, how="left", on=self.user_col)
            )
        res = res.na.fill({"is_test": False})
        if self.session_id_col:
            res = self._recalculate_with_session_id_column(res)
        train = res.filter(
            f"""
                    _row_num > {self.item_test_size} OR
                    NOT is_test
                """
        ).drop("_rand", "_row_num", "is_test")
        test = res.filter(
            f"""
                    _row_num <= {self.item_test_size} AND
                    is_test
                """
        ).drop("_rand", "_row_num", "is_test")

        return train, test

    def _split_quantity_pandas(self, log: PandasDataFrame) -> PandasDataFrame:
        test_users = self._get_test_users(log)
        test_users["is_test"] = True
        if self.shuffle:
            res = self._add_random_partition_pandas(
                log.merge(test_users, how="left", on=self.user_col)
            )
        else:
            res = self._add_time_partition_pandas(
                log.merge(test_users, how="left", on=self.user_col)
            )
        res["is_test"].fillna(False, inplace=True)
        if self.session_id_col:
            res = self._recalculate_with_session_id_column(res)
        train = res[(res["_row_num"] > self.item_test_size) | (~res["is_test"])].drop(
            columns=["_row_num", "is_test"]
        )
        test = res[(res["_row_num"] <= self.item_test_size) & (res["is_test"])].drop(
            columns=["_row_num", "is_test"]
        )

        return train, test

    def _split_quantity(self, log: AnyDataFrame) -> SplitterReturnType:
        """
        Split by quantity

        :param log: input DataFrame `[self.user_col, self.item_col, self.date_col, relevance]`
        :return: train and test DataFrames
        """
        if isinstance(log, SparkDataFrame):
            return self._split_quantity_spark(log)
        else:
            return self._split_quantity_pandas(log)

    def _core_split(self, log: AnyDataFrame) -> SplitterReturnType:
        if 0 <= self.item_test_size < 1.0:
            train, test = self._split_proportion(log)
        elif self.item_test_size >= 1 and isinstance(self.item_test_size, int):
            train, test = self._split_quantity(log)
        else:
            raise ValueError(
                "`test_size` value must be [0, 1) or "
                "a positive integer; "
                f"test_size={self.item_test_size}"
            )

        return [train, test]

    def _add_random_partition_spark(self, dataframe: SparkDataFrame) -> SparkDataFrame:
        """
        Adds `_rand` column and a user index column `_row_num` based on `_rand`.

        :param dataframe: input DataFrame with `user_id` column
        :returns: processed DataFrame
        """
        dataframe = dataframe.withColumn("_rand", sf.rand(self.seed))
        dataframe = dataframe.withColumn(
            "_row_num",
            sf.row_number().over(
                Window.partitionBy(self.user_col).orderBy("_rand")
            ),
        )
        return dataframe

    def _add_random_partition_pandas(self, dataframe: PandasDataFrame) -> PandasDataFrame:
        res = dataframe.sample(frac=1, random_state=self.seed).sort_values(self.user_col)
        res["_row_num"] = res.groupby(self.user_col, sort=False).cumcount() + 1

        return res

    @staticmethod
    def _add_time_partition_spark(
            dataframe: SparkDataFrame,
            user_col: str = "user_idx",
            date_col: str = "timestamp",
    ) -> SparkDataFrame:
        """
        Adds user index `_row_num` based on `timestamp`.

        :param dataframe: input DataFrame with `[timestamp, user_id]`
        :param user_col: user id column name
        :param date_col: timestamp column name
        :returns: processed DataFrame
        """
        res = dataframe.withColumn(
            "_row_num",
            sf.row_number().over(
                Window.partitionBy(user_col).orderBy(
                    sf.col(date_col).desc()
                )
            ),
        )
        return res

    @staticmethod
    def _add_time_partition_pandas(
            dataframe: PandasDataFrame,
            user_col: str = "user_idx",
            date_col: str = "timestamp",
    ) -> PandasDataFrame:
        res = dataframe.copy(deep=True)
        res.sort_values([user_col, date_col], ascending=[True, False], inplace=True)
        res["_row_num"] = res.groupby(user_col, sort=False).cumcount() + 1
        return res


def k_folds(
    log: AnyDataFrame,
    n_folds: Optional[int] = 5,
    seed: Optional[int] = None,
    splitter: Optional[str] = "user",
    user_col: str = "user_idx",
) -> SplitterReturnType:
    """
    Splits log inside each user into folds at random

    :param log: input DataFrame
    :param n_folds: number of folds
    :param seed: random seed
    :param splitter: splitting strategy. Only user variant is available atm.
    :param user_col: user id column name
    :return: yields train and test DataFrames by folds
    """

    if splitter not in {"user"}:
        raise ValueError(f"Wrong splitter parameter: {splitter}")
    if splitter == "user":
        if isinstance(log, SparkDataFrame):
            dataframe = log.withColumn("_rand", sf.rand(seed))
            dataframe = dataframe.withColumn(
                "fold",
                sf.row_number().over(
                    Window.partitionBy(user_col).orderBy("_rand")
                )
                % n_folds,
            ).drop("_rand")
            for i in range(n_folds):
                train = dataframe.filter(f"fold != {i}").drop("fold")
                test = dataframe.filter(f"fold == {i}").drop("fold")
                yield train, test
        else:
            dataframe = log.sample(frac=1, random_state=seed).sort_values(user_col)
            dataframe["fold"] = (dataframe.groupby(user_col, sort=False).cumcount() + 1) % n_folds
            for i in range(n_folds):
                train = dataframe[dataframe["fold"] != i].drop(columns=["fold"])
                test = dataframe[dataframe["fold"] == i].drop(columns=["fold"])
                yield train, test
