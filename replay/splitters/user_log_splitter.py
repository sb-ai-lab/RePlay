"""
This splitter split data for each user separately
"""
from typing import Optional, Union

import pyspark.sql.functions as sf
from pyspark.sql import DataFrame, Window

from replay.constants import AnyDataFrame
from replay.splitters.base_splitter import Splitter, SplitterReturnType
from replay.utils import convert2spark


# pylint: disable=too-few-public-methods
class UserSplitter(Splitter):
    """
    Split data inside each user's history separately.

    Example:

    >>> from replay.session_handler import get_spark_session, State
    >>> spark = get_spark_session(1, 1)
    >>> state = State(spark)

    >>> from replay.splitters import UserSplitter
    >>> import pandas as pd
    >>> data_frame = pd.DataFrame({"user_id": [1,1,1,2,2,2],
    ...    "item_id": [1,2,3,1,2,3],
    ...    "relevance": [1,2,3,4,5,6],
    ...    "timestamp": [1,2,3,3,2,1]})
    >>> data_frame
       user_id  item_id  relevance  timestamp
    0        1        1          1          1
    1        1        2          2          2
    2        1        3          3          3
    3        2        1          4          3
    4        2        2          5          2
    5        2        3          6          1

    By default, test is one last item for each user

    >>> UserSplitter(seed=80083).split(data_frame)[-1].toPandas()
       user_id  item_id  relevance  timestamp
    0        1        3          3          3
    1        2        1          4          3

    Random records can be retrieved with ``shuffle``:

    >>> UserSplitter(shuffle=True, seed=80083).split(data_frame)[-1].toPandas()
       user_id  item_id  relevance  timestamp
    0        1        2          2          2
    1        2        3          6          1

    You can specify the number of items for each user:

    >>> UserSplitter(item_test_size=3, shuffle=True, seed=80083).split(data_frame)[-1].toPandas()
       user_id  item_id  relevance  timestamp
    0        1        2          2          2
    1        1        3          3          3
    2        1        1          1          1
    3        2        3          6          1
    4        2        2          5          2
    5        2        1          4          3

    Or a fraction:

    >>> UserSplitter(item_test_size=0.67, shuffle=True, seed=80083).split(data_frame)[-1].toPandas()
       user_id  item_id  relevance  timestamp
    0        1        2          2          2
    1        1        3          3          3
    2        2        3          6          1
    3        2        2          5          2

    `user_test_size` allows to put exact number of users into test set

    >>> UserSplitter(user_test_size=1, item_test_size=2, seed=42).split(data_frame)[-1].toPandas().user_id.nunique()
    1

    >>> UserSplitter(user_test_size=0.5, item_test_size=2, seed=42).split(data_frame)[-1].toPandas().user_id.nunique()
    1

    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        item_test_size: Union[float, int] = 1,
        user_test_size: Optional[Union[float, int]] = None,
        shuffle=False,
        drop_cold_items: bool = False,
        drop_cold_users: bool = False,
        seed: Optional[int] = None,
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
        """
        super().__init__(
            drop_cold_items=drop_cold_items, drop_cold_users=drop_cold_users
        )
        self.item_test_size = item_test_size
        self.user_test_size = user_test_size
        self.shuffle = shuffle
        self.seed = seed

    def _get_test_users(
        self,
        log: DataFrame,
    ) -> DataFrame:
        """
        :param log: input DataFrame
        :return: Spark DataFrame with single column `user_id`
        """
        all_users = log.select("user_id").distinct()
        user_count = all_users.count()
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
            test_users = (
                all_users.withColumn("rand", sf.rand(self.seed))
                .withColumn(
                    "row_num", sf.row_number().over(Window.orderBy("rand"))
                )
                .filter(f"row_num <= {test_user_count}")
                .drop("rand", "row_num")
            )
        else:
            test_users = all_users
        return test_users

    def _split_proportion(self, log: DataFrame) -> SplitterReturnType:
        """
        Proportionate split

        :param log: input DataFrame `[timestamp, user_id, item_id, relevance]`
        :return: train and test DataFrames
        """

        counts = log.groupBy("user_id").count()
        test_users = self._get_test_users(log).withColumn(
            "test_user", sf.lit(1)
        )
        if self.shuffle:
            res = self._add_random_partition(
                log.join(test_users, how="left", on="user_id")
            )
        else:
            res = self._add_time_partition(
                log.join(test_users, how="left", on="user_id")
            )

        res = res.join(counts, on="user_id", how="left")
        res = res.withColumn("frac", sf.col("row_num") / sf.col("count"))
        train = res.filter(
            f"""
                    frac > {self.item_test_size} OR
                    test_user IS NULL
                """
        ).drop("rand", "row_num", "count", "frac", "test_user")
        test = res.filter(
            f"""
                    frac <= {self.item_test_size} AND
                    test_user IS NOT NULL
                """
        ).drop("rand", "row_num", "count", "frac", "test_user")
        return train, test

    def _split_quantity(self, log: DataFrame) -> SplitterReturnType:
        """
        Split by quantity

        :param log: input DataFrame `[timestamp, user_id, item_id, relevance]`
        :return: train and test DataFrames
        """

        test_users = self._get_test_users(log).withColumn(
            "test_user", sf.lit(1)
        )
        if self.shuffle:
            res = self._add_random_partition(
                log.join(test_users, how="left", on="user_id")
            )
        else:
            res = self._add_time_partition(
                log.join(test_users, how="left", on="user_id")
            )
        train = res.filter(
            f"""
                    row_num > {self.item_test_size} OR
                    test_user IS NULL
                """
        ).drop("rand", "row_num", "test_user")
        test = res.filter(
            f"""
                    row_num <= {self.item_test_size} AND
                    test_user IS NOT NULL
                """
        ).drop("rand", "row_num", "test_user")
        return train, test

    def _core_split(self, log: DataFrame) -> SplitterReturnType:
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

        return train, test

    def _add_random_partition(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds `rand` column and a user index column `row_num` based on `rand`.

        :param dataframe: input DataFrame with `user_id` column
        :returns: processed DataFrame
        """
        dataframe = dataframe.withColumn("rand", sf.rand(self.seed))
        dataframe = dataframe.withColumn(
            "row_num",
            sf.row_number().over(
                Window.partitionBy("user_id").orderBy("rand")
            ),
        )
        return dataframe

    @staticmethod
    def _add_time_partition(dataframe: DataFrame) -> DataFrame:
        """
        Adds user index `row_num` based on `timestamp`.

        :param dataframe: input DataFrame with `[timestamp, user_id]`
        :returns: processed DataFrame
        """
        res = dataframe.withColumn(
            "row_num",
            sf.row_number().over(
                Window.partitionBy("user_id").orderBy(
                    sf.col("timestamp").desc()
                )
            ),
        )
        return res


def k_folds(
    log: AnyDataFrame,
    n_folds: Optional[int] = 5,
    seed: Optional[int] = None,
    splitter: Optional[str] = "user",
) -> SplitterReturnType:
    """
    Splits log inside each user into folds at random

    :param log: input DataFrame
    :param n_folds: number of folds
    :param seed: random seed
    :param splitter: splitting strategy. Only user variant is available atm.
    :return: yields train and test DataFrames by folds
    """
    if splitter not in {"user"}:
        raise ValueError(
            f"Недопустимое значение параметра splitter: {splitter}"
        )
    if splitter == "user":
        dataframe = convert2spark(log).withColumn("rand", sf.rand(seed))
        dataframe = dataframe.withColumn(
            "fold",
            sf.row_number().over(Window.partitionBy("user_id").orderBy("rand"))
            % n_folds,
        ).drop("rand")
        for i in range(n_folds):
            train = dataframe.filter(f"fold != {i}").drop("fold")
            test = dataframe.filter(f"fold == {i}").drop("fold")
            yield train, test
