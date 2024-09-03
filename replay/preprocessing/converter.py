from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, SparkDataFrame

if PYSPARK_AVAILABLE:
    from replay.utils.spark_utils import spark_to_pandas


class CSRConverter:
    """
    Convert input data to csr sparse matrix.
    Where ``data_column``, ``first_dim_column`` and ``second_dim_column`` satisfy the relationship
    ``matrix[first_dim_column[i], second_dim_column[i]] = data_column[i]``.

    >>> import pandas as pd
    >>> interactions = pd.DataFrame({
    ...    "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    ...    "item_id": [3, 7, 10, 5, 8, 11, 4, 9, 2, 5],
    ...    "rating": [1, 2, 3, 3, 2, 1, 3, 12, 1, 4]
    ... })
    >>> interactions
        user_id  item_id  rating
    0        1        3       1
    1        1        7       2
    2        1       10       3
    3        2        5       3
    4        2        8       2
    5        2       11       1
    6        3        4       3
    7        3        9      12
    8        3        2       1
    9        3        5       4
    >>> csr_interactions = CSRConverter(
    ...    first_dim_column="user_id",
    ...    second_dim_column="item_id",
    ...    data_column="rating",
    ... ).transform(interactions)
    >>> csr_interactions.todense()
    matrix([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  1,  0,  0,  0,  2,  0,  0,  3,  0],
            [ 0,  0,  0,  0,  0,  3,  0,  0,  2,  0,  0,  1],
            [ 0,  0,  1,  0,  3,  4,  0,  0,  0, 12,  0,  0]])
    <BLANKLINE>
    """

    def __init__(
        self,
        first_dim_column: str,
        second_dim_column: str,
        data_column: Optional[str] = None,
        row_count: Optional[int] = None,
        column_count: Optional[int] = None,
        allow_collect_to_master: bool = False,
    ):
        """
        :param first_dim_column: The name of the column for first dimension in resulting sparse matrix,
            default: ``user_id``.
        :param second_dim_column: The name of the column for second dimension in resulting sparse matrix,
            default: ``item_id``.
        :param data_column: The name of the column with values at the intersection
            of the corresponding row and column. If ``None`` then ones will be taken. Default: ``None``.
        :param row_count: Number of rows in resulting sparse matrix.
            If ``None`` then it depends on the maximum value in first_dim_column. Default: ``None``.
        :param column_count: Number of columns for resulting sparse matrix.
            If ``None`` then it depends on the maximum value in second_dim_column. Default: ``None``.
        :param allow_collect_to_master: Flag allowing spark
            to make a collection to the master node, default: ``False``.
        """
        self.first_dim_column = first_dim_column
        self.second_dim_column = second_dim_column
        self.data_column = data_column
        self.row_count = row_count
        self.column_count = column_count
        self.allow_collect_to_master = allow_collect_to_master

    def transform(self, data: DataFrameLike) -> csr_matrix:
        """
        Transform Spark or Pandas Data Frame to csr.

        :param data: Spark or Pandas Data Frame containing columns
            ``first_dim_column``, ``second_dim_column``, and optional ``data_column``.

        :returns: Sparse interactions.
        """

        if isinstance(data, SparkDataFrame):
            cols = [self.first_dim_column, self.second_dim_column]
            if self.data_column is not None:
                cols.append(self.data_column)
            data = spark_to_pandas(data.select(cols), self.allow_collect_to_master)

        rows_data = data[self.first_dim_column].values
        cols_data = data[self.second_dim_column].values
        data = data[self.data_column].values if self.data_column is not None else np.ones(data.shape[0])

        def _get_max(data: np.ndarray) -> int:
            return np.max(data) if data.shape[0] > 0 else 0

        row_count = self.row_count if self.row_count is not None else _get_max(rows_data) + 1
        col_count = self.column_count if self.column_count is not None else _get_max(cols_data) + 1
        return csr_matrix(
            (data, (rows_data, cols_data)),
            shape=(row_count, col_count),
        )
