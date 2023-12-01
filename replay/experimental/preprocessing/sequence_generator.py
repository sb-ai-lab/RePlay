from typing import List, Optional, Union

import pandas as pd

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    from pyspark.sql import Column, Window
    from pyspark.sql import functions as sf


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class SequenceGenerator:
    """
    Creating sequences for sequential models.

    E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
    then after processing, there will be generated three cases.

    ``u1, <i1> | i2``

    (Which means given user_id ``u1`` and item_seq ``<i1>``,
    model need to predict the next item ``i2``.)

    The other cases are below:

    ``u1, <i1, i2> | i3``

    ``u1, <i1, i2, i3> | i4``

    >>> import pandas as pd
    >>> time_interactions = pd.DataFrame({
    ...    "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 3],
    ...    "item_id": [3, 7, 10, 5, 8, 11, 4, 9, 2, 5],
    ...    "timestamp": [1, 2, 3, 3, 2, 1, 3, 12, 1, 4]
    ... })
    >>> time_interactions
       user_id  item_id  timestamp
    0        1        3          1
    1        1        7          2
    2        1       10          3
    3        2        5          3
    4        2        8          2
    5        2       11          1
    6        3        4          3
    7        3        9         12
    8        3        2          1
    9        3        5          4
    >>> sequences = (
    ...    SequenceGenerator(
    ...        groupby_column="user_id", transform_columns=["item_id", "timestamp"]
    ...    ).transform(time_interactions)
    ... )
    >>> sequences
       user_id item_id_list timestamp_list  label_item_id  label_timestamp
    0        1          [3]            [1]              7                2
    1        1       [3, 7]         [1, 2]             10                3
    2        2          [5]            [3]              8                2
    3        2       [5, 8]         [3, 2]             11                1
    4        3          [4]            [3]              9               12
    5        3       [4, 9]        [3, 12]              2                1
    6        3    [4, 9, 2]     [3, 12, 1]              5                4
    <BLANKLINE>
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        groupby_column: Union[str, List[str]],
        orderby_column: Union[str, List[str], None] = None,
        transform_columns: Union[None, str, List[str]] = None,
        len_window: int = 50,
        sequence_prefix: Optional[str] = None,
        sequence_suffix: Optional[str] = "_list",
        label_prefix: Optional[str] = "label_",
        label_suffix: Optional[str] = None,
        get_list_len: Optional[bool] = False,
        list_len_column: str = "list_len",
    ):
        """
        :param groupby_column: Name of column to group by.
        :param orderby_column Columns to sort by. If None
            than values are not ordered.
            default: ``None``.
        :param transform_columns: Names of interaction columns to process. If None
            than all columns are processed except grouping ones.
            default: ``None``.
        :param len_window: Max len of sequention, must be positive.
            default: ``50``.
        :param sequence_prefix: prefix added to column name after creating sequences.
            default: ``None``.
        :param sequence_suffix: suffix added to column name after creating sequences.
            default: ``_list``.
        :param label_prefix: prefix added to label column after creating sequences.
            default: ``label_``.
        :param label_suffix: suffix added to label column after creating sequences.
            default: ``None``.
        :param get_list_len: flag to calculate length of processed list or not.
            default: ``False``.
        :param list_len_column: List length column name. Used if get_list_len.
            default: ``list_len``.
        """
        self.groupby_column = groupby_column if not isinstance(groupby_column, str) else [groupby_column]
        self.orderby_column: Union[List, Column, None]
        if orderby_column is None:
            self.orderby_column = None
        else:
            self.orderby_column = orderby_column if not isinstance(orderby_column, str) else [orderby_column]

        self.transform_columns = transform_columns
        self.len_window = len_window

        self.sequence_prefix = "" if sequence_prefix is None else sequence_prefix
        self.sequence_suffix = "" if sequence_suffix is None else sequence_suffix

        self.label_prefix = "" if label_prefix is None else label_prefix
        self.label_suffix = "" if label_suffix is None else label_suffix

        self.get_list_len = get_list_len
        self.list_len_column = list_len_column

    def transform(self, interactions: DataFrameLike) -> DataFrameLike:
        """Create sequences from given interactions.

        :param interactions: DataFrame.

        :returns: DataFrame with transformed interactions. Sequential interactions in list.

        """
        if self.transform_columns is None:
            self.transform_columns = list(set(interactions.columns).difference(self.groupby_column))
        else:
            self.transform_columns = (
                self.transform_columns if not isinstance(self.transform_columns, str) else [self.transform_columns]
            )

        if isinstance(interactions, SparkDataFrame):
            return self._transform_spark(interactions)

        return self._transform_pandas(interactions)

    def _transform_pandas(self, interactions: PandasDataFrame) -> PandasDataFrame:
        assert self.transform_columns is not None
        processed_interactions = interactions.copy(deep=True)

        def seq_rolling(col: pd.Series) -> List:
            return [window.to_list()[:-1] for window in col.rolling(self.len_window + 1)]

        for transform_col in self.transform_columns:
            if self.orderby_column is not None:
                processed_interactions.sort_values(by=self.orderby_column, inplace=True)
            else:
                processed_interactions.sort_values(by=self.groupby_column, inplace=True)

            processed_interactions[self.sequence_prefix + transform_col + self.sequence_suffix] = [
                item
                for sublist in processed_interactions.groupby(self.groupby_column, sort=False)[transform_col].apply(
                    seq_rolling
                )
                for item in sublist
            ]
            processed_interactions[self.label_prefix + transform_col + self.label_suffix] = processed_interactions[
                transform_col
            ]

        first_tranformed_col = self.sequence_prefix + self.transform_columns[0] + self.sequence_suffix
        processed_interactions = processed_interactions[processed_interactions[first_tranformed_col].str.len() > 0]

        transformed_columns = list(
            map(lambda x: self.sequence_prefix + x + self.sequence_suffix, self.transform_columns)
        )
        label_columns = list(map(lambda x: self.label_prefix + x + self.label_suffix, self.transform_columns))
        select_columns = self.groupby_column + transformed_columns + label_columns

        if self.get_list_len:
            processed_interactions[self.list_len_column] = processed_interactions[first_tranformed_col].str.len()
            select_columns += [self.list_len_column]

        processed_interactions.reset_index(inplace=True)

        return processed_interactions[select_columns]

    def _transform_spark(self, interactions: SparkDataFrame) -> SparkDataFrame:
        assert self.transform_columns is not None
        processed_interactions = interactions
        orderby_column: Union[Column, List]
        if self.orderby_column is None:
            orderby_column = sf.lit(1)
        else:
            orderby_column = self.orderby_column

        window = (
            Window.partitionBy(self.groupby_column)  # type: ignore
            .orderBy(orderby_column)
            .rowsBetween(-self.len_window, -1)
        )
        for transform_col in self.transform_columns:
            processed_interactions = processed_interactions.withColumn(
                self.sequence_prefix + transform_col + self.sequence_suffix,
                sf.collect_list(transform_col).over(window),
            ).withColumn(self.label_prefix + transform_col + self.label_suffix, sf.col(transform_col))

        first_tranformed_col = self.sequence_prefix + self.transform_columns[0] + self.sequence_suffix
        processed_interactions = processed_interactions.filter(sf.size(first_tranformed_col) > 0)

        transformed_columns = list(
            map(lambda x: self.sequence_prefix + x + self.sequence_suffix, self.transform_columns)
        )
        label_columns = list(map(lambda x: self.label_prefix + x + self.label_suffix, self.transform_columns))
        select_columns = self.groupby_column + transformed_columns + label_columns

        if self.get_list_len:
            processed_interactions = processed_interactions.withColumn(
                self.list_len_column, sf.size(first_tranformed_col)
            )
            select_columns += [self.list_len_column]

        return processed_interactions.select(select_columns)
