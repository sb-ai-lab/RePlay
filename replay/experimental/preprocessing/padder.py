from collections.abc import Iterable
from typing import List, Optional, Union

from pandas.api.types import is_object_dtype

from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, SparkDataFrame

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


class Padder:
    """
    Pad array columns in dataframe.

    >>> import pandas as pd
    >>> pad_interactions = pd.DataFrame({
    ...    "user_id": [1, 1, 1, 1, 2, 2, 3, 3, 3],
    ...    "timestamp": [[1], [1, 2], [1, 2, 4], [1, 2, 4, 6], [4, 7, 12],
    ...                  [4, 7, 12, 126], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6],
    ...                  [1, 2, 3, 4, 5, 6, 7]],
    ...    "item_id": [['a'], ['a', 'b'], ['a', 'b', 'd'], ['a', 'b', 'd', 'f'], ['d', 'e', 'm'],
    ...                ['d', 'e', 'm', 'g'], ['a', 'b', 'c', 'd', 'a'], ['a', 'b', 'c', 'd', 'a', 'f'],
    ...                ['a', 'b', 'c', 'd', 'a', 'f', 'e']]
    ... })
    >>> pad_interactions
        user_id              timestamp                item_id
    0        1                    [1]                    [a]
    1        1                 [1, 2]                 [a, b]
    2        1              [1, 2, 4]              [a, b, d]
    3        1           [1, 2, 4, 6]           [a, b, d, f]
    4        2             [4, 7, 12]              [d, e, m]
    5        2        [4, 7, 12, 126]           [d, e, m, g]
    6        3        [1, 2, 3, 4, 5]        [a, b, c, d, a]
    7        3     [1, 2, 3, 4, 5, 6]     [a, b, c, d, a, f]
    8        3  [1, 2, 3, 4, 5, 6, 7]  [a, b, c, d, a, f, e]
    >>> Padder(
    ...    pad_columns=["item_id", "timestamp"],
    ...    padding_side="right",
    ...    padding_value=["[PAD]", 0],
    ...    array_size=5,
    ...    cut_array=True,
    ...    cut_side="right"
    ... ).transform(pad_interactions)
       user_id           timestamp                          item_id
    0        1     [1, 0, 0, 0, 0]  [a, [PAD], [PAD], [PAD], [PAD]]
    1        1     [1, 2, 0, 0, 0]      [a, b, [PAD], [PAD], [PAD]]
    2        1     [1, 2, 4, 0, 0]          [a, b, d, [PAD], [PAD]]
    3        1     [1, 2, 4, 6, 0]              [a, b, d, f, [PAD]]
    4        2    [4, 7, 12, 0, 0]          [d, e, m, [PAD], [PAD]]
    5        2  [4, 7, 12, 126, 0]              [d, e, m, g, [PAD]]
    6        3     [1, 2, 3, 4, 5]                  [a, b, c, d, a]
    7        3     [2, 3, 4, 5, 6]                  [b, c, d, a, f]
    8        3     [3, 4, 5, 6, 7]                  [c, d, a, f, e]
    <BLANKLINE>
    """

    def __init__(
        self,
        pad_columns: Union[str, List[str]],
        padding_side: Optional[str] = "right",
        padding_value: Union[str, float, List, None] = 0,
        array_size: Optional[int] = None,
        cut_array: Optional[bool] = True,
        cut_side: Optional[str] = "right",
    ):
        """
        :param pad_columns: Name of columns to pad.
        :param padding_side: side of array to which add padding values. Can be "right" or "left",
            default: ``right``.
        :param padding_value: value to fill missing spacec,
            default: ``0``.
        :param array_size: needed array size,
            default: ``None``
        :param cut_array: is cutting arrays with shape more than array_size needed,
            default: ``True``.
        :param cut_side: side of array on which to cut to needed length. Can be "right" or "left",
                default: ``right``.
        """
        self.pad_columns = (
            pad_columns if (isinstance(pad_columns, Iterable) and not isinstance(pad_columns, str)) else [pad_columns]
        )
        if padding_side not in (
            "right",
            "left",
        ):
            msg = f"padding_side value {padding_side} is not implemented. Should be 'right' or 'left'"
            raise ValueError(msg)

        self.padding_side = padding_side
        self.padding_value = (
            padding_value
            if (isinstance(padding_value, Iterable) and not isinstance(padding_value, str))
            else [padding_value]
        )
        if len(self.padding_value) == 1 and len(self.pad_columns) > 1:
            self.padding_value = self.padding_value * len(self.pad_columns)
        if len(self.pad_columns) != len(self.padding_value):
            msg = "pad_columns and padding_value should have same length"
            raise ValueError(msg)

        self.array_size = array_size
        if self.array_size is not None:
            if self.array_size < 1 or not isinstance(self.array_size, int):
                msg = "array_size should be positive integer greater than 0"
                raise ValueError(msg)
        else:
            self.array_size = -1

        self.cut_array = cut_array
        self.cut_side = cut_side

    def transform(self, interactions: DataFrameLike) -> DataFrameLike:
        """Pad dataframe.

        :param interactions: DataFrame with array columns with names pad_columns.

        :returns: DataFrame with padded array columns.

        """
        df_transformed = interactions
        is_spark = isinstance(interactions, SparkDataFrame)
        column_dtypes = dict(df_transformed.dtypes)

        for col, pad_value in zip(self.pad_columns, self.padding_value):
            if col not in df_transformed.columns:
                msg = f"Column {col} not in DataFrame columns."
                raise ValueError(msg)
            if is_spark is True and not column_dtypes[col].startswith("array"):
                msg = f"Column {col} should have ArrayType to be padded."
                raise ValueError(msg)
            if is_spark is False and not is_object_dtype(df_transformed[col]):
                msg = f"Column {col} should have object dtype to be padded."
                raise ValueError()

            if is_spark is True:
                df_transformed = self._transform_spark(df_transformed, col, pad_value)
            else:
                df_transformed = self._transform_pandas(df_transformed, col, pad_value)

        return df_transformed

    def _transform_pandas(
        self, df_transformed: PandasDataFrame, col: str, pad_value: Union[str, float, List, None]
    ) -> PandasDataFrame:
        max_array_size = df_transformed[col].str.len().max() if self.array_size == -1 else self.array_size

        def right_cut(sample: List) -> List:
            # fmt: off
            return sample[-min(len(sample), max_array_size):]
            # fmt: on

        def left_cut(sample: List) -> List:
            # fmt: off
            return sample[:min(len(sample), max_array_size)]
            # fmt: on

        res = df_transformed.copy(deep=True)
        res[col] = res[col].apply(lambda sample: sample if isinstance(sample, list) else [])
        cut_col_name = f"{col}_cut"
        if self.cut_array:
            cut_func = right_cut if self.cut_side == "right" else left_cut

            res[cut_col_name] = res[col].apply(cut_func)
        else:
            res[cut_col_name] = res[col]

        paddings = res[cut_col_name].apply(lambda x: [pad_value for _ in range(max_array_size - len(x))])
        if self.padding_side == "right":
            res[col] = res[cut_col_name] + paddings
        else:
            res[col] = paddings + res[cut_col_name]

        res.drop(columns=[cut_col_name], inplace=True)

        return res

    def _transform_spark(
        self, df_transformed: SparkDataFrame, col: str, pad_value: Union[str, float, List, None]
    ) -> SparkDataFrame:
        if self.array_size == -1:
            max_array_size = df_transformed.agg(sf.max(sf.size(col)).alias("max_array_len")).collect()[0][0]
        else:
            max_array_size = self.array_size

        df_transformed = df_transformed.withColumn(col, sf.coalesce(col, sf.array()))
        insert_value = pad_value if not isinstance(pad_value, str) else "'" + pad_value + "'"

        cut_col_name = f"{col}_cut"
        drop_cols = [cut_col_name, "pre_zeros", "zeros"]

        if self.cut_array:
            slice_col_name = f"{col}_slice_point"
            drop_cols += [slice_col_name]

            if self.cut_side == "right":
                slice_func = (-1) * sf.least(sf.size(col), sf.lit(max_array_size))
                cut_func = sf.when(
                    sf.size(col) > 0, sf.expr(f"slice({col}, {slice_col_name}, {max_array_size})")
                ).otherwise(sf.array())
            else:
                slice_func = sf.least(sf.size(col), sf.lit(max_array_size))
                cut_func = sf.when(sf.size(col) > 0, sf.expr(f"slice({col}, 1, {slice_col_name})")).otherwise(
                    sf.array()
                )

            df_transformed = df_transformed.withColumn(slice_col_name, slice_func).withColumn(cut_col_name, cut_func)

        else:
            df_transformed = df_transformed.withColumn(cut_col_name, sf.col(col))

        if self.padding_side == "right":
            concat_func = sf.concat(sf.col(cut_col_name), sf.col("zeros"))
        else:
            concat_func = sf.concat(sf.col("zeros"), sf.col(cut_col_name))

        df_transformed = (
            df_transformed.withColumn(
                "pre_zeros",
                sf.sequence(sf.lit(0), sf.greatest(sf.lit(max_array_size) - sf.size(sf.col(cut_col_name)), sf.lit(0))),
            )
            .withColumn(
                "zeros", sf.expr(f"transform(slice(pre_zeros, 1, size(pre_zeros) - 1), element -> {insert_value})")
            )
            .withColumn(col, concat_func)
            .drop(*drop_cols)
        )

        return df_transformed
