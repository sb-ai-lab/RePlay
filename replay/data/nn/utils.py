from typing import Optional

import polars as pl

from replay.utils.spark_utils import spark_to_pandas
from replay.utils import PYSPARK_AVAILABLE, DataFrameLike, PandasDataFrame, PolarsDataFrame

if PYSPARK_AVAILABLE:  # pragma: no cover
    import pyspark.sql.functions as F


def groupby_sequences(events: DataFrameLike, groupby_col: str, sort_col: Optional[str] = None) -> DataFrameLike:
    """
    :param events: dataframe with interactions
    :param groupby_col: divide column to group by
    :param sort_col: column to sort by

    :returns: dataframe with sequences for each value in groupby_col
    """
    if isinstance(events, PandasDataFrame):
        event_cols_without_groupby = events.columns.values.tolist()
        event_cols_without_groupby.remove(groupby_col)

        if sort_col:
            event_cols_without_groupby.remove(sort_col)
            event_cols_without_groupby.insert(0, sort_col)
            events = events.sort_values(event_cols_without_groupby)

        grouped_sequences = (
            events.groupby(groupby_col).agg({col: list for col in event_cols_without_groupby}).reset_index()
        )
    elif isinstance(events, PolarsDataFrame):
        event_cols_without_groupby = events.columns
        event_cols_without_groupby.remove(groupby_col)

        if sort_col:
            event_cols_without_groupby.remove(sort_col)
            event_cols_without_groupby.insert(0, sort_col)
            events = events.sort(event_cols_without_groupby)

        grouped_sequences = events.group_by(groupby_col).agg(
            *[pl.col(x) for x in event_cols_without_groupby]
        )
    else:
        event_cols_without_groupby = events.columns.copy()
        event_cols_without_groupby.remove(groupby_col)

        if sort_col:
            event_cols_without_groupby.remove(sort_col)
            event_cols_without_groupby.insert(0, sort_col)

        all_cols_struct = F.struct(event_cols_without_groupby)  # type: ignore

        collect_fn = F.collect_list(all_cols_struct)
        if sort_col:
            collect_fn = F.sort_array(collect_fn)

        grouped_sequences = (
            events.groupby(groupby_col)
            .agg(collect_fn.alias("_"))
            .select([F.col(groupby_col)] + [F.col(f"_.{col}").alias(col) for col in event_cols_without_groupby])
            .drop("_")
        )

    return grouped_sequences


def ensure_pandas(
    df: DataFrameLike,
    allow_collect_to_master: bool = False,
) -> PandasDataFrame:
    """
    :param df: dataframe
    :param allow_collect_to_master: Flag allowing spark to make a collection to the master node,
        default: ``False``.

    :returns: Pandas DataFrame object
    """
    if isinstance(df, PandasDataFrame):
        return df
    return spark_to_pandas(df, allow_collect_to_master)
