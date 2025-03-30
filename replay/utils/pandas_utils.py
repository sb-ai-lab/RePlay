import collections
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

from replay.utils import PandasDataFrame


def filter_cold(
    df: Optional[pd.DataFrame],
    warm_df: PandasDataFrame,
    col_name: str,
) -> Tuple[int, Optional[pd.DataFrame]]:
    """
    Filter out new user/item ids absent in warm_df.
    Return number of new users/items and filtered dataframe.

    :param df: pandas dataframe with columns `[col_name, ...]`
    :param warm_df: pandas dataframe with column `[col_name]`,
    containing ids of warm users/items
    :param col_name: name of a column
    :return: filtered pandas dataframe with columns `[col_name, ...]`
    """
    if df is None:
        return 0, df

    unique_ids = set(df[col_name].unique())
    warm_ids = set(warm_df[col_name].unique())
    cold_ids = unique_ids - warm_ids
    num_cold = len(cold_ids)

    if num_cold == 0:
        return 0, df

    filtered_df = df[df[col_name].isin(warm_ids)]
    return num_cold, filtered_df


def get_unique_entities(
    df: Union[Iterable, pd.DataFrame],
    column: str,
) -> pd.DataFrame:
    """
    Get unique values from ``df`` and put them into a dataframe with column ``column``.

    :param df: pandas dataframe with column ``column`` or python iterable
    :param column: name of a column
    :return: pandas dataframe with column ``[column]``
    """
    if isinstance(df, pd.DataFrame):
        unique = df[[column]].drop_duplicates()
    elif isinstance(df, collections.abc.Iterable):
        unique = pd.DataFrame(pd.unique(list(df)), columns=[column])
    else:
        msg = f"Wrong type {type(df)}"
        raise ValueError(msg)
    return unique


def get_top_k(
    dataframe: PandasDataFrame,
    partition_by_col: str,
    order_by_col: List[Tuple[str, bool]],
    k: int,
) -> pd.DataFrame:
    """
    Return top `k` rows for each entity in `partition_by_col` ordered by `order_by_col`.

    For each unique value in the column specified by `partition_by_col`,
    the rows are sorted according to the columns in `order_by_col` (with the given
    ascending/descending flags) and the first `k` rows per group are returned.

    :param dataframe: pandas dataframe to filter.
    :param partition_by_col: column name to partition by.
    :param order_by_col: list of tuples (column_name, ascending) used to order rows within each partition.
    :param k: number of first rows for each entity in `partition_by_col` to return.
    :return: filtered pandas dataframe.
    """
    if dataframe.empty:
        return dataframe
    sort_columns = [partition_by_col] + [col for col, _ in order_by_col]
    sort_ascending = [True] + [asc for _, asc in order_by_col]

    sorted_df = dataframe.sort_values(by=sort_columns, ascending=sort_ascending)
    # Group by the partition column and take the first k rows from each group.
    top_k_df = sorted_df.groupby(partition_by_col, group_keys=False).head(k)
    return top_k_df.reset_index(drop=True)


def return_recs(recs: PandasDataFrame, recs_file_path: Optional[str] = None) -> pd.DataFrame:
    """
    Save dataframe `recs` to `recs_file_path` if presents otherwise cache
    and materialize the dataframe.

    :param recs: dataframe with recommendations
    :param recs_file_path: absolute path to save recommendations as a parquet file.
    :return: cached and materialized dataframe `recs` if `recs_file_path` is provided otherwise None
    """
    if recs_file_path is None:
        return recs

    recs.to_parquet(recs_file_path, index=False)
    return None
