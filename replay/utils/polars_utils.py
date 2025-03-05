import collections
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
import polars as pl


def filter_cold(
    df: Optional[pl.DataFrame],
    warm_df: pl.DataFrame,
    col_name: str,
) -> Tuple[int, Optional[pl.DataFrame]]:
    """
    Filter out new user/item ids absent in warm_df.
    Return number of new users/items and filtered DataFrame.

    :param df: polars DataFrame with columns `[col_name, ...]`
    :param warm_df: polars DataFrame with column `[col_name]`,
        containing ids of warm users/items
    :param col_name: name of a column
    :return: filtered polars DataFrame with columns `[col_name, ...]`
    """
    if df is None:
        return 0, df

    unique_ids = set(df[col_name].unique().to_list())
    warm_ids = set(warm_df[col_name].unique().to_list())
    cold_ids = unique_ids - warm_ids
    num_cold = len(cold_ids)

    if num_cold == 0:
        return 0, df

    filtered_df = df.filter(pl.col(col_name).is_in(list(warm_ids)))
    return num_cold, filtered_df


def get_unique_entities(
    df: Union[Iterable, pl.DataFrame],
    column: str,
) -> pl.DataFrame:
    """
    Get unique values from ``df`` and put them into a DataFrame with column ``column``.

    :param df: polars DataFrame with column ``column`` or a python iterable
    :param column: name of a column
    :return: polars DataFrame with column ``[column]``
    """
    if isinstance(df, pl.DataFrame):
        unique = df.select(pl.col(column)).unique()
    elif isinstance(df, collections.abc.Iterable):
        # Preserve order of appearance
        unique_sequence = pd.Series(pd.unique(list(df)), columns=[column])
        unique = pl.DataFrame(unique_sequence, schema=[f"{column}"])
    else:
        msg = f"Wrong type {type(df)}"
        raise ValueError(msg)
    return unique


def get_top_k(
    dataframe: pl.DataFrame,
    partition_by_col: str,
    order_by_col: List[Tuple[str, bool]],
    k: int,
) -> pl.DataFrame:
    """
    Return top `k` rows for each entity in `partition_by_col` ordered by `order_by_col`.

    For each unique value in the column specified by `partition_by_col`,
    the rows are sorted according to the columns in `order_by_col` (with the given
    ascending/descending flags) and the first `k` rows per group are returned.

    :param dataframe: polars DataFrame to filter.
    :param partition_by_col: column name to partition by.
    :param order_by_col: list of tuples (column_name, ascending) used to order rows within each partition.
    :param k: number of first rows for each entity in `partition_by_col` to return.
    :return: filtered polars DataFrame.
    """
    sort_columns = [partition_by_col] + [col for col, _ in order_by_col]
    sort_ascending = [True] + [asc for _, asc in order_by_col]
    # In polars the sort 'descending' parameter is the opposite of pandas 'ascending' parameter.
    sort_descending = [not asc for asc in sort_ascending]
    sorted_df = dataframe.sort(by=sort_columns, descending=sort_descending)
    # Group by the partition column and take the first k rows per group.
    top_k_df = sorted_df.group_by(partition_by_col, maintain_order=True).head(k)
    return top_k_df


def get_top_k_recs(
    recs: pl.DataFrame,
    k: int,
    query_column: str = "user_idx",
    rating_column: str = "relevance",
) -> pl.DataFrame:
    """
    Get top k recommendations by `rating`.

    For each unique query (user) in the recommendations DataFrame, the rows are ordered
    in descending order by the rating, and the top k rows are returned.

    :param recs: recommendations DataFrame with columns [query_column, item_idx, rating].
    :param k: length of a recommendation list.
    :param query_column: name of the column containing query (user) ids.
    :param rating_column: name of the column used for ordering (ratings).
    :return: top k recommendations DataFrame with columns [query_column, item_idx, rating].
    """
    return get_top_k(
        dataframe=recs,
        partition_by_col=query_column,
        order_by_col=[(rating_column, False)],
        k=k,
    )


def return_recs(recs: pl.DataFrame, recs_file_path: Optional[str] = None) -> Optional[pl.DataFrame]:
    """
    Save DataFrame `recs` to `recs_file_path` if provided otherwise cache
    and materialize the DataFrame.

    :param recs: DataFrame with recommendations
    :param recs_file_path: absolute path to save recommendations as a parquet file.
    :return: the cached and materialized DataFrame `recs` if `recs_file_path` is not provided, otherwise None.
    """
    if recs_file_path is None:
        return recs

    recs.write_parquet(recs_file_path)
    return None
