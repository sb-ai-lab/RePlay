from typing import List, Literal, Optional, Sequence

import logging
import pandas as pd
import polars as pl

from replay.utils import (
    PYSPARK_AVAILABLE,
    DataFrameLike,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame,
)

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as sf


def _ensure_columns_match(df, ref_cols, index: int, check_columns: bool) -> None:
    if check_columns and set(df.columns) != set(ref_cols):
        msg = f"Columns mismatch in dataframe #{index}: {sorted(df.columns)} != {sorted(ref_cols)}"
        raise ValueError(msg)


def _merge_subsets_pandas(
    dfs: Sequence[PandasDataFrame],
    columns: Optional[Sequence[str]],
    check_columns: bool,
    subset_for_duplicates: Optional[Sequence[str]],
    on_duplicate: Literal["error", "drop", "ignore"],
) -> PandasDataFrame:
    ref_cols = list(dfs[0].columns) if columns is None else list(columns)

    aligned: List[PandasDataFrame] = []
    for i, df in enumerate(dfs):
        _ensure_columns_match(df, ref_cols, i, check_columns)
        aligned.append(df[ref_cols])

    merged = pd.concat(aligned, axis=0, ignore_index=True)

    if on_duplicate == "ignore":
        return merged

    dup_subset = (
        ref_cols if subset_for_duplicates is None else list(subset_for_duplicates)
    )
    dup_mask = merged.duplicated(subset=dup_subset, keep="first")
    dup_count = int(dup_mask.sum())

    if dup_count > 0:
        if on_duplicate == "error":
            msg = f"Found {dup_count} duplicate rows on subset {dup_subset}"
            raise ValueError(msg)
        if on_duplicate == "drop":
            merged = merged.drop_duplicates(
                subset=dup_subset, keep="first"
            ).reset_index(drop=True)
            logging.getLogger("replay").warning(
                f"Found {dup_count} duplicate rows on subset {dup_subset} and dropped them"
            )

    return merged


def _merge_subsets_polars(
    dfs: Sequence[PolarsDataFrame],
    columns: Optional[Sequence[str]],
    check_columns: bool,
    subset_for_duplicates: Optional[Sequence[str]],
    on_duplicate: Literal["error", "drop", "ignore"],
) -> PolarsDataFrame:
    ref_cols = list(dfs[0].columns) if columns is None else list(columns)

    aligned: List[PolarsDataFrame] = []
    for i, df in enumerate(dfs):
        _ensure_columns_match(df, ref_cols, i, check_columns)
        aligned.append(df.select(ref_cols))

    merged = pl.concat(aligned, how="vertical")

    if on_duplicate == "ignore":
        return merged

    dup_subset = (
        ref_cols if subset_for_duplicates is None else list(subset_for_duplicates)
    )
    dup_mask = merged.is_duplicated(subset=dup_subset)
    dup_count = int(dup_mask.sum())

    if dup_count > 0:
        if on_duplicate == "error":
            msg = f"Found {dup_count} duplicate rows on subset {dup_subset}"
            raise ValueError(msg)
        if on_duplicate == "drop":
            merged = merged.unique(subset=dup_subset, keep="first", maintain_order=True)
            logging.getLogger("replay").warning(
                f"Found {dup_count} duplicate rows on subset {dup_subset} and dropped them"
            )

    return merged


def _merge_subsets_spark(
    dfs: Sequence[SparkDataFrame],
    columns: Optional[Sequence[str]],
    check_columns: bool,
    subset_for_duplicates: Optional[Sequence[str]],
    on_duplicate: Literal["error", "drop", "ignore"],
) -> SparkDataFrame:
    ref_cols = list(dfs[0].columns) if columns is None else list(columns)

    merged = None
    for i, df in enumerate(dfs):
        _ensure_columns_match(df, ref_cols, i, check_columns)
        part = df.select(*ref_cols)
        merged = part if merged is None else merged.unionByName(part)

    if on_duplicate == "ignore":
        return merged

    dup_subset = (
        ref_cols if subset_for_duplicates is None else list(subset_for_duplicates)
    )
    if (
        on_duplicate == "error"
        and merged.groupBy(*dup_subset)
        .count()
        .filter(sf.col("count") > 1)
        .limit(1)
        .count()
        > 0
    ):
        msg = f"Found duplicate rows on subset {dup_subset}"
        raise ValueError(msg)
    if on_duplicate == "drop":
        unique = merged.dropDuplicates(dup_subset)
        logging.getLogger("replay").warning(
            f"Found {merged.count() - unique.count()} duplicate rows on subset {dup_subset} and dropped them"
        )
        merged = unique

    return merged


def merge_subsets(
    dfs: Sequence[DataFrameLike],
    columns: Optional[Sequence[str]] = None,
    check_columns: bool = True,
    subset_for_duplicates: Optional[Sequence[str]] = None,
    on_duplicate: Literal["error", "drop", "ignore"] = "error",
) -> DataFrameLike:
    """Merge multiple dataframes of the same backend into a single one.

    All inputs must be of the same dataframe type (pandas/Polars/Spark). Before
    concatenation, each dataframe is aligned to a common set of columns: either
    the provided ``columns`` or the columns of the first dataframe. Duplicate
    rows are handled according to ``on_duplicate``.

    Parameters
    ----------
    dfs : Sequence[DataFrameLike]
        Dataframes to merge.
    columns : Optional[Sequence[str]]
        Columns to align to. If ``None``, columns of the first dataframe are used.
    check_columns : bool
        Whether to validate that all inputs have the same column set.
    subset_for_duplicates : Optional[Sequence[str]]
        Columns subset used to detect duplicates. If ``None``, all aligned columns
        are used.
    on_duplicate : {"error", "drop", "ignore"}
        How to handle duplicates: raise an error, drop them, or ignore.

    Returns
    -------
    DataFrameLike
        Merged dataframe of the same backend as the inputs.

    Raises
    ------
    ValueError
        If ``dfs`` is empty, if duplicates are found with ``on_duplicate='error'``,
        or if column sets differ when validation is enabled.
    TypeError
        If inputs are of different dataframe types.
    """
    if not dfs:
        msg = "At least one dataframe is required"
        raise ValueError(msg)

    first = dfs[0]
    if any(not isinstance(df, type(first)) for df in dfs):
        msg = "All input dataframes must be of the same type"
        raise TypeError(msg)

    if isinstance(first, PandasDataFrame):
        return _merge_subsets_pandas(
            dfs,
            columns=columns,
            check_columns=check_columns,
            subset_for_duplicates=subset_for_duplicates,
            on_duplicate=on_duplicate,
        )
    if isinstance(first, PolarsDataFrame):
        return _merge_subsets_polars(
            dfs,
            columns=columns,
            check_columns=check_columns,
            subset_for_duplicates=subset_for_duplicates,
            on_duplicate=on_duplicate,
        )
    if isinstance(first, SparkDataFrame):
        return _merge_subsets_spark(
            dfs,
            columns=columns,
            check_columns=check_columns,
            subset_for_duplicates=subset_for_duplicates,
            on_duplicate=on_duplicate,
        )

    msg = f"Unsupported data frame type: {type(first)}"
    raise NotImplementedError(msg)
