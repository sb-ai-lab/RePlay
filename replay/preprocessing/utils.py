from typing import Literal, List, Optional, Sequence

import polars as pl
import pandas as pd

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
        raise ValueError(
            f"Columns mismatch in dataframe #{index}: {sorted(df.columns)} != {sorted(ref_cols)}"
        )


def _merge_subsets_pandas(
    dfs: Sequence[PandasDataFrame],
    columns: Optional[Sequence[str]],
    check_columns: bool,
    subset_for_duplicates: Optional[Sequence[str]],
    on_duplicate: Literal["error", "drop", "ignore"],
) -> PandasDataFrame:
    if not dfs:
        raise ValueError("At least one dataframe is required")

    ref_cols = list(dfs[0].columns) if columns is None else list(columns)

    aligned: List[PandasDataFrame] = []
    for i, df in enumerate(dfs):
        _ensure_columns_match(df, ref_cols, i, check_columns)
        aligned.append(df[ref_cols])

    merged = pd.concat(aligned, axis=0, ignore_index=True)

    dup_subset = (
        ref_cols if subset_for_duplicates is None else list(subset_for_duplicates)
    )
    dup_mask = merged.duplicated(subset=dup_subset, keep="first")
    dup_count = int(dup_mask.sum())

    if dup_count > 0:
        if on_duplicate == "error":
            sample = merged.loc[dup_mask, dup_subset].head(5)
            raise ValueError(
                f"Found {dup_count} duplicate rows on subset {dup_subset}. Sample:\n{sample}"
            )
        if on_duplicate == "drop":
            merged = merged.drop_duplicates(
                subset=dup_subset, keep="first"
            ).reset_index(drop=True)

    return merged


def _merge_subsets_polars(
    dfs: Sequence[PolarsDataFrame],
    columns: Optional[Sequence[str]],
    check_columns: bool,
    subset_for_duplicates: Optional[Sequence[str]],
    on_duplicate: Literal["error", "drop", "ignore"],
) -> PolarsDataFrame:
    if not dfs:
        raise ValueError("At least one dataframe is required")

    ref_cols = list(dfs[0].columns) if columns is None else list(columns)

    aligned: List[PolarsDataFrame] = []
    for i, df in enumerate(dfs):
        _ensure_columns_match(df, ref_cols, i, check_columns)
        aligned.append(df.select(ref_cols))

    merged = pl.concat(aligned, how="vertical")

    dup_subset = (
        ref_cols if subset_for_duplicates is None else list(subset_for_duplicates)
    )
    dup_mask = merged.is_duplicated(subset=dup_subset)
    dup_count = int(dup_mask.sum())

    if dup_count > 0:
        if on_duplicate == "error":
            raise ValueError(f"Found {dup_count} duplicate rows on subset {dup_subset}")
        if on_duplicate == "drop":
            merged = merged.unique(subset=dup_subset, keep="first", maintain_order=True)

    return merged


def _merge_subsets_spark(
    dfs: Sequence[SparkDataFrame],
    *,
    columns: Optional[Sequence[str]],
    check_columns: bool,
    subset_for_duplicates: Optional[Sequence[str]],
    on_duplicate: Literal["error", "drop", "ignore"],
) -> SparkDataFrame:
    if not dfs:
        raise ValueError("At least one dataframe is required")

    ref_cols = list(dfs[0].columns) if columns is None else list(columns)

    merged = None
    for i, df in enumerate(dfs):
        _ensure_columns_match(df, ref_cols, i, check_columns)
        part = df.select(*ref_cols)
        merged = part if merged is None else merged.unionByName(part)

    dup_subset = (
        ref_cols if subset_for_duplicates is None else list(subset_for_duplicates)
    )
    if on_duplicate in ("error", "drop"):
        dup_groups = merged.groupBy(*dup_subset).count().filter(sf.col("count") > 1)
        has_dups = dup_groups.limit(1).count() > 0
        if has_dups and on_duplicate == "error":
            raise ValueError(f"Found duplicate rows on subset {dup_subset}")
        if has_dups and on_duplicate == "drop":
            merged = merged.dropDuplicates(dup_subset)

    return merged


def merge_subsets(
    *dfs: DataFrameLike,
    columns: Optional[Sequence[str]] = None,
    check_columns: bool = True,
    subset_for_duplicates: Optional[Sequence[str]] = None,
    on_duplicate: Literal["error", "drop", "ignore"] = "error",
) -> DataFrameLike:
    """
    Vertically concatenate multiple datasets of the same backend with column alignment
    and optional duplicate control.

    The function accepts pandas, Polars, or Spark DataFrames, but all inputs must
    belong to the same backend type. Columns are aligned to the order given by
    ``columns`` (or by the first dataframe if ``columns`` is None). Duplicate rows
    can be handled according to ``on_duplicate``.

    Parameters
    ----------
    dfs : DataFrameLike
        One or more dataframes to concatenate.
    columns : Sequence[str] | None, default None
        Explicit column order/subset to use. If None, the columns of the first
        dataframe define the order.
    check_columns : bool, default True
        Validate that every dataframe has exactly the same set of columns as the
        reference set (defined by ``columns`` or the first dataframe).
    subset_for_duplicates : Sequence[str] | None, default None
        Columns to use for duplicate detection. If None, all columns are used.
    on_duplicate : {"error", "drop", "ignore"}, default "error"
        Duplicate handling policy: raise an error, drop duplicates keeping the
        first occurrence, or ignore them.

    Returns
    -------
    DataFrameLike
        Concatenated dataframe of the same backend type as the inputs.

    Raises
    ------
    ValueError
        If no dataframes are provided or column sets do not match when
        ``check_columns`` is True.
    TypeError
        If inputs are not all of the same backend type.
    NotImplementedError
        If the dataframe backend is not supported.
    """
    if not dfs:
        raise ValueError("At least one dataframe is required")

    first = dfs[0]
    if any(not isinstance(df, type(first)) for df in dfs):
        raise TypeError("All input dataframes must be of the same type")

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

    raise NotImplementedError(f"Unsupported data frame type: {type(first)}")
