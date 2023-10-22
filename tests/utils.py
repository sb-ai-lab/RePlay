# pylint: skip-file
import os
import re

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from pyspark.ml.linalg import DenseVector
from pyspark.sql import DataFrame


def assertDictAlmostEqual(d1: Dict, d2: Dict) -> None:
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        assert_allclose(d1[key], d2[key])


def unify_dataframe(data_frame: DataFrame):
    pandas_df = data_frame.toPandas()
    columns_to_sort_by: List[str] = []

    if len(pandas_df) == 0:
        columns_to_sort_by = pandas_df.columns
    else:
        for column in pandas_df.columns:
            if not type(pandas_df[column][0]) in {
                DenseVector,
                list,
                np.ndarray,
            }:
                columns_to_sort_by.append(column)

    return (
        pandas_df[sorted(data_frame.columns)]
        .sort_values(by=sorted(columns_to_sort_by))
        .reset_index(drop=True)
    )


def sparkDataFrameEqual(df1: DataFrame, df2: DataFrame):
    return pd.testing.assert_frame_equal(
        unify_dataframe(df1), unify_dataframe(df2), check_like=True
    )


def sparkDataFrameNotEqual(df1: DataFrame, df2: DataFrame):
    try:
        sparkDataFrameEqual(df1, df2)
    except AssertionError:
        pass
    else:
        raise AssertionError("spark dataframes are equal")


def del_files_by_pattern(directory: str, pattern: str) -> None:
    """
    Deletes files by pattern
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            os.remove(os.path.join(directory, filename))


def find_file_by_pattern(directory: str, pattern: str) -> Optional[str]:
    """
    Returns path to first found file, if exists
    """
    for filename in os.listdir(directory):
        if re.match(pattern, filename):
            return os.path.join(directory, filename)
    return None
