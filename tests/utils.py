import os
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

from replay.data import Dataset, FeatureHint, FeatureInfo, FeatureSchema, FeatureType, get_schema
from replay.utils import PYSPARK_AVAILABLE, SparkDataFrame
from replay.utils.spark_utils import convert2spark

if PYSPARK_AVAILABLE:
    from pyspark.ml.linalg import DenseVector

    INTERACTIONS_SCHEMA = get_schema("user_idx", "item_idx", "timestamp", "relevance")


def assertDictAlmostEqual(d1: Dict, d2: Dict) -> None:
    assert set(d1.keys()) == set(d2.keys())
    for key in d1:
        assert_allclose(d1[key], d2[key])


def unify_dataframe(data_frame: SparkDataFrame):
    pandas_df = data_frame.toPandas()
    columns_to_sort_by: List[str] = []

    if len(pandas_df) == 0:
        columns_to_sort_by = pandas_df.columns
    else:
        for column in pandas_df.columns:
            if type(pandas_df[column][0]) not in {
                DenseVector,
                list,
                np.ndarray,
            }:
                columns_to_sort_by.append(column)

    return pandas_df[sorted(data_frame.columns)].sort_values(by=sorted(columns_to_sort_by)).reset_index(drop=True)


def sparkDataFrameEqual(df1: SparkDataFrame, df2: SparkDataFrame):
    return pd.testing.assert_frame_equal(unify_dataframe(df1), unify_dataframe(df2), check_like=True)


def sparkDataFrameNotEqual(df1: SparkDataFrame, df2: SparkDataFrame):
    try:
        sparkDataFrameEqual(df1, df2)
    except AssertionError:
        pass
    else:
        msg = "spark dataframes are equal"
        raise AssertionError(msg)


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


def create_dataset(log, user_features=None, item_features=None, feature_schema=None):
    log = convert2spark(log)
    if user_features is not None:
        user_features = convert2spark(user_features)
    if item_features is not None:
        item_features = convert2spark(item_features)

    if feature_schema is None:
        feature_schema = FeatureSchema(
            [
                FeatureInfo(
                    column="user_idx",
                    feature_type=FeatureType.CATEGORICAL,
                    feature_hint=FeatureHint.QUERY_ID,
                ),
                FeatureInfo(
                    column="item_idx",
                    feature_type=FeatureType.CATEGORICAL,
                    feature_hint=FeatureHint.ITEM_ID,
                ),
                FeatureInfo(
                    column="relevance",
                    feature_type=FeatureType.NUMERICAL,
                    feature_hint=FeatureHint.RATING,
                ),
                FeatureInfo(
                    column="timestamp",
                    feature_type=FeatureType.NUMERICAL,
                    feature_hint=FeatureHint.TIMESTAMP,
                ),
            ]
        )
    return Dataset(
        feature_schema=feature_schema,
        interactions=log,
        query_features=user_features,
        item_features=item_features,
        check_consistency=False,
    )
