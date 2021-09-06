"""
Constants are stored here
"""
from typing import Iterable, Union

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

LOG_SCHEMA = StructType(
    [
        StructField("user_id", StringType()),
        StructField("item_id", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("relevance", DoubleType()),
    ]
)

REC_SCHEMA = StructType(
    [
        StructField("user_id", StringType()),
        StructField("item_id", StringType()),
        StructField("relevance", DoubleType()),
    ]
)

BASE_SCHEMA = StructType(
    [
        StructField("user_id", StringType()),
        StructField("item_id", StringType()),
    ]
)

IDX_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType()),
        StructField("item_idx", IntegerType()),
    ]
)

IDX_REC_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType()),
        StructField("item_idx", IntegerType()),
        StructField("relevance", DoubleType()),
    ]
)

IntOrList = Union[Iterable[int], int]
NumType = Union[int, float]
AnyDataFrame = Union[DataFrame, pd.DataFrame]
