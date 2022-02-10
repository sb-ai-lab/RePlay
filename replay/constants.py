"""
Constants are stored here
"""
from typing import Iterable, Union

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StructField,
    StructType,
    TimestampType,
)

LOG_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType()),
        StructField("item_idx", IntegerType()),
        StructField("timestamp", TimestampType()),
        StructField("relevance", DoubleType()),
    ]
)

REC_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType()),
        StructField("item_idx", IntegerType()),
        StructField("relevance", DoubleType()),
    ]
)

BASE_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType()),
        StructField("item_idx", IntegerType()),
    ]
)

IntOrList = Union[Iterable[int], int]
NumType = Union[int, float]
AnyDataFrame = Union[DataFrame, pd.DataFrame]
