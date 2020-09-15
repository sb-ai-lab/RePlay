"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту.
"""
from typing import Iterable, Union

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    DoubleType,
    FloatType,
    StringType,
    StructField,
    StructType,
    TimestampType,
    IntegerType,
)

LOG_SCHEMA = StructType(
    [
        StructField("user_id", StringType()),
        StructField("item_id", StringType()),
        StructField("timestamp", TimestampType()),
        StructField("relevance", FloatType()),
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

IntOrList = Union[Iterable[int], int]
NumType = Union[int, float]
AnyDataFrame = Union[DataFrame, pd.DataFrame]
