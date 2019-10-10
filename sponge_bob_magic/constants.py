"""
Библиотека рекомендательных систем Лаборатории по искусственному интеллекту
"""
from pyspark.sql.types import (DoubleType, FloatType, StringType, StructField,
                               StructType, TimestampType)

LOG_SCHEMA = StructType([
    StructField("user_id", StringType()),
    StructField("item_id", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("context", StringType()),
    StructField("relevance", FloatType())
])

DEFAULT_CONTEXT = 'no_context'

REC_SCHEMA = StructType([
    StructField("user_id", StringType()),
    StructField("item_id", StringType()),
    StructField("context", StringType()),
    StructField("relevance", DoubleType())
])
