from pyspark.sql.types import (FloatType, StringType, StructField, StructType,
                               TimestampType)

LOG_SCHEMA = StructType([
    StructField("user_id", StringType()),
    StructField("item_id", StringType()),
    StructField("timestamp", TimestampType()),
    StructField("context", StringType()),
    StructField("relevance", FloatType())
])

DEFAULT_CONTEXT = 'no_context'
