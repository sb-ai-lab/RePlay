from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StructField,
    StructType,
    TimestampType,
)
from replay.data.dataset import Dataset


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
        StructField("user_id", IntegerType()),
        StructField("item_id", IntegerType()),
        StructField("relevance", DoubleType()),
    ]
)


BASE_SCHEMA = StructType(
    [
        StructField("user_idx", IntegerType()),
        StructField("item_idx", IntegerType()),
    ]
)


def get_rec_schema(dataset: Dataset):
    return StructType(
        [
            StructField(dataset.feature_schema.query_id_column, IntegerType()),
            StructField(dataset.feature_schema.item_id_column, IntegerType()),
            StructField(dataset.feature_schema.interactions_rating_column, DoubleType()),
        ]
)