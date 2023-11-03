from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StructField,
    StructType,
    TimestampType,
)
from replay.data.dataset import Dataset


INTERACTIONS_SCHEMA = StructType(
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


def get_interactions_schema(dataset: Dataset):
    """
    Get Spark Schema with query_id, item_id, timestamp, rating columns

    :param dataset: instance of Dataset
    """
    return StructType(
        [
            StructField(dataset.feature_schema.query_id_column, IntegerType()),
            StructField(dataset.feature_schema.item_id_column, IntegerType()),
            StructField(dataset.feature_schema.interactions_timestamp_column, TimestampType()),
            StructField(dataset.feature_schema.interactions_rating_column, DoubleType()),
        ]
    )


def get_rec_schema(dataset: Dataset):
    """
    Get Spark Schema with query_id, item_id, rating columns

    :param dataset: instance of Dataset
    """
    return StructType(
        [
            StructField(dataset.feature_schema.query_id_column, IntegerType()),
            StructField(dataset.feature_schema.item_id_column, IntegerType()),
            StructField(dataset.feature_schema.interactions_rating_column, DoubleType()),
        ]
    )


def get_base_schema(dataset: Dataset):
    """
    Get Spark Schema with query_id, item_id columns

    :param dataset: instance of Dataset
    """
    return StructType(
        [
            StructField(dataset.feature_schema.query_id_column, IntegerType()),
            StructField(dataset.feature_schema.item_id_column, IntegerType()),
        ]
    )
