from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StructField,
    StructType,
    TimestampType,
)


def get_interactions_schema(
    query_column: str = "query_id",
    item_column: str = "item_id",
    timestamp_column: str = "timestamp",
    rating_column: str = "rating",
):
    """
    Get Spark Schema with query_id, item_id, rating, timestamp columns

    :param query_column: column name with query ids
    :param item_column: column name with item ids
    :param timestamp_column: column name with timestamps
    :param rating_column: column name with ratings
    """
    return StructType(
        [
            StructField(query_column, IntegerType()),
            StructField(item_column, IntegerType()),
            StructField(timestamp_column, TimestampType()),
            StructField(rating_column, DoubleType()),
        ]
    )


def get_rec_schema(
    query_column: str = "query_id",
    item_column: str = "item_id",
    rating_column: str = "rating",
):
    """
    Get Spark Schema with query_id, item_id, rating columns

    :param query_column: column name with query ids
    :param item_column: column name with item ids
    :param rating_column: column name with ratings
    """
    return StructType(
        [
            StructField(query_column, IntegerType()),
            StructField(item_column, IntegerType()),
            StructField(rating_column, DoubleType()),
        ]
    )


def get_base_schema(
    query_column: str = "query_id",
    item_column: str = "item_id",
):
    """
    Get Spark Schema with query_id, item_id columns

    :param query_column: column name with query ids
    :param item_column: column name with item ids
    """
    return StructType(
        [
            StructField(query_column, IntegerType()),
            StructField(item_column, IntegerType()),
        ]
    )
