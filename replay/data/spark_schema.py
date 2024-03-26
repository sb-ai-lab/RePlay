from replay.utils import PYSPARK_AVAILABLE

if PYSPARK_AVAILABLE:
    from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType, TimestampType


def get_schema(
    query_column: str = "query_id",
    item_column: str = "item_id",
    timestamp_column: str = "timestamp",
    rating_column: str = "rating",
    has_timestamp: bool = True,
    has_rating: bool = True,
):
    """
    Get Spark Schema with query_id, item_id, rating, timestamp columns

    :param query_column: column name with query ids
    :param item_column: column name with item ids
    :param timestamp_column: column name with timestamps
    :param rating_column: column name with ratings
    :param has_rating: flag to add rating to schema
    :param has_timestamp: flag to add tomestamp to schema
    """
    base = [
        StructField(query_column, IntegerType()),
        StructField(item_column, IntegerType()),
    ]
    if has_timestamp:
        base += [StructField(timestamp_column, TimestampType())]
    if has_rating:
        base += [StructField(rating_column, DoubleType())]
    return StructType(base)
