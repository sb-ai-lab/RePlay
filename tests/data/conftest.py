import pandas as pd
import polars as pl
import pytest

from tests.utils import DEFAULT_SPARK_NUM_PARTITIONS


@pytest.fixture(scope="module")
def full_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    items = pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pl.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    items = pl.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    users = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "gender": [0, 1, 0],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    items = spark.createDataFrame(
        pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_pandas_dataset_cutted_interactions():
    events = pd.DataFrame(
        {
            "user_id": [0, 1, 1, 1],
            "item_id": [0, 0, 2, 3],
            "timestamp": [0, 2, 3, 4],
            "rating": [1.1, 1.3, 2, 3],
            "feature1": [1.1, 1.2, 1.3, 1.4],
        }
    )

    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    items = pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature2": [1.1, 1.2, 1.3, 1.4]})

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_polars_dataset_cutted_interactions():
    events = pl.DataFrame(
        {
            "user_id": [0, 1, 1, 1],
            "item_id": [0, 0, 2, 3],
            "timestamp": [0, 2, 3, 4],
            "rating": [1.1, 1.3, 2, 3],
            "feature1": [1.1, 1.2, 1.3, 1.4],
        }
    )

    users = pl.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    items = pl.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature2": [1.1, 1.2, 1.3, 1.4]})

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_spark_dataset_cutted_interactions(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 1, 1],
                "item_id": [0, 0, 2, 3],
                "timestamp": [0, 2, 3, 4],
                "rating": [1.1, 1.3, 2, 3],
                "feature1": [1.1, 1.2, 1.3, 1.4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    users = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "gender": [0, 1, 0],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    items = spark.createDataFrame(
        pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature2": [1.1, 1.2, 1.3, 1.4]})
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_item_full_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 5],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    items = pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_item_full_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 5],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pl.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    items = pl.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_item_full_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 5],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    users = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "gender": [0, 1, 0],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    items = spark.createDataFrame(
        pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_user_full_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 3],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    items = pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_user_full_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 3],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pl.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    items = pl.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_user_full_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 3],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    users = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "gender": [0, 1, 0],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    items = spark.createDataFrame(
        pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_full_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_full_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_full_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_timestamp_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_timestamp_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_timestamp_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_rating_pandas_dataset():
    events = pd.DataFrame(
        {"user_id": [0, 0, 1, 1, 1, 2], "item_id": [0, 1, 0, 2, 3, 1], "rating": [1.1, 1.2, 1.3, 2, 3, 4]}
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_rating_polars_dataset():
    events = pl.DataFrame(
        {"user_id": [0, 0, 1, 1, 1, 2], "item_id": [0, 1, 0, 2, 3, 1], "rating": [1.1, 1.2, 1.3, 2, 3, 4]}
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_rating_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame({"user_id": [0, 0, 1, 1, 1, 2], "item_id": [0, 1, 0, 2, 3, 1], "rating": [1.1, 1.2, 1.3, 2, 3, 4]})
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_ids_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_users_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    return {
        "interactions": events,
        "users": users,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_users_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pl.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    return {
        "interactions": events,
        "users": users,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_users_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    users = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "gender": [0, 1, 0],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "users": users,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_items_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    items = pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})

    return {
        "interactions": events,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_items_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    items = pl.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})

    return {
        "interactions": events,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_items_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    items = spark.createDataFrame(
        pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def wrong_user_pandas_dataset(spark):
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "gender": [0, 1, 0],
            }
        )
    )

    return {
        "interactions": events,
        "users": users,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def wrong_item_pandas_dataset(spark):
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    items = spark.createDataFrame(
        pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})
    )

    return {
        "interactions": events,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def not_int_user_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": ["0", "0", "1", "1", "1", "2"],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def not_int_item_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": ["0", "1", "0", "2", "3", "1"],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def not_int_user_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": ["0", "0", "1", "1", "1", "2"],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def not_int_item_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": ["0", "1", "0", "2", "3", "1"],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def not_int_user_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": ["0", "0", "1", "1", "1", "2"],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def not_int_item_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": ["0", "1", "0", "2", "3", "1"],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def less_than_zero_user_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, -1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 4,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def less_than_zero_item_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, -1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 5,
    }


@pytest.fixture(scope="module")
def less_than_zero_user_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, -1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 4,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def less_than_zero_item_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, -1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 5,
    }


@pytest.fixture(scope="module")
def less_than_zero_user_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, -1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 4,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def less_than_zero_item_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, -1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 5,
    }


@pytest.fixture(scope="module")
def more_than_count_user_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 10, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 4,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def more_than_count_item_pandas_dataset():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 10, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 5,
    }


@pytest.fixture(scope="module")
def more_than_count_user_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 10, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 4,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def more_than_count_item_polars_dataset():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 10, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 5,
    }


@pytest.fixture(scope="module")
def more_than_count_user_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 10, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 4,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def more_than_count_item_spark_dataset(spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 10, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    ).repartition(DEFAULT_SPARK_NUM_PARTITIONS)

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 5,
    }


@pytest.fixture(scope="module")
def full_pandas_dataset_nonunique_columns():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
            "feature1": [0.1, 0.2, 0.3],
        }
    )

    items = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 3],
            "category_id": [0, 0, 1, 2],
            "feature1": [1.1, 1.2, 1.3, 1.4],
        }
    )

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def users_pandas_dataset_different_columns():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pd.DataFrame(
        {
            "user_ids": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    return {
        "interactions": events,
        "users": users,
        "user_col": "user_id",
        "user_col2": "user_ids",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def items_pandas_dataset_different_columns():
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    items = pd.DataFrame(
        {
            "item_ids": [0, 1, 2, 3],
            "category_id": [0, 0, 1, 2],
            "feature1": [1.1, 1.2, 1.3, 1.4],
        }
    )

    return {
        "interactions": events,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "item_col2": "item_ids",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_polars_dataset_nonunique_columns():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pl.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
            "feature1": [0.1, 0.2, 0.3],
        }
    )

    items = pl.DataFrame(
        {
            "item_id": [0, 1, 2, 3],
            "category_id": [0, 0, 1, 2],
            "feature1": [1.1, 1.2, 1.3, 1.4],
        }
    )

    return {
        "interactions": events,
        "users": users,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def users_polars_dataset_different_columns():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = pl.DataFrame(
        {
            "user_ids": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    return {
        "interactions": events,
        "users": users,
        "user_col": "user_id",
        "user_col2": "user_ids",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def items_polars_dataset_different_columns():
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    items = pl.DataFrame(
        {
            "item_ids": [0, 1, 2, 3],
            "category_id": [0, 0, 1, 2],
            "feature1": [1.1, 1.2, 1.3, 1.4],
        }
    )

    return {
        "interactions": events,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "item_col2": "item_ids",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }
