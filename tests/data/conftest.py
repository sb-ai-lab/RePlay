import pandas as pd
import polars as pl
import pytest


@pytest.fixture(scope="module")
def full_events_pandas():
    return pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )


@pytest.fixture(scope="module")
def full_events_polars(full_events_pandas):
    return pl.from_pandas(full_events_pandas)


@pytest.fixture(scope="module")
def full_events_spark(spark, full_events_pandas):
    return spark.createDataFrame(full_events_pandas)


@pytest.fixture(scope="module")
def full_users_pandas():
    return pd.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )


@pytest.fixture(scope="module")
def full_users_polars(full_users_pandas):
    return pl.from_pandas(full_users_pandas)


@pytest.fixture(scope="module")
def full_users_spark(spark, full_users_pandas):
    return spark.createDataFrame(full_users_pandas)


@pytest.fixture(scope="module")
def full_items_pandas():
    return pd.DataFrame(
        {
            "item_id": [0, 1, 2, 3],
            "category_id": [0, 0, 1, 2],
            "feature1": [1.1, 1.2, 1.3, 1.4],
            "genres": [
                [0, 1],
                [2],
                [3, 0, 2, 1],
                [0, 0, 3, 2],
            ],
        }
    )


@pytest.fixture(scope="module")
def full_items_polars(full_items_pandas):
    return pl.from_pandas(full_items_pandas)


@pytest.fixture(scope="module")
def full_items_spark(spark, full_items_pandas):
    return spark.createDataFrame(full_items_pandas)


@pytest.fixture(scope="module")
def full_pandas_dataset(full_events_pandas, full_users_pandas, full_items_pandas):
    return {
        "interactions": full_events_pandas,
        "users": full_users_pandas,
        "items": full_items_pandas,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_polars_dataset(full_events_polars, full_users_polars, full_items_polars):
    return {
        "interactions": full_events_polars,
        "users": full_users_polars,
        "items": full_items_polars,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_spark_dataset(full_events_spark, full_users_spark, full_items_spark):
    return {
        "interactions": full_events_spark,
        "users": full_users_spark,
        "items": full_items_spark,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_pandas_dataset_cutted_interactions(full_users_pandas, full_items_pandas):
    events = pd.DataFrame(
        {
            "user_id": [0, 1, 1, 1],
            "item_id": [0, 0, 2, 3],
            "timestamp": [0, 2, 3, 4],
            "rating": [1.1, 1.3, 2, 3],
            "feature2": [1.1, 1.2, 1.3, 1.4],
        }
    )

    return {
        "interactions": events,
        "users": full_users_pandas,
        "items": full_items_pandas,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_polars_dataset_cutted_interactions(full_users_polars, full_items_polars):
    events = pl.DataFrame(
        {
            "user_id": [0, 1, 1, 1],
            "item_id": [0, 0, 2, 3],
            "timestamp": [0, 2, 3, 4],
            "rating": [1.1, 1.3, 2, 3],
            "feature2": [1.1, 1.2, 1.3, 1.4],
        }
    )

    return {
        "interactions": events,
        "users": full_users_polars,
        "items": full_items_polars,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def full_spark_dataset_cutted_interactions(spark, full_users_spark, full_items_spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 1, 1],
                "item_id": [0, 0, 2, 3],
                "timestamp": [0, 2, 3, 4],
                "rating": [1.1, 1.3, 2, 3],
                "feature2": [1.1, 1.2, 1.3, 1.4],
            }
        )
    )

    return {
        "interactions": events,
        "users": full_users_spark,
        "items": full_items_spark,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_item_full_pandas_dataset(full_users_pandas, full_items_pandas):
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 5],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "users": full_users_pandas,
        "items": full_items_pandas,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_item_full_polars_dataset(full_users_polars, full_items_polars):
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 5],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "users": full_users_polars,
        "items": full_items_polars,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_item_full_spark_dataset(spark, full_users_spark, full_items_spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 5],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    )

    return {
        "interactions": events,
        "users": full_users_spark,
        "items": full_items_spark,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_user_full_pandas_dataset(full_users_pandas, full_items_pandas):
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 3],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "users": full_users_pandas,
        "items": full_items_pandas,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_user_full_polars_dataset(full_users_polars, full_items_polars):
    events = pl.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 3],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    return {
        "interactions": events,
        "users": full_users_polars,
        "items": full_items_polars,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def inconsistent_user_full_spark_dataset(spark, full_users_spark, full_items_spark):
    events = spark.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 3],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    )

    return {
        "interactions": events,
        "users": full_users_spark,
        "items": full_items_spark,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def interactions_full_pandas_dataset(full_pandas_dataset):
    res = full_pandas_dataset.copy()
    res.pop("users")
    res.pop("items")
    return res


@pytest.fixture(scope="module")
def interactions_full_polars_dataset(full_polars_dataset):
    res = full_polars_dataset.copy()
    res.pop("users")
    res.pop("items")
    return res


@pytest.fixture(scope="module")
def interactions_full_spark_dataset(full_spark_dataset):
    res = full_spark_dataset.copy()
    res.pop("users")
    res.pop("items")
    return res


@pytest.fixture(scope="module")
def interactions_timestamp_pandas_dataset(full_pandas_dataset):
    res = full_pandas_dataset.copy()
    res.pop("users")
    res.pop("items")
    res["interactions"] = res["interactions"].drop("rating", axis=1)
    return res


@pytest.fixture(scope="module")
def interactions_timestamp_polars_dataset(full_polars_dataset):
    res = full_polars_dataset.copy()
    res.pop("users")
    res.pop("items")
    res["interactions"] = res["interactions"].drop("rating")
    return res


@pytest.fixture(scope="module")
def interactions_timestamp_spark_dataset(full_spark_dataset):
    res = full_spark_dataset.copy()
    res.pop("users")
    res.pop("items")
    res["interactions"] = res["interactions"].drop("rating")
    return res


@pytest.fixture(scope="module")
def interactions_rating_pandas_dataset(full_pandas_dataset):
    res = full_pandas_dataset.copy()
    res.pop("users")
    res.pop("items")
    res.pop("timestamp_col")
    res["interactions"] = res["interactions"].drop("timestamp", axis=1)
    return res


@pytest.fixture(scope="module")
def interactions_rating_polars_dataset(full_polars_dataset):
    res = full_polars_dataset.copy()
    res.pop("users")
    res.pop("items")
    res.pop("timestamp_col")
    res["interactions"] = res["interactions"].drop("timestamp")
    return res


@pytest.fixture(scope="module")
def interactions_rating_spark_dataset(full_spark_dataset):
    res = full_spark_dataset.copy()
    res.pop("users")
    res.pop("items")
    res.pop("timestamp_col")
    res["interactions"] = res["interactions"].drop("timestamp")
    return res


@pytest.fixture(scope="module")
def interactions_ids_spark_dataset(full_spark_dataset):
    res = full_spark_dataset.copy()
    res.pop("users")
    res.pop("items")
    res.pop("timestamp_col")
    res.pop("ratings_col")
    res["interactions"] = res["interactions"].select("user_id", "item_id")
    return res


@pytest.fixture(scope="module")
def interactions_users_pandas_dataset(full_pandas_dataset):
    without_items = full_pandas_dataset.copy()
    without_items.pop("items")
    return without_items


@pytest.fixture(scope="module")
def interactions_users_polars_dataset(full_polars_dataset):
    without_items = full_polars_dataset.copy()
    without_items.pop("items")
    return without_items


@pytest.fixture(scope="module")
def interactions_users_spark_dataset(full_spark_dataset):
    without_items = full_spark_dataset.copy()
    without_items.pop("items")
    return without_items


@pytest.fixture(scope="module")
def interactions_items_pandas_dataset(full_pandas_dataset):
    without_users = full_pandas_dataset.copy()
    without_users.pop("users")
    return without_users


@pytest.fixture(scope="module")
def interactions_items_polars_dataset(full_polars_dataset):
    without_users = full_polars_dataset.copy()
    without_users.pop("users")
    return without_users


@pytest.fixture(scope="module")
def interactions_items_spark_dataset(full_spark_dataset):
    without_users = full_spark_dataset.copy()
    without_users.pop("users")
    return without_users


@pytest.fixture(scope="module")
def wrong_user_pandas_dataset(full_pandas_dataset, full_users_spark):
    res = full_pandas_dataset.copy()
    res["users"] = full_users_spark
    return res


@pytest.fixture(scope="module")
def wrong_item_pandas_dataset(full_pandas_dataset, full_items_spark):
    res = full_pandas_dataset.copy()
    res["items"] = full_items_spark
    return res


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
def full_pandas_dataset_nonunique_columns(full_items_pandas):
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

    return {
        "interactions": events,
        "users": users,
        "items": full_items_pandas,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def users_pandas_dataset_different_columns(full_events_pandas):
    users = pd.DataFrame(
        {
            "user_ids": [0, 1, 2],
            "gender": [0, 1, 0],
        }
    )

    return {
        "interactions": full_events_pandas,
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
def items_pandas_dataset_different_columns(full_events_pandas):
    items = pd.DataFrame(
        {
            "item_ids": [0, 1, 2, 3],
            "category_id": [0, 0, 1, 2],
            "feature1": [1.1, 1.2, 1.3, 1.4],
            "genres": [
                [0, 1],
                [2],
                [3, 0, 2, 1],
                [0, 0, 3, 2],
            ],
        }
    )

    return {
        "interactions": full_events_pandas,
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
def full_polars_dataset_nonunique_columns(full_events_polars, full_items_pandas):
    users = pl.DataFrame(
        {
            "user_id": [0, 1, 2],
            "gender": [0, 1, 0],
            "feature1": [0.1, 0.2, 0.3],
        }
    )

    return {
        "interactions": full_events_polars,
        "users": users,
        "items": full_items_pandas,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def users_polars_dataset_different_columns(full_events_polars, full_users_polars):
    return {
        "interactions": full_events_polars,
        "users": full_users_polars,
        "user_col": "user_id",
        "user_col2": "user_ids",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture(scope="module")
def items_polars_dataset_different_columns(full_events_polars):
    items = pl.DataFrame(
        {
            "item_ids": [0, 1, 2, 3],
            "category_id": [0, 0, 1, 2],
            "feature1": [1.1, 1.2, 1.3, 1.4],
            "genres": [
                [0, 1],
                [2],
                [3, 0, 2, 1],
                [0, 0, 3, 2],
            ],
        }
    )

    return {
        "interactions": full_events_polars,
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
def items_pandas_dataset_string_genres(full_events_pandas):
    items = pd.DataFrame(
        {
            "item_id": [0, 1, 2, 3],
            "category_id": [0, 0, 1, 2],
            "feature1": [1.1, 1.2, 1.3, 1.4],
            "genres": [
                ["Animation", "Fantasy"],
                ["Action"],
                ["Comedy", "Animation", "Action", "Fantasy"],
                ["Animation", "Animation", "Comedy", "Action"],
            ],
        }
    )

    return {
        "interactions": full_events_pandas,
        "items": items,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }
