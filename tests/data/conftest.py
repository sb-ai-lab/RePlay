import pandas as pd
import pytest


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def full_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    )

    users = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "gender": [0, 1, 0],
            }
        )
    )

    items = spark_session.createDataFrame(
        pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})
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


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def full_spark_dataset_cutted_interactions(spark_session):
    events = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 1, 1],
                "item_id": [0, 0, 2, 3],
                "timestamp": [0, 2, 3, 4],
                "rating": [1.1, 1.3, 2, 3],
                "feature1": [1.1, 1.2, 1.3, 1.4],
            }
        )
    )

    users = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "gender": [0, 1, 0],
            }
        )
    )

    items = spark_session.createDataFrame(
        pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature2": [1.1, 1.2, 1.3, 1.4]})
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


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def inconsistent_item_full_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 5],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    )

    users = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "gender": [0, 1, 0],
            }
        )
    )

    items = spark_session.createDataFrame(
        pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})
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


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def inconsistent_user_full_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 3],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    )

    users = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 1, 2],
                "gender": [0, 1, 0],
            }
        )
    )

    items = spark_session.createDataFrame(
        pd.DataFrame({"item_id": [0, 1, 2, 3], "category_id": [0, 0, 1, 2], "feature1": [1.1, 1.2, 1.3, 1.4]})
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


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def interactions_full_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
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


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def interactions_timestamp_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
            }
        )
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "timestamp_col": "timestamp",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def interactions_rating_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
        pd.DataFrame({"user_id": [0, 0, 1, 1, 1, 2], "item_id": [0, 1, 0, 2, 3, 1], "rating": [1.1, 1.2, 1.3, 2, 3, 4]})
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "ratings_col": "rating",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def interactions_ids_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
            }
        )
    )

    return {
        "interactions": events,
        "user_col": "user_id",
        "item_col": "item_id",
        "users_cardinality": 3,
        "items_cardinality": 4,
    }


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def interactions_users_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    )

    users = spark_session.createDataFrame(
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


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def interactions_items_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
        pd.DataFrame(
            {
                "user_id": [0, 0, 1, 1, 1, 2],
                "item_id": [0, 1, 0, 2, 3, 1],
                "timestamp": [0, 1, 2, 3, 4, 5],
                "rating": [1.1, 1.2, 1.3, 2, 3, 4],
            }
        )
    )

    items = spark_session.createDataFrame(
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def wrong_user_pandas_dataset(spark_session):
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    users = spark_session.createDataFrame(
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def wrong_item_pandas_dataset(spark_session):
    events = pd.DataFrame(
        {
            "user_id": [0, 0, 1, 1, 1, 2],
            "item_id": [0, 1, 0, 2, 3, 1],
            "timestamp": [0, 1, 2, 3, 4, 5],
            "rating": [1.1, 1.2, 1.3, 2, 3, 4],
        }
    )

    items = spark_session.createDataFrame(
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def not_int_user_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def not_int_item_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def less_than_zero_user_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def less_than_zero_item_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def more_than_count_user_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
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


@pytest.fixture
@pytest.mark.usefixtures("spark_session")
def more_than_count_item_spark_dataset(spark_session):
    events = spark_session.createDataFrame(
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


@pytest.fixture
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


@pytest.fixture
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


@pytest.fixture
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
