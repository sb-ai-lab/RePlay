import pandas as pd
import pytest

recs_data = [
    (1, 3, 0.6),
    (1, 7, 0.5),
    (1, 10, 0.4),
    (1, 11, 0.3),
    (1, 2, 0.2),
    (2, 5, 0.6),
    (2, 8, 0.5),
    (2, 11, 0.4),
    (2, 1, 0.3),
    (2, 3, 0.2),
    (3, 4, 1.0),
    (3, 9, 0.5),
    (3, 2, 0.1),
]


groundtruth_data = [
    (1, 5),
    (1, 6),
    (1, 7),
    (1, 8),
    (1, 9),
    (1, 10),
    (2, 6),
    (2, 7),
    (2, 4),
    (2, 10),
    (2, 11),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
]


train_data = [
    (1, 5),
    (1, 6),
    (1, 8),
    (1, 9),
    (1, 2),
    (2, 5),
    (2, 8),
    (2, 11),
    (2, 1),
    (2, 3),
    (3, 4),
    (3, 9),
    (3, 2),
]


base_recs_data = [
    (1, 3, 0.5),
    (1, 7, 0.5),
    (1, 2, 0.7),
    (2, 5, 0.6),
    (2, 8, 0.6),
    (2, 3, 0.3),
    (3, 4, 1.0),
    (3, 9, 0.5),
]


@pytest.fixture(autouse=True)
def add_dataset(doctest_namespace):
    columns = ["query_id", "item_id", "timestamp"]
    data = [
        (1, 1, "01-01-2020"),
        (1, 2, "02-01-2020"),
        (1, 3, "03-01-2020"),
        (1, 4, "04-01-2020"),
        (1, 5, "05-01-2020"),
        (2, 1, "06-01-2020"),
        (2, 2, "07-01-2020"),
        (2, 3, "08-01-2020"),
        (2, 9, "09-01-2020"),
        (2, 10, "10-01-2020"),
        (3, 1, "01-01-2020"),
        (3, 5, "02-01-2020"),
        (3, 3, "03-01-2020"),
        (3, 1, "04-01-2020"),
        (3, 2, "05-01-2020"),
    ]
    interactions = pd.DataFrame(data, columns=columns)
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"], format="%d-%m-%Y")
    doctest_namespace["dataset"] = interactions


@pytest.fixture(autouse=True)
def add_recommendations(doctest_namespace):
    recommendations = pd.DataFrame(recs_data, columns=["query_id", "item_id", "rating"])
    doctest_namespace["recommendations"] = recommendations


@pytest.fixture(autouse=True)
def add_groundtruth(doctest_namespace):
    groundtruth = pd.DataFrame(groundtruth_data, columns=["query_id", "item_id"])
    doctest_namespace["groundtruth"] = groundtruth


@pytest.fixture(autouse=True)
def add_category_recommendations(doctest_namespace):
    category_recommendations = pd.DataFrame(recs_data, columns=["query_id", "category_id", "rating"])
    doctest_namespace["category_recommendations"] = category_recommendations


@pytest.fixture(autouse=True)
def add_train(doctest_namespace):
    train = pd.DataFrame(train_data, columns=["query_id", "item_id"])
    doctest_namespace["train"] = train


@pytest.fixture(autouse=True)
def add_base_rec(doctest_namespace):
    base_rec = pd.DataFrame(base_recs_data, columns=["query_id", "item_id", "rating"])
    doctest_namespace["base_rec"] = base_rec
