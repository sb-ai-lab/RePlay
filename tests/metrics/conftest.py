from random import shuffle

import pandas as pd
import pytest
from tests.utils import spark

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

gt_data = [
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

base_recs_data = [
    (1, 3, 0.5),
    (1, 7, 0.5),
    (1, 10, 0.3),
    (1, 11, 0.3),
    (1, 2, 0.7),
    (2, 5, 0.6),
    (2, 8, 0.6),
    (2, 11, 0.4),
    (2, 1, 0.3),
    (2, 3, 0.3),
    (3, 4, 1.0),
    (3, 9, 0.5),
    (3, 2, 0.3),
]


@pytest.mark.usefixtures("spark")
@pytest.fixture()
def predict_spark(spark):
    return spark.createDataFrame(recs_data, schema=["uid", "iid", "scores"])


@pytest.fixture(scope="module")
def predict_pd():
    return pd.DataFrame(recs_data, columns=["uid", "iid", "scores"])


@pytest.fixture(scope="module")
def predict_sorted_dict():
    converted_dict = {}
    for user, item, score in recs_data:
        converted_dict.setdefault(user, [])
        converted_dict[user].append((item, score))
    for user, items in converted_dict.items():
        items = sorted(items, key=lambda x: x[1], reverse=True)
        converted_dict[user] = items
    return converted_dict


@pytest.fixture(scope="module")
@pytest.mark.usefixtures("predict_sorted_dict")
def predict_unsorted_dict(predict_sorted_dict):
    converted_dict = {}
    for user, items in predict_sorted_dict.items():
        shuffle(items)
        converted_dict[user] = items
    return converted_dict


@pytest.fixture(scope="module")
def fake_train_dict():
    converted_dict = {}
    for user, item, _ in recs_data:
        converted_dict.setdefault(user, [])
        converted_dict[user].append(item)
    return converted_dict


@pytest.mark.usefixtures("spark")
@pytest.fixture()
def gt_spark(spark):
    return spark.createDataFrame(gt_data, schema=["uid", "iid"])


@pytest.fixture(scope="module")
def gt_pd():
    return pd.DataFrame(gt_data, columns=["uid", "iid"])


@pytest.fixture(scope="module")
def gt_dict():
    converted_dict = {}
    for user, item in gt_data:
        converted_dict.setdefault(user, [])
        converted_dict[user].append(item)
    return converted_dict


@pytest.mark.usefixtures("spark")
@pytest.fixture()
def base_recs_spark(spark):
    return spark.createDataFrame(
        base_recs_data, schema=["uid", "iid", "scores"]
    )


@pytest.fixture(scope="module")
def base_recs_pd():
    return pd.DataFrame(base_recs_data, columns=["uid", "iid", "scores"])


@pytest.fixture(scope="module")
def base_recs_dict():
    converted_dict = {}
    for user, item, score in base_recs_data:
        converted_dict.setdefault(user, [])
        converted_dict[user].append((item, score))
    for user, items in converted_dict.items():
        items = sorted(items, key=lambda x: x[1], reverse=True)
        converted_dict[user] = items
    return converted_dict
