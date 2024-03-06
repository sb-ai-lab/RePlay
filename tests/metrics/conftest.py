from os.path import dirname, join
from random import shuffle
from typing import Tuple

import pandas as pd
import polars as pl
import pytest

import replay
from replay.models import PopRec
from replay.preprocessing import LabelEncoder, LabelEncodingRule
from replay.splitters import RatioSplitter
from replay.utils import PandasDataFrame, SparkDataFrame
from replay.utils.spark_utils import convert2spark
from tests.utils import create_dataset, spark

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
def predict_pl():
    return pl.DataFrame(recs_data, schema=["uid", "iid", "scores"])


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
def gt_pl():
    return pl.DataFrame(gt_data, schema=["uid", "iid"])


@pytest.fixture(scope="module")
def predict_fake_query_pd():
    return pd.DataFrame(gt_data, columns=["fake_query_id", "iid"])


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
def base_recs_pl():
    return pl.DataFrame(base_recs_data, schema=["uid", "iid", "scores"])


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


def encode_data(data: SparkDataFrame) -> SparkDataFrame:
    encoder = LabelEncoder([LabelEncodingRule("user_idx"), LabelEncodingRule("item_idx")])
    return encoder.fit_transform(data)


def split_data(data: SparkDataFrame) -> Tuple[SparkDataFrame, SparkDataFrame]:
    train, test = RatioSplitter(test_size=0.3, query_column="user_idx", divide_column="user_idx").split(data)
    return train, test


@pytest.fixture(scope="module")
def random_train_test_recs() -> Tuple[PandasDataFrame, PandasDataFrame, PandasDataFrame]:
    folder = dirname(replay.__file__)
    ml_1m = pd.read_csv(
        join(folder, "../examples/data/ml1m_ratings.dat"),
        sep="\t",
        names=["user_idx", "item_idx", "relevance", "timestamp"],
    )
    ml_1m = convert2spark(ml_1m)
    encoded_data = encode_data(ml_1m)
    train, test = split_data(encoded_data)

    model = PopRec()
    model.fit(create_dataset(train))
    recs = model.predict(create_dataset(test), 20)

    return train.toPandas(), test.toPandas(), recs.toPandas()
