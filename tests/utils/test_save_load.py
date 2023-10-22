# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, wildcard-import, unused-wildcard-import
from os.path import dirname, join

import pytest
import pandas as pd

import replay
from replay.preprocessing.data_preparator import Indexer
from replay.utils.model_handler import (
    save_indexer,
    load_indexer,
    save_splitter,
    load_splitter,
)
from replay.utils.spark_utils import convert2spark
from replay.splitters import *


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame(
        [(1, 20.0, -3.0, 1), (2, 30.0, 4.0, 0), (3, 40.0, 0.0, 1)]
    ).toDF("user_idx", "age", "mood", "gender")


@pytest.fixture
def df():
    folder = dirname(replay.__file__)
    res = pd.read_csv(
        join(folder, "../experiments/data/ml1m_ratings.dat"),
        sep="\t",
        names=["user_id", "item_id", "relevance", "timestamp"],
    ).head(1000)
    res = convert2spark(res)
    indexer = Indexer()
    indexer.fit(res, res)
    res = indexer.transform(res)
    return res


def test_indexer(df, tmp_path):
    path = (tmp_path / "indexer").resolve()
    indexer = Indexer("user_idx", "item_idx")
    df = convert2spark(df)
    indexer.fit(df, df)
    save_indexer(indexer, path)
    i = load_indexer(path)
    i.inverse_transform(i.transform(df))
    assert i.user_indexer.inputCol == indexer.user_indexer.inputCol


@pytest.mark.parametrize(
    "splitter, init_args",
    [
        (DateSplitter, {"test_start": 0.8}),
        (RandomSplitter, {"test_size": 0.8, "seed": 123}),
        (NewUsersSplitter, {"test_size": 0.8}),
        (ColdUserRandomSplitter, {"test_size": 0.8, "seed": 123}),
        (
            UserSplitter,
            {"item_test_size": 1, "user_test_size": 0.2, "seed": 123},
        ),
    ],
)
def test_splitter(splitter, init_args, df, tmp_path):
    path = (tmp_path / "splitter").resolve()
    splitter = splitter(**init_args)
    save_splitter(splitter, path)
    train, test = splitter.split(df)
    restored_splitter = load_splitter(path)
    for arg_, value_ in init_args.items():
        assert getattr(restored_splitter, arg_) == value_
    new_train, new_test = restored_splitter.split(df)
    assert new_train.count() == train.count()
    assert new_test.count() == test.count()
