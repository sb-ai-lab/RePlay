from os.path import dirname, join

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

import replay
from replay.experimental.preprocessing.data_preparator import Indexer
from replay.experimental.utils.model_handler import load_indexer, save_indexer
from replay.utils.spark_utils import convert2spark


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame([(1, 20.0, -3.0, 1), (2, 30.0, 4.0, 0), (3, 40.0, 0.0, 1)]).toDF(
        "user_idx", "age", "mood", "gender"
    )


@pytest.fixture
def df():
    folder = dirname(replay.__file__)
    res = pd.read_csv(
        join(folder, "../examples/data/ml1m_ratings.dat"),
        sep="\t",
        names=["user_id", "item_id", "relevance", "timestamp"],
    ).head(1000)
    res = convert2spark(res)
    indexer = Indexer()
    indexer.fit(res, res)
    res = indexer.transform(res)
    return res


@pytest.mark.experimental
def test_indexer(df, tmp_path):
    path = (tmp_path / "indexer").resolve()
    indexer = Indexer("user_idx", "item_idx")
    df = convert2spark(df)
    indexer.fit(df, df)
    save_indexer(indexer, path)
    i = load_indexer(path)
    i.inverse_transform(i.transform(df))
    assert i.user_indexer.inputCol == indexer.user_indexer.inputCol
