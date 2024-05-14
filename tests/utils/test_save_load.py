from os.path import dirname, join

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

import replay
from replay.models import ItemKNN
from replay.preprocessing.label_encoder import LabelEncoder, LabelEncodingRule
from replay.splitters import (
    ColdUserRandomSplitter,
    LastNSplitter,
    NewUsersSplitter,
    RandomSplitter,
    RatioSplitter,
    TimeSplitter,
    TwoStageSplitter,
)
from replay.utils.model_handler import (
    load,
    load_encoder,
    load_splitter,
    save,
    save_encoder,
    save_splitter,
)
from replay.utils.spark_utils import convert2spark
from tests.utils import create_dataset, sparkDataFrameEqual


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
    encoder = LabelEncoder(
        [
            LabelEncodingRule("user_id"),
            LabelEncodingRule("item_id"),
        ]
    )
    res = encoder.fit_transform(res)
    return res


@pytest.mark.spark
@pytest.mark.parametrize(
    "splitter, init_args",
    [
        (TimeSplitter, {"time_threshold": 0.8, "query_column": "user_id"}),
        (LastNSplitter, {"N": 2, "query_column": "user_id", "divide_column": "user_id"}),
        (RatioSplitter, {"test_size": 0.8, "query_column": "user_id", "divide_column": "user_id"}),
        (RandomSplitter, {"test_size": 0.8, "seed": 123}),
        (NewUsersSplitter, {"test_size": 0.8, "query_column": "user_id"}),
        (ColdUserRandomSplitter, {"test_size": 0.8, "seed": 123, "query_column": "user_id"}),
        (
            TwoStageSplitter,
            {
                "second_divide_size": 1,
                "first_divide_size": 0.2,
                "seed": 123,
                "query_column": "user_id",
                "first_divide_column": "user_id",
            },
        ),
    ],
)
def test_splitter(splitter, init_args, df, tmp_path):
    path = (tmp_path / "splitter").resolve()
    splitter = splitter(**init_args)
    df = df.withColumnRenamed("user_idx", "user_id").withColumnRenamed("item_idx", "item_id")
    save_splitter(splitter, path)
    save_splitter(splitter, path, overwrite=True)
    train, test = splitter.split(df)
    restored_splitter = load_splitter(path)
    for arg_, value_ in init_args.items():
        assert getattr(restored_splitter, arg_) == value_
    new_train, new_test = restored_splitter.split(df)
    assert new_train.count() == train.count()
    assert new_test.count() == test.count()


@pytest.mark.spark
def test_save_load_model(long_log_with_features, tmp_path):
    model = ItemKNN()
    path = (tmp_path / "test").resolve()
    dataset = create_dataset(long_log_with_features)
    model.fit(dataset)
    base_pred = model.predict(dataset, 5)
    save(model, path)
    loaded_model = load(path)
    new_pred = loaded_model.predict(dataset, 5)
    sparkDataFrameEqual(base_pred, new_pred)


@pytest.mark.spark
def test_save_raise(long_log_with_features, tmp_path):
    model = ItemKNN()
    path = (tmp_path / "test").resolve()
    dataset = create_dataset(long_log_with_features)
    model.fit(dataset)
    save(model, path)
    with pytest.raises(FileExistsError):
        save(model, path)


@pytest.mark.spark
def test_save_load_encoder(long_log_with_features, tmp_path):
    encoder = LabelEncoder(
        [
            LabelEncodingRule("user_idx"),
            LabelEncodingRule("item_idx"),
        ]
    )
    encoder.fit(long_log_with_features)
    save_encoder(encoder, tmp_path)
    loaded_encoder = load_encoder(tmp_path)

    assert encoder.mapping == loaded_encoder.mapping
