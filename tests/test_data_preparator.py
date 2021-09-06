# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import, too-many-arguments
from datetime import datetime
from copy import deepcopy
from unittest.mock import Mock

import pytest
from pyspark.sql import functions as sf
from pyspark.sql.types import StringType, StructType

from replay.constants import LOG_SCHEMA
from replay.data_preparator import DataPreparator, CatFeaturesTransformer
from tests.utils import (
    item_features,
    long_log_with_features,
    short_log_with_features,
    spark,
    sparkDataFrameEqual,
)


@pytest.fixture
def data_preparator():
    return DataPreparator()


column_names = [
    {"blabla": ""},
    {"timestamp": "", "blabla": ""},
    {"relevance": "", "blabla": ""},
    {"timestamp": "", "blabla": ""},
    {"timestamp": "", "relevance": "", "blabla": ""},
]

logs_with_nones = [
    # log_data, log_schema, columns_names
    (
        [["user1", "item1"], ["user1", "item2"], ["user2", None]],
        ["user", "item"],
        {"user_id": "user", "item_id": "item"},
    ),
    (
        [
            ["1", "1", "2019-01-01"],
            ["1", "2", None],
            ["2", "3", "2019-01-01"],
        ],
        ["user", "item", "ts"],
        {"user_id": "user", "item_id": "item", "timestamp": "ts"},
    ),
    (
        [["1", "1", 1.0], ["1", "2", 1.0], ["2", "3", None]],
        ["user", "item", "r"],
        {"user_id": "user", "item_id": "item", "relevance": "r"},
    ),
]


# checks in _read_data
def test_read_data_invalid_format(data_preparator):
    with pytest.raises(ValueError, match=r"Invalid value of format_type.*"):
        data_preparator._read_data(path="/test_path", format_type="blabla")


# checks in _check_columns
@pytest.mark.parametrize("columns_names", column_names)
def test_transform_log_redundant_columns_exception(
    data_preparator, columns_names
):
    data_preparator._read_data = Mock(return_value=long_log_with_features)
    # adding mandatory log columns
    columns_names_local = deepcopy(columns_names)
    columns_names_local.update({"user_id": "", "item_id": ""})
    with pytest.raises(
        ValueError, match=r"'columns_names' has excess columns.*"
    ):
        data_preparator.transform(
            path="/test_path",
            format_type="table",
            columns_names=columns_names_local,
        )


# checks in _check_dataframe
def test_transform_log_empty_dataframe_exception(data_preparator, spark):
    log = spark.createDataFrame(data=[], schema=StructType([]))
    data_preparator._read_data = Mock(return_value=log)
    with pytest.raises(ValueError, match=r"DataFrame is empty.*"):
        data_preparator.transform(
            path="/test_path",
            format_type="json",
            columns_names={"user_id": "", "item_id": ""},
        )


@pytest.mark.parametrize(
    "col_kwargs",
    [
        {"columns_names": {"user_id": "absent_id"}},
        {
            "columns_names": {"user_id": "user_id"},
            "features_columns": ["absent_col"],
        },
    ],
)
def test_absent_columns(data_preparator, long_log_with_features, col_kwargs):
    data_preparator._read_data = Mock(return_value=long_log_with_features)
    with pytest.raises(
        ValueError, match="feature_columns or columns_names has columns.*"
    ):
        data_preparator.transform(
            path="/test_path", format_type="table", **col_kwargs
        )


@pytest.mark.parametrize(
    "log_data, log_schema, columns_names", logs_with_nones
)
def test_transform_log_null_column_exception(
    spark, data_preparator, log_data, log_schema, columns_names
):
    print(columns_names)
    log = spark.createDataFrame(data=log_data, schema=log_schema)
    data_preparator._read_data = Mock(return_value=log)

    with pytest.raises(ValueError, match=r".* has NULL values"):
        data_preparator.transform(
            path="/test_path",
            format_type="parquet",
            columns_names=columns_names,
        )


# checks in transform
def test_read_data_empty_pass(data_preparator):
    with pytest.raises(
        ValueError, match=r"Either data or path parameters must not be None.*",
    ):
        data_preparator.transform(
            columns_names={"user_id": ""}, path=None, data=None
        )


@pytest.mark.parametrize("columns_names", column_names)
def test_transform_log_no_cols(spark, data_preparator, columns_names):
    log = spark.createDataFrame(
        data=[
            ["user1", "item1", "2019-01-01", 0],
            ["user1", "item2", "2019-01-01", 1],
        ],
        schema=["user_id", "item_id", "timestamp", "relevance"],
    )
    data_preparator._read_data = Mock(return_value=log)
    columns_names_local = deepcopy(columns_names)
    columns_names_local.update({"user_id": "user_id"})
    with pytest.raises(
        ValueError,
        match="Feature DataFrame mappings must contain mapping only for one id.*",
    ):
        data_preparator.transform(
            path="/test_path",
            format_type="table",
            columns_names=columns_names_local,
        )


# checks in feature_columns
def test_transform_log_required_columns_exception(
    data_preparator, long_log_with_features
):
    data_preparator._read_data = Mock(return_value=long_log_with_features)
    with pytest.raises(
        ValueError, match="columns_names have neither 'user_id', nor 'item_id'"
    ):
        data_preparator.transform(
            path="/test_path",
            format_type="json",
            columns_names={"timestamp": "timestamp"},
        )


def test_transform_no_feature_columns(data_preparator, long_log_with_features):
    data_preparator._read_data = Mock(
        return_value=long_log_with_features.select("item_id")
    )
    with pytest.raises(ValueError, match="Feature columns missing"):
        data_preparator.transform(
            path="/test_path",
            format_type="json",
            columns_names={"item_id": "item_id"},
        )


# checks in base_columns
def test_features_columns(data_preparator, long_log_with_features):
    data_preparator._read_data = Mock(return_value=long_log_with_features)
    with pytest.raises(ValueError, match="features are not used"):
        data_preparator.transform(
            path="/test_path",
            format_type="json",
            columns_names={"item_id": "item_id", "user_id": "user_id"},
            features_columns=["timestamp"],
        )


# test transform
big_test = [
    # log_data, log_schema, true_log_data, columns_names
    (
        [["user1", "item1"], ["user1", "item2"], ["user2", "item1"]],
        ["user", "item"],
        [
            ["user1", "item1", datetime(1999, 5, 1), 1.0],
            ["user1", "item2", datetime(1999, 5, 1), 1.0],
            ["user2", "item1", datetime(1999, 5, 1), 1.0],
        ],
        {"user_id": "user", "item_id": "item"},
    ),
    (
        [
            ["u1", "i10", "2045-09-18"],
            ["u2", "12", "1935-12-15"],
            ["u5", "303030", "1989-06-26"],
        ],
        ["user_like", "item_like", "ts"],
        [
            ["u1", "i10", datetime(2045, 9, 18), 1.0],
            ["u2", "12", datetime(1935, 12, 15), 1.0],
            ["u5", "303030", datetime(1989, 6, 26), 1.0],
        ],
        {"user_id": "user_like", "item_id": "item_like", "timestamp": "ts"},
    ),
    (
        [
            ["1010", "4944", "1945-05-25"],
            ["4565", "134232", "2045-11-18"],
            ["56756", "item1", "2019-02-05"],
        ],
        ["a", "b", "c"],
        [
            ["1010", "4944", datetime(1945, 5, 25), 1.0],
            ["4565", "134232", datetime(2045, 11, 18), 1.0],
            ["56756", "item1", datetime(2019, 2, 5), 1.0],
        ],
        {"user_id": "a", "item_id": "b", "timestamp": "c"},
    ),
    (
        [
            ["1945-01-25", 123.0, "12", "ue123"],
            ["2045-07-18", 1.0, "1", "u6788888"],
            ["2019-09-30", 0.001, "item10000", "1222222"],
        ],
        ["d", "r", "i", "u"],
        [
            ["ue123", "12", datetime(1945, 1, 25), 123.0],
            ["u6788888", "1", datetime(2045, 7, 18), 1.0],
            ["1222222", "item10000", datetime(2019, 9, 30), 0.001],
        ],
        {"user_id": "u", "item_id": "i", "timestamp": "d", "relevance": "r"},
    ),
]


@pytest.mark.parametrize(
    "log_data,log_schema,true_log_data,columns_names",
    big_test,
    ids=[
        "no ts and rel",
        "str ts, no rel",
        "no string conversion",
        "all cols given",
    ],
)
def test_transform_log(
    spark, data_preparator, log_data, log_schema, true_log_data, columns_names
):
    log = spark.createDataFrame(data=log_data, schema=log_schema)
    # convert all data to StringType
    for column in log.columns:
        log = log.withColumn(column, sf.col(column).cast(StringType()))

    true_log = spark.createDataFrame(data=true_log_data, schema=LOG_SCHEMA)

    test_log = data_preparator.transform(data=log, columns_names=columns_names)
    sparkDataFrameEqual(true_log, test_log)


timestamp_data = [
    # log_data, log_schema, true_log_data, columns_names, date_format
    (
        [
            ["user1", "item1", 3],
            ["user1", "item2", 2 * 365],
            ["user2", "item1", 365],
        ],
        ["user", "item", "ts"],
        [
            ["user1", "item1", datetime.fromtimestamp(3), 1.0],
            ["user1", "item2", datetime.fromtimestamp(730), 1.0],
            ["user2", "item1", datetime.fromtimestamp(365), 1.0],
        ],
        {"user_id": "user", "item_id": "item", "timestamp": "ts"},
        None,
    ),
    (
        [
            ["user1", "item1", "2019$01$01"],
            ["user1", "item2", "1995$11$01 00:00:00"],
            ["user2", "item1", "2000$03$30 13:00:00"],
        ],
        ["user", "item", "string_time"],
        [
            ["user1", "item1", datetime(2019, 1, 1,), 1.0,],
            ["user1", "item2", datetime(1995, 11, 1), 1.0],
            ["user2", "item1", datetime(2000, 3, 30, 13), 1.0],
        ],
        {"user_id": "user", "item_id": "item", "timestamp": "string_time"},
        "yyyy$MM$dd[ HH:mm:ss]",
    ),
]


@pytest.mark.parametrize(
    "log_data, log_schema, true_log_data, columns_names, date_format",
    timestamp_data,
)
def test_transform_log_timestamp_column(
    data_preparator,
    spark,
    log_data,
    log_schema,
    true_log_data,
    columns_names,
    date_format,
):
    log = spark.createDataFrame(data=log_data, schema=log_schema)
    true_log = spark.createDataFrame(data=true_log_data, schema=LOG_SCHEMA)
    test_log = data_preparator.transform(
        data=log, columns_names=columns_names, date_format=date_format
    )
    sparkDataFrameEqual(true_log, test_log)


transform_features_data = [
    # feature_data, feature_schema, true_feature_data,
    # columns_names, features_columns
    (
        [["user1", "feature1"], ["user1", "feature2"], ["user2", "feature1"]],
        ["user", "f0"],
        [["user1", "feature1"], ["user1", "feature2"], ["user2", "feature1"]],
        {"user_id": "user"},
        "f0",
    ),
    (
        [
            ["u1", "f1", "2019-01-01", 1],
            ["u1", "f2", "2019-01-01", 2],
            ["u2", "f1", "2019-01-01", 3],
        ],
        ["user", "f0", "f1", "f2"],
        [
            ["u1", "f1", "2019-01-01", 1],
            ["u1", "f2", "2019-01-01", 2],
            ["u2", "f1", "2019-01-01", 3],
        ],
        {"user_id": "user"},
        ["f0", "f1", "f2"],
    ),
]


@pytest.mark.parametrize(
    "feature_data, feature_schema, true_feature_data, "
    "columns_names, features_columns",
    transform_features_data,
)
def test_transform_features(
    spark,
    data_preparator,
    feature_data,
    feature_schema,
    true_feature_data,
    columns_names,
    features_columns,
):
    features = spark.createDataFrame(data=feature_data, schema=feature_schema)

    if "timestamp" in columns_names:
        schema = ["user_id", "timestamp"] + [
            f"f{i}" for i in range(len(true_feature_data[0]) - 2)
        ]
    else:
        schema = ["user_id"] + [
            f"f{i}" for i in range(len(true_feature_data[0]) - 1)
        ]

    true_features = spark.createDataFrame(
        data=true_feature_data, schema=schema
    )
    true_features = true_features.withColumn(
        "user_id", sf.col("user_id").cast(StringType())
    )
    test_features = data_preparator.transform(
        data=features,
        columns_names=columns_names,
        features_columns=features_columns,
    )
    sparkDataFrameEqual(true_features, test_features)


# categorical features transformer tests
def get_transformed_features(transformer, train, test):
    transformer.fit(train)
    return transformer.transform(test)


def test_cat_features_transformer(item_features):
    transformed = get_transformed_features(
        transformer=CatFeaturesTransformer(cat_cols_list=["class"]),
        train=item_features.filter(sf.col("class") != "dog"),
        test=item_features,
    )
    assert "class" not in transformed.columns
    assert "iq" in transformed.columns and "color" in transformed.columns
    assert (
        "ohe_class_dog" not in transformed.columns
        and "ohe_class_cat" in transformed.columns
    )
    assert (
        transformed.filter(sf.col("item_id") == "i6")
        .select("ohe_class_mouse")
        .collect()[0][0]
        == 1.0
    )


def test_cat_features_transformer_date(
    long_log_with_features, short_log_with_features,
):
    transformed = get_transformed_features(
        transformer=CatFeaturesTransformer(["timestamp"]),
        train=long_log_with_features,
        test=short_log_with_features,
    )
    assert (
        "ohe_timestamp_20190101000000" in transformed.columns
        and "item_id" in transformed.columns
    )


def test_cat_features_transformer_empty_list(
    long_log_with_features, short_log_with_features,
):
    transformed = get_transformed_features(
        transformer=CatFeaturesTransformer([]),
        train=long_log_with_features,
        test=short_log_with_features,
    )
    assert len(transformed.columns) == 4
    assert "timestamp" in transformed.columns
