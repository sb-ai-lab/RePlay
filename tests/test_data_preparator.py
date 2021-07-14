# pylint: disable-all
from datetime import datetime
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


def test_read_data_wrong_columns_exception(data_preparator):
    with pytest.raises(ValueError):
        data_preparator._read_data(path="", format_type="blabla")


def test_transform_log_empty_dataframe_exception(data_preparator, spark):
    log = spark.createDataFrame(data=[], schema=StructType([]))
    data_preparator._read_data = Mock(return_value=log)
    with pytest.raises(ValueError):
        data_preparator.transform(
            path="",
            format_type="",
            columns_names={"user_id": "", "item_id": ""},
        )


@pytest.mark.parametrize("column_names", [{"user_id": ""}, {"item_id": ""}])
def test_transform_log_required_columns_exception(
    data_preparator, column_names
):
    with pytest.raises(ValueError):
        data_preparator._read_data(
            path="", format_type="", columns_names=column_names
        )


test_data = [
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
        [["1", "1"], ["1", "2"], ["2", "3"]],
        ["user", "item"],
        {"user_id": "user", "item_id": "item"},
    ),
    (
        [["1", "1", 1.0], ["1", "2", 1.0], ["2", "3", None]],
        ["user", "item", "r"],
        {"user_id": "user", "item_id": "item", "relevance": "r"},
    ),
]


@pytest.mark.parametrize("log_data,log_schema,columns_names", test_data)
def test_transform_log_null_column_exception(
    spark, data_preparator, log_data, log_schema, columns_names
):
    print(columns_names)
    log = spark.createDataFrame(data=log_data, schema=log_schema)
    data_preparator._read_data = Mock(return_value=log)
    with pytest.raises(ValueError):
        data_preparator.transform(
            path="", format_type="", columns_names=columns_names
        )


column_names = [
    {"blabla": ""},
    {"timestamp": "", "blabla": ""},
    {"relevance": "", "blabla": ""},
    {"timestamp": "", "blabla": ""},
    {"timestamp": "", "relevance": "", "blabla": ""},
]


@pytest.mark.parametrize("columns_names", column_names)
def test_transform_log_redundant_columns_exception(
    data_preparator, columns_names
):
    # добавим обязательные колонки
    columns_names.update({"user_id": "", "item_id": ""})
    with pytest.raises(ValueError):
        data_preparator.transform(
            path="", format_type="", columns_names=columns_names
        )


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
    "log_data,log_schema,true_log_data,columns_names", big_test
)
def test_transform_log(
    spark, data_preparator, log_data, log_schema, true_log_data, columns_names
):
    log = spark.createDataFrame(data=log_data, schema=log_schema)
    # явно преобразовываем все к стрингам
    for column in log.columns:
        log = log.withColumn(column, sf.col(column).cast(StringType()))

    true_log = spark.createDataFrame(data=true_log_data, schema=LOG_SCHEMA)

    test_log = data_preparator.transform(data=log, columns_names=columns_names)
    sparkDataFrameEqual(true_log, test_log)


timestamp_data = [
    # log_data, log_schema, true_log_data, columns_names
    (
        [
            ["user1", "item1", 32],
            ["user1", "item2", 12],
            ["user2", "item1", 0],
        ],
        ["user", "item", "ts"],
        [
            ["user1", "item1", datetime.fromtimestamp(32), 1.0],
            ["user1", "item2", datetime.fromtimestamp(12), 1.0],
            ["user2", "item1", datetime.fromtimestamp(0), 1.0],
        ],
        {"user_id": "user", "item_id": "item", "timestamp": "ts"},
    ),
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
    ),
]


@pytest.mark.parametrize(
    "log_data, log_schema, true_log_data, columns_names", timestamp_data
)
def test_transform_log_timestamp_column(
    data_preparator, spark, log_data, log_schema, true_log_data, columns_names
):
    log = spark.createDataFrame(data=log_data, schema=log_schema)

    true_log = spark.createDataFrame(data=true_log_data, schema=LOG_SCHEMA)

    test_log = data_preparator.transform(data=log, columns_names=columns_names)
    sparkDataFrameEqual(true_log, test_log)


format_data = [
    # log_data, log_schema, true_log_data, columns_names
    (
        [
            ["u1", "f1", "2019-01-01 10:00:00"],
            ["u1", "f2", "1995-11-01 00:00:00"],
            ["u2", "f1", "2000-03-30 00:00:00"],
        ],
        ["user", "item", "string_time"],
        [
            ["u1", "f1", datetime(2019, 1, 1, 10), 1.0],
            ["u1", "f2", datetime(1995, 11, 1), 1.0],
            ["u2", "f1", datetime(2000, 3, 30), 1.0],
        ],
        {"user_id": "user", "item_id": "item", "timestamp": "string_time"},
    ),
]


@pytest.mark.parametrize(
    "log_data, log_schema, true_log_data, columns_names", format_data
)
def test_transform_log_timestamp_format(
    data_preparator, spark, log_data, log_schema, true_log_data, columns_names
):
    log = spark.createDataFrame(data=log_data, schema=log_schema)
    log.show()
    print(LOG_SCHEMA)
    true_log = spark.createDataFrame(data=true_log_data, schema=LOG_SCHEMA)

    test_log = data_preparator.transform(
        data=log,
        columns_names=columns_names,
        date_format="yyyy-MM-dd HH:mm:ss",
    )
    test_log.show()
    sparkDataFrameEqual(true_log, test_log)


def test_transform_features_empty_dataframe_exception(spark, data_preparator):
    features = spark.createDataFrame(data=[], schema=StructType([]))
    data_preparator._read_data = Mock(return_value=features)
    with pytest.raises(ValueError):
        data_preparator.transform(
            path="", format_type="", columns_names={"user_id": ""}
        )


@pytest.mark.parametrize(
    "columns_names", [{"timestamp": ""}, {"": ""}, {"blabla": ""}]
)
def test_transform_features_required_columns_exception(
    data_preparator, columns_names
):
    with pytest.raises(ValueError):
        data_preparator.transform(
            path="", format_type="", columns_names=columns_names
        )


null_column_data = [
    # feature_data, feature_schema, columns_names
    (
        [["user1", 1], ["user1", 1], ["user2", None]],
        ["user", "feature"],
        {"user_id": "user", "feature": "feature"},
    ),
    (
        [["1", "2019-01-01"], ["2", None], ["3", "2019-01-01"]],
        ["item", "ts"],
        {"item_id": "item", "timestamp": "ts"},
    ),
    (
        [["1", 1, None], ["1", 2, "2019-01-01"], ["2", 3, "2019-01-01"]],
        ["user", "feature", "timestamp"],
        {"user_id": "user", "feature": "feature", "timestamp": "timestamp"},
    ),
    (
        [
            ["1", 1, 100, "2019-01-01"],
            ["1", 2, 100, "2019-01-01"],
            ["2", 3, None, "2019-01-01"],
        ],
        ["user", "f1", "f2", "timestamp"],
        {
            "user_id": "user",
            "feature": ["f1", "f2"],
            "timestamp": "timestamp",
        },
    ),
]


@pytest.mark.parametrize(
    "feature_data, feature_schema, columns_names", null_column_data
)
def test_transform_features_null_column_exception(
    data_preparator, spark, feature_data, feature_schema, columns_names
):
    features = spark.createDataFrame(data=feature_data, schema=feature_schema)
    data_preparator._read_data = Mock(return_value=features)

    with pytest.raises(ValueError):
        data_preparator.transform(
            path="", format_type="", columns_names=columns_names
        )


extra_columns = [
    # columns_names
    ({"item_id": "", "blabla": ""},),
    ({"item_id": "", "timestamp": "", "blabla": ""},),
    ({"user_id": "", "blabla": ""},),
    ({"user_id": "", "timestamp": "", "blabla": ""},),
]


@pytest.mark.parametrize("columns_names", extra_columns)
def test_transform_features_redundant_columns_exception(
    data_preparator, columns_names
):
    with pytest.raises(ValueError):
        data_preparator.transform(
            path="", format_type="", columns_names=columns_names
        )


no_features_data = [
    # columns_names, features_schema
    ({"item_id": "item"}, ["item"]),
    ({"user_id": "user"}, ["user"]),
    ({"item_id": "item", "timestamp": "ts"}, ["item", "ts"]),
    ({"user_id": "user", "timestamp": "ts"}, ["user", "ts"]),
]


@pytest.mark.parametrize("columns_names, features_schema", no_features_data)
def test_transform_features_no_feature_columns_exception(
    spark, data_preparator, columns_names, features_schema
):
    feature_data = [
        ["id1", "2019-01-01"],
        ["id1", "2019-01-01"],
        ["id2", "2019-01-01"],
    ]
    features = spark.createDataFrame(data=feature_data, schema=features_schema)
    features = features.select(features_schema)
    # явно преобразовываем все к стрингам
    for column in features.columns:
        features = features.withColumn(
            column, sf.col(column).cast(StringType())
        )

    data_preparator._read_data = Mock(return_value=features)

    with pytest.raises(ValueError):
        data_preparator.transform(
            path="", format_type="", columns_names=columns_names
        )


transform_data = [
    # feature_data, feature_schema, true_feature_data, columns_names, features_columns
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
    "feature_data, feature_schema, true_feature_data, columns_names, features_columns",
    transform_data,
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


def get_transformed_features(transformer, train, test):
    transformer.fit(train)
    return transformer.transform(test)


def test_cat_features_transformer(item_features):
    transformed = get_transformed_features(
        # при передаче списка можно сделать кодирование и по числовой фиче
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
