# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
import logging
import pytest
import pandas as pd
from pyspark.sql import functions as sf
from pyspark.sql.types import TimestampType, StringType

from replay.experimental.preprocessing.data_preparator import (
    DataPreparator,
    CatFeaturesTransformer,
    Indexer,
)
from replay.utils.spark_utils import convert2spark
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


@pytest.fixture
def mapping():
    return {
        "user_id": "user_idx",
        "item_id": "item_idx",
        "timestamp": "timestamp",
        "relevance": "relevance",
    }


# checks in read_as_spark_df
def test_read_data_invalid_format(data_preparator):
    with pytest.raises(ValueError, match=r"Invalid value of format_type.*"):
        data_preparator.read_as_spark_df(
            path="/test_path", format_type="blabla"
        )

    with pytest.raises(
        ValueError, match="Either data or path parameters must not be None"
    ):
        data_preparator.read_as_spark_df(format_type="csv")


# errors in check_df
def test_check_df_errors(data_preparator, long_log_with_features, mapping):
    with pytest.raises(ValueError, match="DataFrame is empty"):
        data_preparator.check_df(
            dataframe=long_log_with_features.filter(sf.col("user_idx") > 10),
            columns_mapping=mapping,
        )

    with pytest.raises(
        ValueError,
        match="Column `relevance` stated in mapping is absent in dataframe",
    ):
        col_map = mapping
        data_preparator.check_df(
            dataframe=long_log_with_features.drop("relevance"),
            columns_mapping=col_map,
        )


# logging in check_df
def test_read_check_df_logger_msg(
    data_preparator, long_log_with_features, mapping, caplog
):
    with caplog.at_level(logging.INFO):
        mapping.pop("timestamp")
        data_preparator.check_df(
            dataframe=long_log_with_features.withColumn(
                "relevance",
                sf.when(sf.col("user_idx") == 1, None).otherwise(
                    sf.col("relevance").cast(StringType())
                ),
            ).drop("timestamp"),
            columns_mapping=mapping,
        )
        assert (
            "Column `relevance` has NULL values. "
            "Handle NULL values before the next data preprocessing/model training steps"
            in caplog.text
        )

        assert (
            "Columns ['timestamp'] are absent, but may be required for models training. "
            in caplog.text
        )

        assert (
            "Relevance column `relevance` should be numeric, but it is StringType"
            in caplog.text
        )


def test_generate_cols(data_preparator, long_log_with_features, mapping):
    mapping.pop("timestamp")
    df = data_preparator.add_absent_log_cols(
        dataframe=long_log_with_features.drop("timestamp"),
        columns_mapping=mapping,
    )
    assert "timestamp" in df.columns
    assert isinstance(df.schema["timestamp"].dataType, TimestampType)


def test_indexer(long_log_with_features):
    indexer = Indexer()
    df = long_log_with_features.withColumnRenamed("user_idx", "user_id")
    df = df.withColumnRenamed("item_idx", "item_id")
    indexer.fit(df, df)
    res = indexer.transform(df)
    log = indexer.inverse_transform(res)
    sparkDataFrameEqual(log, df)


def test_indexer_without_renaming():
    indexer = Indexer("user_idx", "item_idx")
    df = pd.DataFrame({"user_idx": [3], "item_idx": [5]})
    df = convert2spark(df)
    indexer.fit(df, df)
    res = indexer.transform(df)
    cols = res.columns
    assert "user_idx" in cols and "item_idx" in cols
    assert res.toPandas().iloc[0].user_idx == 0
    df_conv = indexer.inverse_transform(res)
    sparkDataFrameEqual(df, df_conv)


def test_indexer_new_dataset(long_log_with_features, short_log_with_features):
    indexer = Indexer("user_idx", "item_idx")
    indexer.fit(
        long_log_with_features.select("user_idx").distinct(),
        long_log_with_features.select("item_idx").distinct(),
    )
    res = indexer.transform(short_log_with_features)
    assert "user_idx" in res.columns and "item_idx" in res.columns


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
        transformed.filter(sf.col("item_idx") == 5)
        .select("ohe_class_mouse")
        .collect()[0][0]
        == 1.0
    )


def test_cat_features_transformer_date(
    long_log_with_features,
    short_log_with_features,
):
    transformed = get_transformed_features(
        transformer=CatFeaturesTransformer(["timestamp"]),
        train=long_log_with_features,
        test=short_log_with_features,
    )
    assert (
        "ohe_timestamp_20190101000000" in transformed.columns
        and "item_idx" in transformed.columns
    )


def test_cat_features_transformer_empty_list(
    long_log_with_features,
    short_log_with_features,
):
    transformed = get_transformed_features(
        transformer=CatFeaturesTransformer([]),
        train=long_log_with_features,
        test=short_log_with_features,
    )
    assert len(transformed.columns) == 4
    assert "timestamp" in transformed.columns
