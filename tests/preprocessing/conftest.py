import random

import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.utils import PYSPARK_AVAILABLE

if PYSPARK_AVAILABLE:
    from pyspark.sql.functions import col, to_date, unix_timestamp
    from pyspark.sql.types import ArrayType, IntegerType, LongType, StringType, StructField, StructType


@pytest.fixture(scope="module")
def pandas_df_for_labelencoder():
    return pd.DataFrame({"user_id": ["u1", "u2"], "item1": ["item_1", "item_2"], "item2": ["item_1", "item_2"]})


@pytest.fixture(scope="module")
def pandas_df_for_labelencoder_modified():
    return pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3"],
            "item1": ["item_1", "item_2", "item_3"],
            "item2": ["item_1", "item_2", "item_3"],
        }
    )


@pytest.fixture(scope="module")
def pandas_df_for_labelencoder_new_data():
    return pd.DataFrame({"user_id": ["u4"], "item1": ["item_4"], "item2": ["item_4"]})


@pytest.fixture(scope="module")
def pandas_df_for_grouped_labelencoder_new_data():
    return pd.DataFrame({"user_id": [["u4", "u5"]], "item1": [["item_4", "item_5"]], "item2": [["item_4", "item_5"]]})


@pytest.fixture(scope="module")
def pandas_df_for_grouped_labelencoder():
    return pd.DataFrame(
        {
            "user_id": [["u1", "u2"], ["u2", "u1"]],
            "item1": [["item_2", "item_1"], ["item_2", "item_2"]],
            "item2": [["item_1", "item_1"], ["item_1", "item_2"]],
        }
    )


@pytest.fixture(scope="module")
def pandas_df_for_grouped_labelencoder_modified():
    return pd.DataFrame(
        {
            "user_id": [["u1", "u2"], ["u2", "u3"]],
            "item1": [["item_2", "item_3"], ["item_2", "item_1"]],
            "item2": [["item_2", "item_1"], ["item_3", "item_3"]],
        }
    )


@pytest.fixture(scope="module")
def polars_df_for_labelencoder(pandas_df_for_labelencoder):
    return pl.from_pandas(pandas_df_for_labelencoder)


@pytest.fixture(scope="module")
def spark_df_for_labelencoder(spark, pandas_df_for_labelencoder):
    return spark.createDataFrame(pandas_df_for_labelencoder)


@pytest.fixture(scope="module")
def polars_df_for_labelencoder_modified(pandas_df_for_labelencoder_modified):
    return pl.from_pandas(pandas_df_for_labelencoder_modified)


@pytest.fixture(scope="module")
def spark_df_for_labelencoder_modified(spark, pandas_df_for_labelencoder_modified):
    return spark.createDataFrame(pandas_df_for_labelencoder_modified)


@pytest.fixture(scope="module")
def polars_df_for_labelencoder_new_data(pandas_df_for_labelencoder_new_data):
    return pl.from_pandas(pandas_df_for_labelencoder_new_data)


@pytest.fixture(scope="module")
def polars_df_for_grouped_labelencoder_new_data(pandas_df_for_grouped_labelencoder_new_data):
    return pl.from_pandas(pandas_df_for_grouped_labelencoder_new_data)


@pytest.fixture(scope="module")
def polars_df_for_grouped_labelencoder(pandas_df_for_grouped_labelencoder):
    return pl.from_pandas(pandas_df_for_grouped_labelencoder)


@pytest.fixture(scope="module")
def spark_df_for_grouped_labelencoder(spark, pandas_df_for_grouped_labelencoder):
    return spark.createDataFrame(pandas_df_for_grouped_labelencoder)


@pytest.fixture(scope="module")
def polars_df_for_grouped_labelencoder_modified(pandas_df_for_grouped_labelencoder_modified):
    return pl.from_pandas(pandas_df_for_grouped_labelencoder_modified)


@pytest.fixture(scope="module")
def spark_df_for_grouped_labelencoder_modified(spark, pandas_df_for_grouped_labelencoder_modified):
    return spark.createDataFrame(pandas_df_for_grouped_labelencoder_modified)


@pytest.fixture(scope="module")
def spark_df_for_labelencoder_new_data(pandas_df_for_labelencoder_new_data, spark):
    return spark.createDataFrame(pandas_df_for_labelencoder_new_data)


@pytest.fixture(scope="module")
def spark_df_for_grouped_labelencoder_new_data(pandas_df_for_grouped_labelencoder_new_data, spark):
    return spark.createDataFrame(pandas_df_for_grouped_labelencoder_new_data)


@pytest.fixture(scope="module")
def schema():
    return StructType(
        [
            StructField("user_id", LongType(), False),
            StructField("item_id", ArrayType(LongType()), False),
            StructField("timestamp", ArrayType(LongType()), False),
        ]
    )


@pytest.fixture(scope="module")
def schema_string():
    return StructType(
        [
            StructField("user_id", LongType(), False),
            StructField("item_id", ArrayType(StringType()), False),
            StructField("timestamp", ArrayType(LongType()), False),
        ]
    )


@pytest.fixture(scope="module")
def dataframe(spark, schema):
    data = [
        (1, [2], [19842]),
        (1, [2, 4], [19842, 19844]),
        (1, [2, 4, 3], [19842, 19844, 19843]),
        (1, [2, 4, 3, 5], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [1], [19841]),
        (2, [1, 2], [19841, 19842]),
        (2, [1, 2, 3], [19841, 19842, 19843]),
        (2, [1, 2, 3, 4], [19841, 19842, 19843, 19844]),
        (3, [10], [19844]),
        (4, [10, 11], [19844, 19843]),
        (4, [10, 11, 12], [19844, 19843, 19845]),
        (10, [1], [19841]),
    ]
    return spark.createDataFrame(data, schema=schema)


@pytest.fixture(scope="module")
def dataframe_pandas():
    data = [
        (1, [2], [19842]),
        (1, [2, 4], [19842, 19844]),
        (1, [2, 4, 3], [19842, 19844, 19843]),
        (1, [2, 4, 3, 5], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [1], [19841]),
        (2, [1, 2], [19841, 19842]),
        (2, [1, 2, 3], [19841, 19842, 19843]),
        (2, [1, 2, 3, 4], [19841, 19842, 19843, 19844]),
        (3, [10], [19844]),
        (4, [10, 11], [19844, 19843]),
        (4, [10, 11, 12], [19844, 19843, 19845]),
        (10, [1], [19841]),
    ]
    return pd.DataFrame(data, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_special(spark, schema):
    data_special = [
        (1, [2], [19842]),
        (1, [2, 4], [19842, 19844]),
        (1, [2, 4, 3], [19842, 19844, 19843]),
        (1, [2, 4, 3, 5], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, [1], [19841]),
        (2, [1, 2], [19841, 19842]),
        (2, [1, 2, 3], [19841, 19842, 19843]),
        (2, [1, 2, 3, 4], [19841, 19842, 19843, 19844]),
        (3, [10], [19844]),
        (4, [10, 11], [19844, 19843]),
        (4, [10, 11, 12], [19844, 19843, 19845]),
        (10, [1], [19841]),
    ]
    return spark.createDataFrame(data_special, schema=schema)


@pytest.fixture(scope="module")
def dataframe_special_pandas():
    data_special = [
        (1, [2], [19842]),
        (1, [2, 4], [19842, 19844]),
        (1, [2, 4, 3], [19842, 19844, 19843]),
        (1, [2, 4, 3, 5], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, [1], [19841]),
        (2, [1, 2], [19841, 19842]),
        (2, [1, 2, 3], [19841, 19842, 19843]),
        (2, [1, 2, 3, 4], [19841, 19842, 19843, 19844]),
        (3, [10], [19844]),
        (4, [10, 11], [19844, 19843]),
        (4, [10, 11, 12], [19844, 19843, 19845]),
        (10, [1], [19841]),
    ]
    return pd.DataFrame(data_special, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_only_item(spark, schema):
    data_only_item = [
        (1, [2, 0, 0, 0, 0], [19842]),
        (1, [2, 4, 0, 0, 0], [19842, 19844]),
        (1, [2, 4, 3, 0, 0], [19842, 19844, 19843]),
        (1, [2, 4, 3, 5, 0], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0], [19841]),
        (2, [1, 2, 0, 0, 0], [19841, 19842]),
        (2, [1, 2, 3, 0, 0], [19841, 19842, 19843]),
        (2, [1, 2, 3, 4, 0], [19841, 19842, 19843, 19844]),
        (3, [10, 0, 0, 0, 0], [19844]),
        (4, [10, 11, 0, 0, 0], [19844, 19843]),
        (4, [10, 11, 12, 0, 0], [19844, 19843, 19845]),
        (10, [1, 0, 0, 0, 0], [19841]),
    ]
    return spark.createDataFrame(data_only_item, schema=schema)


@pytest.fixture(scope="module")
def dataframe_only_item_pandas():
    data_only_item = [
        (1, [2, 0, 0, 0, 0], [19842]),
        (1, [2, 4, 0, 0, 0], [19842, 19844]),
        (1, [2, 4, 3, 0, 0], [19842, 19844, 19843]),
        (1, [2, 4, 3, 5, 0], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0], [19841]),
        (2, [1, 2, 0, 0, 0], [19841, 19842]),
        (2, [1, 2, 3, 0, 0], [19841, 19842, 19843]),
        (2, [1, 2, 3, 4, 0], [19841, 19842, 19843, 19844]),
        (3, [10, 0, 0, 0, 0], [19844]),
        (4, [10, 11, 0, 0, 0], [19844, 19843]),
        (4, [10, 11, 12, 0, 0], [19844, 19843, 19845]),
        (10, [1, 0, 0, 0, 0], [19841]),
    ]
    return pd.DataFrame(data_only_item, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_only_item_none(spark, schema):
    data_only_item_none = [
        (1, [2, 0, 0, 0, 0, 0], [19842]),
        (1, [2, 4, 0, 0, 0, 0], [19842, 19844]),
        (1, [2, 4, 3, 0, 0, 0], [19842, 19844, 19843]),
        (1, [2, 4, 3, 5, 0, 0], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6, 0], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0, 0], [19841]),
        (2, [1, 2, 0, 0, 0, 0], [19841, 19842]),
        (2, [1, 2, 3, 0, 0, 0], [19841, 19842, 19843]),
        (2, [1, 2, 3, 4, 0, 0], [19841, 19842, 19843, 19844]),
        (3, [10, 0, 0, 0, 0, 0], [19844]),
        (4, [10, 11, 0, 0, 0, 0], [19844, 19843]),
        (4, [10, 11, 12, 0, 0, 0], [19844, 19843, 19845]),
        (10, [1, 0, 0, 0, 0, 0], [19841]),
    ]
    return spark.createDataFrame(data_only_item_none, schema=schema)


@pytest.fixture(scope="module")
def dataframe_only_item_none_pandas():
    data_only_item_none = [
        (1, [2, 0, 0, 0, 0, 0], [19842]),
        (1, [2, 4, 0, 0, 0, 0], [19842, 19844]),
        (1, [2, 4, 3, 0, 0, 0], [19842, 19844, 19843]),
        (1, [2, 4, 3, 5, 0, 0], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6, 0], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0, 0], [19841]),
        (2, [1, 2, 0, 0, 0, 0], [19841, 19842]),
        (2, [1, 2, 3, 0, 0, 0], [19841, 19842, 19843]),
        (2, [1, 2, 3, 4, 0, 0], [19841, 19842, 19843, 19844]),
        (3, [10, 0, 0, 0, 0, 0], [19844]),
        (4, [10, 11, 0, 0, 0, 0], [19844, 19843]),
        (4, [10, 11, 12, 0, 0, 0], [19844, 19843, 19845]),
        (10, [1, 0, 0, 0, 0, 0], [19841]),
    ]
    return pd.DataFrame(data_only_item_none, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_two_columns_none(spark, schema):
    data_two_columns_none = [
        (1, [2, 0, 0, 0, 0, 0], [19842, 0, 0, 0, 0]),
        (1, [2, 4, 0, 0, 0, 0], [19842, 19844, 0, 0, 0]),
        (1, [2, 4, 3, 0, 0, 0], [19842, 19844, 19843, 0, 0]),
        (1, [2, 4, 3, 5, 0, 0], [19842, 19844, 19843, 19845, 0]),
        (1, [2, 4, 3, 5, 6, 0], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0, 0], [19841, 0, 0, 0, 0]),
        (2, [1, 2, 0, 0, 0, 0], [19841, 19842, 0, 0, 0]),
        (2, [1, 2, 3, 0, 0, 0], [19841, 19842, 19843, 0, 0]),
        (2, [1, 2, 3, 4, 0, 0], [19841, 19842, 19843, 19844, 0]),
        (3, [10, 0, 0, 0, 0, 0], [19844, 0, 0, 0, 0]),
        (4, [10, 11, 0, 0, 0, 0], [19844, 19843, 0, 0, 0]),
        (4, [10, 11, 12, 0, 0, 0], [19844, 19843, 19845, 0, 0]),
        (10, [1, 0, 0, 0, 0, 0], [19841, 0, 0, 0, 0]),
    ]
    return spark.createDataFrame(data_two_columns_none, schema=schema)


@pytest.fixture(scope="module")
def dataframe_two_columns_none_pandas():
    data_two_columns_none = [
        (1, [2, 0, 0, 0, 0, 0], [19842, 0, 0, 0, 0]),
        (1, [2, 4, 0, 0, 0, 0], [19842, 19844, 0, 0, 0]),
        (1, [2, 4, 3, 0, 0, 0], [19842, 19844, 19843, 0, 0]),
        (1, [2, 4, 3, 5, 0, 0], [19842, 19844, 19843, 19845, 0]),
        (1, [2, 4, 3, 5, 6, 0], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0, 0], [19841, 0, 0, 0, 0]),
        (2, [1, 2, 0, 0, 0, 0], [19841, 19842, 0, 0, 0]),
        (2, [1, 2, 3, 0, 0, 0], [19841, 19842, 19843, 0, 0]),
        (2, [1, 2, 3, 4, 0, 0], [19841, 19842, 19843, 19844, 0]),
        (3, [10, 0, 0, 0, 0, 0], [19844, 0, 0, 0, 0]),
        (4, [10, 11, 0, 0, 0, 0], [19844, 19843, 0, 0, 0]),
        (4, [10, 11, 12, 0, 0, 0], [19844, 19843, 19845, 0, 0]),
        (10, [1, 0, 0, 0, 0, 0], [19841, 0, 0, 0, 0]),
    ]
    return pd.DataFrame(data_two_columns_none, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_two_columns(spark, schema):
    data_two_columns = [
        (1, [2, 0, 0, 0, 0], [19842, -1, -1, -1, -1]),
        (1, [2, 4, 0, 0, 0], [19842, 19844, -1, -1, -1]),
        (1, [2, 4, 3, 0, 0], [19842, 19844, 19843, -1, -1]),
        (1, [2, 4, 3, 5, 0], [19842, 19844, 19843, 19845, -1]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
        (2, [1, 2, 0, 0, 0], [19841, 19842, -1, -1, -1]),
        (2, [1, 2, 3, 0, 0], [19841, 19842, 19843, -1, -1]),
        (2, [1, 2, 3, 4, 0], [19841, 19842, 19843, 19844, -1]),
        (3, [10, 0, 0, 0, 0], [19844, -1, -1, -1, -1]),
        (4, [10, 11, 0, 0, 0], [19844, 19843, -1, -1, -1]),
        (4, [10, 11, 12, 0, 0], [19844, 19843, 19845, -1, -1]),
        (10, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
    ]
    return spark.createDataFrame(data_two_columns, schema=schema)


@pytest.fixture(scope="module")
def dataframe_two_columns_pandas():
    data_two_columns = [
        (1, [2, 0, 0, 0, 0], [19842, -1, -1, -1, -1]),
        (1, [2, 4, 0, 0, 0], [19842, 19844, -1, -1, -1]),
        (1, [2, 4, 3, 0, 0], [19842, 19844, 19843, -1, -1]),
        (1, [2, 4, 3, 5, 0], [19842, 19844, 19843, 19845, -1]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
        (2, [1, 2, 0, 0, 0], [19841, 19842, -1, -1, -1]),
        (2, [1, 2, 3, 0, 0], [19841, 19842, 19843, -1, -1]),
        (2, [1, 2, 3, 4, 0], [19841, 19842, 19843, 19844, -1]),
        (3, [10, 0, 0, 0, 0], [19844, -1, -1, -1, -1]),
        (4, [10, 11, 0, 0, 0], [19844, 19843, -1, -1, -1]),
        (4, [10, 11, 12, 0, 0], [19844, 19843, 19845, -1, -1]),
        (10, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
    ]
    return pd.DataFrame(data_two_columns, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_two_columns_len_two(spark, schema):
    data_two_columns_len_two = [
        (1, [2, 0], [19842, -1]),
        (1, [2, 4], [19842, 19844]),
        (1, [4, 3], [19844, 19843]),
        (1, [3, 5], [19843, 19845]),
        (1, [5, 6], [19845, 19846]),
        (1, [6, 7], [19846, 19847]),
        (2, [1, 0], [19841, -1]),
        (2, [1, 2], [19841, 19842]),
        (2, [2, 3], [19842, 19843]),
        (2, [3, 4], [19843, 19844]),
        (3, [10, 0], [19844, -1]),
        (4, [10, 11], [19844, 19843]),
        (4, [11, 12], [19843, 19845]),
        (10, [1, 0], [19841, -1]),
    ]
    return spark.createDataFrame(data_two_columns_len_two, schema=schema)


@pytest.fixture(scope="module")
def dataframe_two_columns_len_two_pandas():
    data_two_columns_len_two = [
        (1, [2, 0], [19842, -1]),
        (1, [2, 4], [19842, 19844]),
        (1, [4, 3], [19844, 19843]),
        (1, [3, 5], [19843, 19845]),
        (1, [5, 6], [19845, 19846]),
        (1, [6, 7], [19846, 19847]),
        (2, [1, 0], [19841, -1]),
        (2, [1, 2], [19841, 19842]),
        (2, [2, 3], [19842, 19843]),
        (2, [3, 4], [19843, 19844]),
        (3, [10, 0], [19844, -1]),
        (4, [10, 11], [19844, 19843]),
        (4, [11, 12], [19843, 19845]),
        (10, [1, 0], [19841, -1]),
    ]
    return pd.DataFrame(data_two_columns_len_two, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_two_columns_zeros(spark, schema):
    data_two_columns_zeros = [
        (1, [2, 0, 0, 0, 0], [19842, 0, 0, 0, 0]),
        (1, [2, 4, 0, 0, 0], [19842, 19844, 0, 0, 0]),
        (1, [2, 4, 3, 0, 0], [19842, 19844, 19843, 0, 0]),
        (1, [2, 4, 3, 5, 0], [19842, 19844, 19843, 19845, 0]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0], [19841, 0, 0, 0, 0]),
        (2, [1, 2, 0, 0, 0], [19841, 19842, 0, 0, 0]),
        (2, [1, 2, 3, 0, 0], [19841, 19842, 19843, 0, 0]),
        (2, [1, 2, 3, 4, 0], [19841, 19842, 19843, 19844, 0]),
        (3, [10, 0, 0, 0, 0], [19844, 0, 0, 0, 0]),
        (4, [10, 11, 0, 0, 0], [19844, 19843, 0, 0, 0]),
        (4, [10, 11, 12, 0, 0], [19844, 19843, 19845, 0, 0]),
        (10, [1, 0, 0, 0, 0], [19841, 0, 0, 0, 0]),
    ]
    return spark.createDataFrame(data_two_columns_zeros, schema=schema)


@pytest.fixture(scope="module")
def dataframe_two_columns_zeros_pandas():
    data_two_columns_zeros = [
        (1, [2, 0, 0, 0, 0], [19842, 0, 0, 0, 0]),
        (1, [2, 4, 0, 0, 0], [19842, 19844, 0, 0, 0]),
        (1, [2, 4, 3, 0, 0], [19842, 19844, 19843, 0, 0]),
        (1, [2, 4, 3, 5, 0], [19842, 19844, 19843, 19845, 0]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0], [19841, 0, 0, 0, 0]),
        (2, [1, 2, 0, 0, 0], [19841, 19842, 0, 0, 0]),
        (2, [1, 2, 3, 0, 0], [19841, 19842, 19843, 0, 0]),
        (2, [1, 2, 3, 4, 0], [19841, 19842, 19843, 19844, 0]),
        (3, [10, 0, 0, 0, 0], [19844, 0, 0, 0, 0]),
        (4, [10, 11, 0, 0, 0], [19844, 19843, 0, 0, 0]),
        (4, [10, 11, 12, 0, 0], [19844, 19843, 19845, 0, 0]),
        (10, [1, 0, 0, 0, 0], [19841, 0, 0, 0, 0]),
    ]
    return pd.DataFrame(data_two_columns_zeros, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_two_columns_cut_left(spark, schema):
    data_two_columns_cut_left = [
        (1, [2, 0, 0, 0, 0], [19842, -1, -1, -1, -1]),
        (1, [2, 4, 0, 0, 0], [19842, 19844, -1, -1, -1]),
        (1, [2, 4, 3, 0, 0], [19842, 19844, 19843, -1, -1]),
        (1, [2, 4, 3, 5, 0], [19842, 19844, 19843, 19845, -1]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (2, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
        (2, [1, 2, 0, 0, 0], [19841, 19842, -1, -1, -1]),
        (2, [1, 2, 3, 0, 0], [19841, 19842, 19843, -1, -1]),
        (2, [1, 2, 3, 4, 0], [19841, 19842, 19843, 19844, -1]),
        (3, [10, 0, 0, 0, 0], [19844, -1, -1, -1, -1]),
        (4, [10, 11, 0, 0, 0], [19844, 19843, -1, -1, -1]),
        (4, [10, 11, 12, 0, 0], [19844, 19843, 19845, -1, -1]),
        (10, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
    ]
    return spark.createDataFrame(data_two_columns_cut_left, schema=schema)


@pytest.fixture(scope="module")
def dataframe_two_columns_cut_left_pandas():
    data_two_columns_cut_left = [
        (1, [2, 0, 0, 0, 0], [19842, -1, -1, -1, -1]),
        (1, [2, 4, 0, 0, 0], [19842, 19844, -1, -1, -1]),
        (1, [2, 4, 3, 0, 0], [19842, 19844, 19843, -1, -1]),
        (1, [2, 4, 3, 5, 0], [19842, 19844, 19843, 19845, -1]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (2, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
        (2, [1, 2, 0, 0, 0], [19841, 19842, -1, -1, -1]),
        (2, [1, 2, 3, 0, 0], [19841, 19842, 19843, -1, -1]),
        (2, [1, 2, 3, 4, 0], [19841, 19842, 19843, 19844, -1]),
        (3, [10, 0, 0, 0, 0], [19844, -1, -1, -1, -1]),
        (4, [10, 11, 0, 0, 0], [19844, 19843, -1, -1, -1]),
        (4, [10, 11, 12, 0, 0], [19844, 19843, 19845, -1, -1]),
        (10, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
    ]
    return pd.DataFrame(data_two_columns_cut_left, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_two_columns_no_cut(spark, schema):
    data_two_columns_no_cut = [
        (1, [2, 0, 0, 0, 0], [19842, -1, -1, -1, -1]),
        (1, [2, 4, 0, 0, 0], [19842, 19844, -1, -1, -1]),
        (1, [2, 4, 3, 0, 0], [19842, 19844, 19843, -1, -1]),
        (1, [2, 4, 3, 5, 0], [19842, 19844, 19843, 19845, -1]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
        (2, [1, 2, 0, 0, 0], [19841, 19842, -1, -1, -1]),
        (2, [1, 2, 3, 0, 0], [19841, 19842, 19843, -1, -1]),
        (2, [1, 2, 3, 4, 0], [19841, 19842, 19843, 19844, -1]),
        (3, [10, 0, 0, 0, 0], [19844, -1, -1, -1, -1]),
        (4, [10, 11, 0, 0, 0], [19844, 19843, -1, -1, -1]),
        (4, [10, 11, 12, 0, 0], [19844, 19843, 19845, -1, -1]),
        (10, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
    ]
    return spark.createDataFrame(data_two_columns_no_cut, schema=schema)


@pytest.fixture(scope="module")
def dataframe_two_columns_no_cut_pandas():
    data_two_columns_no_cut = [
        (1, [2, 0, 0, 0, 0], [19842, -1, -1, -1, -1]),
        (1, [2, 4, 0, 0, 0], [19842, 19844, -1, -1, -1]),
        (1, [2, 4, 3, 0, 0], [19842, 19844, 19843, -1, -1]),
        (1, [2, 4, 3, 5, 0], [19842, 19844, 19843, 19845, -1]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [2, 4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
        (2, [1, 2, 0, 0, 0], [19841, 19842, -1, -1, -1]),
        (2, [1, 2, 3, 0, 0], [19841, 19842, 19843, -1, -1]),
        (2, [1, 2, 3, 4, 0], [19841, 19842, 19843, 19844, -1]),
        (3, [10, 0, 0, 0, 0], [19844, -1, -1, -1, -1]),
        (4, [10, 11, 0, 0, 0], [19844, 19843, -1, -1, -1]),
        (4, [10, 11, 12, 0, 0], [19844, 19843, 19845, -1, -1]),
        (10, [1, 0, 0, 0, 0], [19841, -1, -1, -1, -1]),
    ]
    return pd.DataFrame(data_two_columns_no_cut, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_only_item_left(spark, schema):
    data_only_item_left = [
        (1, [0, 0, 0, 0, 2], [19842]),
        (1, [0, 0, 0, 2, 4], [19842, 19844]),
        (1, [0, 0, 2, 4, 3], [19842, 19844, 19843]),
        (1, [0, 2, 4, 3, 5], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [0, 0, 0, 0, 1], [19841]),
        (2, [0, 0, 0, 1, 2], [19841, 19842]),
        (2, [0, 0, 1, 2, 3], [19841, 19842, 19843]),
        (2, [0, 1, 2, 3, 4], [19841, 19842, 19843, 19844]),
        (3, [0, 0, 0, 0, 10], [19844]),
        (4, [0, 0, 0, 10, 11], [19844, 19843]),
        (4, [0, 0, 10, 11, 12], [19844, 19843, 19845]),
        (10, [0, 0, 0, 0, 1], [19841]),
    ]
    return spark.createDataFrame(data_only_item_left, schema=schema)


@pytest.fixture(scope="module")
def dataframe_only_item_left_pandas():
    data_only_item_left = [
        (1, [0, 0, 0, 0, 2], [19842]),
        (1, [0, 0, 0, 2, 4], [19842, 19844]),
        (1, [0, 0, 2, 4, 3], [19842, 19844, 19843]),
        (1, [0, 2, 4, 3, 5], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [0, 0, 0, 0, 1], [19841]),
        (2, [0, 0, 0, 1, 2], [19841, 19842]),
        (2, [0, 0, 1, 2, 3], [19841, 19842, 19843]),
        (2, [0, 1, 2, 3, 4], [19841, 19842, 19843, 19844]),
        (3, [0, 0, 0, 0, 10], [19844]),
        (4, [0, 0, 0, 10, 11], [19844, 19843]),
        (4, [0, 0, 10, 11, 12], [19844, 19843, 19845]),
        (10, [0, 0, 0, 0, 1], [19841]),
    ]
    return pd.DataFrame(data_only_item_left, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_two_columns_left(spark, schema):
    data_two_columns_left = [
        (1, [0, 0, 0, 0, 2], [-1, -1, -1, -1, 19842]),
        (1, [0, 0, 0, 2, 4], [-1, -1, -1, 19842, 19844]),
        (1, [0, 0, 2, 4, 3], [-1, -1, 19842, 19844, 19843]),
        (1, [0, 2, 4, 3, 5], [-1, 19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 19841]),
        (2, [0, 0, 0, 1, 2], [-1, -1, -1, 19841, 19842]),
        (2, [0, 0, 1, 2, 3], [-1, -1, 19841, 19842, 19843]),
        (2, [0, 1, 2, 3, 4], [-1, 19841, 19842, 19843, 19844]),
        (3, [0, 0, 0, 0, 10], [-1, -1, -1, -1, 19844]),
        (4, [0, 0, 0, 10, 11], [-1, -1, -1, 19844, 19843]),
        (4, [0, 0, 10, 11, 12], [-1, -1, 19844, 19843, 19845]),
        (10, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 19841]),
    ]
    return spark.createDataFrame(data_two_columns_left, schema=schema)


@pytest.fixture(scope="module")
def dataframe_two_columns_left_pandas():
    data_two_columns_left = [
        (1, [0, 0, 0, 0, 2], [-1, -1, -1, -1, 19842]),
        (1, [0, 0, 0, 2, 4], [-1, -1, -1, 19842, 19844]),
        (1, [0, 0, 2, 4, 3], [-1, -1, 19842, 19844, 19843]),
        (1, [0, 2, 4, 3, 5], [-1, 19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 19841]),
        (2, [0, 0, 0, 1, 2], [-1, -1, -1, 19841, 19842]),
        (2, [0, 0, 1, 2, 3], [-1, -1, 19841, 19842, 19843]),
        (2, [0, 1, 2, 3, 4], [-1, 19841, 19842, 19843, 19844]),
        (3, [0, 0, 0, 0, 10], [-1, -1, -1, -1, 19844]),
        (4, [0, 0, 0, 10, 11], [-1, -1, -1, 19844, 19843]),
        (4, [0, 0, 10, 11, 12], [-1, -1, 19844, 19843, 19845]),
        (10, [0, 0, 0, 0, 1], [-1, -1, -1, -1, 19841]),
    ]
    return pd.DataFrame(data_two_columns_left, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def dataframe_string(spark, schema_string):
    data_string = [
        (1, ["2", "[PAD]", "[PAD]", "[PAD]", "[PAD]"], [19842]),
        (1, ["2", "4", "[PAD]", "[PAD]", "[PAD]"], [19842, 19844]),
        (1, ["2", "4", "3", "[PAD]", "[PAD]"], [19842, 19844, 19843]),
        (1, ["2", "4", "3", "5", "[PAD]"], [19842, 19844, 19843, 19845]),
        (1, ["2", "4", "3", "5", "6"], [19842, 19844, 19843, 19845, 19846]),
        (1, ["4", "3", "5", "6", "7"], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, ["1", "[PAD]", "[PAD]", "[PAD]", "[PAD]"], [19841]),
        (2, ["1", "2", "[PAD]", "[PAD]", "[PAD]"], [19841, 19842]),
        (2, ["1", "2", "3", "[PAD]", "[PAD]"], [19841, 19842, 19843]),
        (2, ["1", "2", "3", "4", "[PAD]"], [19841, 19842, 19843, 19844]),
        (3, ["10", "[PAD]", "[PAD]", "[PAD]", "[PAD]"], [19844]),
        (4, ["10", "11", "[PAD]", "[PAD]", "[PAD]"], [19844, 19843]),
        (4, ["10", "11", "12", "[PAD]", "[PAD]"], [19844, 19843, 19845]),
        (10, ["1", "[PAD]", "[PAD]", "[PAD]", "[PAD]"], [19841]),
    ]
    return spark.createDataFrame(data_string, schema=schema_string)


@pytest.fixture(scope="module")
def dataframe_string_pandas():
    data_string = [
        (1, [2, "[PAD]", "[PAD]", "[PAD]", "[PAD]"], [19842]),
        (1, [2, 4, "[PAD]", "[PAD]", "[PAD]"], [19842, 19844]),
        (1, [2, 4, 3, "[PAD]", "[PAD]"], [19842, 19844, 19843]),
        (1, [2, 4, 3, 5, "[PAD]"], [19842, 19844, 19843, 19845]),
        (1, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, [4, 3, 5, 6, 7], [19842, 19844, 19843, 19845, 19846, 19847]),
        (2, [1, "[PAD]", "[PAD]", "[PAD]", "[PAD]"], [19841]),
        (2, [1, 2, "[PAD]", "[PAD]", "[PAD]"], [19841, 19842]),
        (2, [1, 2, 3, "[PAD]", "[PAD]"], [19841, 19842, 19843]),
        (2, [1, 2, 3, 4, "[PAD]"], [19841, 19842, 19843, 19844]),
        (3, [10, "[PAD]", "[PAD]", "[PAD]", "[PAD]"], [19844]),
        (4, [10, 11, "[PAD]", "[PAD]", "[PAD]"], [19844, 19843]),
        (4, [10, 11, 12, "[PAD]", "[PAD]"], [19844, 19843, 19845]),
        (10, [1, "[PAD]", "[PAD]", "[PAD]", "[PAD]"], [19841]),
    ]
    return pd.DataFrame(data_string, columns=["user_id", "item_id", "timestamp"])


@pytest.fixture(scope="module")
def columns():
    return ["user_id", "item_id", "timestamp"]


@pytest.fixture(scope="module")
def columns_target():
    return ["user_id", "item_id", "timestamp", "item_id_list", "timestamp_list"]


@pytest.fixture(scope="module")
def columns_target_list_len():
    return ["user_id", "item_id", "timestamp", "item_id_list", "timestamp_list", "list_len"]


@pytest.fixture(scope="module")
def schema_target():
    return StructType(
        [
            StructField("user_id", LongType(), True),
            StructField("item_id", LongType(), True),
            StructField("timestamp", LongType(), True),
            StructField("item_id_list", ArrayType(LongType(), False), False),
            StructField("timestamp_list", ArrayType(LongType(), False), False),
        ]
    )


@pytest.fixture(scope="module")
def schema_target_list_len():
    return StructType(
        [
            StructField("user_id", LongType(), True),
            StructField("item_id", LongType(), True),
            StructField("timestamp", LongType(), True),
            StructField("item_id_list", ArrayType(LongType(), False), False),
            StructField("timestamp_list", ArrayType(LongType(), False), False),
            StructField("list_len", IntegerType(), False),
        ]
    )


@pytest.fixture(scope="module")
def simple_dataframe(spark, columns):
    data = [
        (1, 2, 19842),
        (1, 4, 19844),
        (1, 3, 19843),
        (1, 5, 19845),
        (1, 6, 19846),
        (1, 7, 19847),
        (2, 1, 19841),
        (2, 2, 19842),
        (2, 3, 19843),
        (2, 4, 19844),
        (3, 10, 19844),
        (4, 11, 19843),
        (4, 12, 19845),
        (1, 1, 19841),
    ]
    return spark.createDataFrame(data, schema=columns)


@pytest.fixture(scope="module")
def static_string_pd_df():
    data = []
    for _ in range(5000):
        data.append(["Moscow"])
        data.append(["Novgorod"])
    return pd.DataFrame(data, columns=["random_string"])


@pytest.fixture(scope="module")
def static_string_spark_df(
    spark,
    static_string_pd_df,
):
    return spark.createDataFrame(static_string_pd_df, schema=list(static_string_pd_df.columns))


@pytest.fixture(scope="module")
def static_string_pl_df(static_string_pd_df):
    return pl.from_pandas(static_string_pd_df)


@pytest.fixture(scope="module")
def random_string_spark_df(spark):
    random.seed(42)

    def generate_random_string(length=10):
        return "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=random.randint(1, length)))

    data = [(generate_random_string(),) for _ in range(100_000)] * 4
    return spark.createDataFrame(data, schema=["random_string"])


@pytest.fixture(scope="module")
def simple_dataframe_array(spark):
    columns_array = ["user_id", "item_id", "timestamp"]
    data_array = [
        (1, [2, 1, 0], 19842),
        (1, [4, 1], 19844),
        (1, [3, 1, 0], 19843),
        (1, [5, 1], 19845),
        (1, [6, 1, 0], 19846),
        (1, [7, 1], 19847),
        (2, [1, 0, 1], 19841),
        (2, [2, 0], 19842),
        (2, [3, 0, 1], 19843),
        (2, [4, 0], 19844),
        (3, [10, 0], 19844),
        (4, [11, 0, 1], 19843),
        (4, [12, 0], 19845),
        (1, [1, 0], 19841),
    ]
    return spark.createDataFrame(data_array, schema=columns_array)


@pytest.fixture(scope="module")
def simple_dataframe_additional(spark):
    columns_additional = ["user_id", "item_id", "timestamp", "other_column"]
    data_additional = [
        (1, 2, 19842, 0),
        (1, 4, 19844, 1),
        (1, 3, 19843, 0),
        (1, 5, 19845, 1),
        (1, 6, 19846, 0),
        (1, 7, 19847, 1),
        (2, 1, 19841, 0),
        (2, 2, 19842, 0),
        (2, 3, 19843, 0),
        (2, 4, 19844, 1),
        (3, 10, 19844, 0),
        (4, 11, 19843, 1),
        (4, 12, 19845, 1),
        (1, 1, 19841, 1),
    ]
    return spark.createDataFrame(data_additional, schema=columns_additional)


@pytest.fixture(scope="module")
def simple_dataframe_target(spark, schema_target):
    data_target = [
        (1, 4, 19844, [2], [19842]),
        (1, 3, 19843, [2, 4], [19842, 19844]),
        (1, 5, 19845, [2, 4, 3], [19842, 19844, 19843]),
        (1, 6, 19846, [2, 4, 3, 5], [19842, 19844, 19843, 19845]),
        (1, 7, 19847, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, 1, 19841, [4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, 2, 19842, [1], [19841]),
        (2, 3, 19843, [1, 2], [19841, 19842]),
        (2, 4, 19844, [1, 2, 3], [19841, 19842, 19843]),
        (4, 12, 19845, [11], [19843]),
    ]
    return spark.createDataFrame(data_target, schema=schema_target)


@pytest.fixture(scope="module")
def simple_dataframe_target_ordered(spark, schema_target):
    data_target_ordered = [
        (1, 2, 19842, [1], [19841]),
        (1, 3, 19843, [1, 2], [19841, 19842]),
        (1, 4, 19844, [1, 2, 3], [19841, 19842, 19843]),
        (1, 5, 19845, [1, 2, 3, 4], [19841, 19842, 19843, 19844]),
        (1, 6, 19846, [1, 2, 3, 4, 5], [19841, 19842, 19843, 19844, 19845]),
        (1, 7, 19847, [2, 3, 4, 5, 6], [19842, 19843, 19844, 19845, 19846]),
        (2, 2, 19842, [1], [19841]),
        (2, 3, 19843, [1, 2], [19841, 19842]),
        (2, 4, 19844, [1, 2, 3], [19841, 19842, 19843]),
        (4, 12, 19845, [11], [19843]),
    ]
    return spark.createDataFrame(data_target_ordered, schema=schema_target)


@pytest.fixture(scope="module")
def simple_dataframe_target_ordered_list_len(spark, schema_target_list_len):
    data_target_ordered_list_len = [
        (1, 2, 19842, [1], [19841], 1),
        (1, 3, 19843, [1, 2], [19841, 19842], 2),
        (1, 4, 19844, [1, 2, 3], [19841, 19842, 19843], 3),
        (1, 5, 19845, [1, 2, 3, 4], [19841, 19842, 19843, 19844], 4),
        (1, 6, 19846, [1, 2, 3, 4, 5], [19841, 19842, 19843, 19844, 19845], 5),
        (1, 7, 19847, [2, 3, 4, 5, 6], [19842, 19843, 19844, 19845, 19846], 5),
        (2, 2, 19842, [1], [19841], 1),
        (2, 3, 19843, [1, 2], [19841, 19842], 2),
        (2, 4, 19844, [1, 2, 3], [19841, 19842, 19843], 3),
        (4, 12, 19845, [11], [19843], 1),
    ]
    return spark.createDataFrame(data_target_ordered_list_len, schema=schema_target_list_len)


@pytest.fixture(scope="module")
def simple_dataframe_pandas(columns):
    data = [
        (1, 2, 19842),
        (1, 4, 19844),
        (1, 3, 19843),
        (1, 5, 19845),
        (1, 6, 19846),
        (1, 7, 19847),
        (2, 1, 19841),
        (2, 2, 19842),
        (2, 3, 19843),
        (2, 4, 19844),
        (3, 10, 19844),
        (4, 11, 19843),
        (4, 12, 19845),
        (1, 1, 19841),
    ]
    return pd.DataFrame(data, columns=columns)


@pytest.fixture(scope="module")
def simple_dataframe_polars(simple_dataframe_pandas):
    return pl.from_pandas(simple_dataframe_pandas)


@pytest.fixture(scope="module")
def dataframe_not_implemented(simple_dataframe_pandas):
    return simple_dataframe_pandas.to_numpy()


@pytest.fixture(scope="module")
def simple_dataframe_array_pandas():
    columns_array = ["user_id", "item_id", "timestamp"]
    data_array = [
        (1, [2, 1, 0], 19842),
        (1, [4, 1], 19844),
        (1, [3, 1, 0], 19843),
        (1, [5, 1], 19845),
        (1, [6, 1, 0], 19846),
        (1, [7, 1], 19847),
        (2, [1, 0, 1], 19841),
        (2, [2, 0], 19842),
        (2, [3, 0, 1], 19843),
        (2, [4, 0], 19844),
        (3, [10, 0], 19844),
        (4, [11, 0, 1], 19843),
        (4, [12, 0], 19845),
        (1, [1, 0], 19841),
    ]
    return pd.DataFrame(data_array, columns=columns_array)


@pytest.fixture(scope="module")
def simple_dataframe_array_polars(simple_dataframe_array_pandas):
    return pl.from_pandas(simple_dataframe_array_pandas)


@pytest.fixture(scope="module")
def simple_dataframe_additional_pandas():
    columns_additional = ["user_id", "item_id", "timestamp", "other_column"]
    data_additional = [
        (1, 2, 19842, 0),
        (1, 4, 19844, 1),
        (1, 3, 19843, 0),
        (1, 5, 19845, 1),
        (1, 6, 19846, 0),
        (1, 7, 19847, 1),
        (2, 1, 19841, 0),
        (2, 2, 19842, 0),
        (2, 3, 19843, 0),
        (2, 4, 19844, 1),
        (3, 10, 19844, 0),
        (4, 11, 19843, 1),
        (4, 12, 19845, 1),
        (1, 1, 19841, 1),
    ]
    return pd.DataFrame(data_additional, columns=columns_additional)


@pytest.fixture(scope="module")
def simple_dataframe_target_pandas(columns_target):
    data_target = [
        (1, 4, 19844, [2], [19842]),
        (1, 3, 19843, [2, 4], [19842, 19844]),
        (1, 5, 19845, [2, 4, 3], [19842, 19844, 19843]),
        (1, 6, 19846, [2, 4, 3, 5], [19842, 19844, 19843, 19845]),
        (1, 7, 19847, [2, 4, 3, 5, 6], [19842, 19844, 19843, 19845, 19846]),
        (1, 1, 19841, [4, 3, 5, 6, 7], [19844, 19843, 19845, 19846, 19847]),
        (2, 2, 19842, [1], [19841]),
        (2, 3, 19843, [1, 2], [19841, 19842]),
        (2, 4, 19844, [1, 2, 3], [19841, 19842, 19843]),
        (4, 12, 19845, [11], [19843]),
    ]
    return pd.DataFrame(data_target, columns=columns_target)


@pytest.fixture(scope="module")
def simple_dataframe_target_ordered_pandas(columns_target):
    data_target_ordered = [
        (1, 2, 19842, [1], [19841]),
        (1, 3, 19843, [1, 2], [19841, 19842]),
        (1, 4, 19844, [1, 2, 3], [19841, 19842, 19843]),
        (1, 5, 19845, [1, 2, 3, 4], [19841, 19842, 19843, 19844]),
        (1, 6, 19846, [1, 2, 3, 4, 5], [19841, 19842, 19843, 19844, 19845]),
        (1, 7, 19847, [2, 3, 4, 5, 6], [19842, 19843, 19844, 19845, 19846]),
        (2, 2, 19842, [1], [19841]),
        (2, 3, 19843, [1, 2], [19841, 19842]),
        (2, 4, 19844, [1, 2, 3], [19841, 19842, 19843]),
        (4, 12, 19845, [11], [19843]),
    ]
    return pd.DataFrame(data_target_ordered, columns=columns_target)


@pytest.fixture(scope="module")
def simple_dataframe_target_ordered_list_len_pandas(columns_target_list_len):
    data_target_ordered_list_len = [
        (1, 2, 19842, [1], [19841], 1),
        (1, 3, 19843, [1, 2], [19841, 19842], 2),
        (1, 4, 19844, [1, 2, 3], [19841, 19842, 19843], 3),
        (1, 5, 19845, [1, 2, 3, 4], [19841, 19842, 19843, 19844], 4),
        (1, 6, 19846, [1, 2, 3, 4, 5], [19841, 19842, 19843, 19844, 19845], 5),
        (1, 7, 19847, [2, 3, 4, 5, 6], [19842, 19843, 19844, 19845, 19846], 5),
        (2, 2, 19842, [1], [19841], 1),
        (2, 3, 19843, [1, 2], [19841, 19842], 2),
        (2, 4, 19844, [1, 2, 3], [19841, 19842, 19843], 3),
        (4, 12, 19845, [11], [19843], 1),
    ]
    return pd.DataFrame(data_target_ordered_list_len, columns=columns_target_list_len)


@pytest.fixture(scope="module")
def dataframe_sessionizer(spark):
    columns = ["user_id", "item_id", "timestamp"]
    data = [
        (1, 1, "01-01-2020"),
        (1, 2, "02-01-2020"),
        (1, 3, "03-01-2020"),
        (2, 1, "06-01-2020"),
        (2, 2, "07-01-2020"),
        (2, 3, "08-01-2020"),
        (2, 9, "09-01-2020"),
        (3, 1, "01-01-2020"),
        (3, 5, "02-01-2020"),
        (3, 3, "03-01-2020"),
        (3, 1, "04-01-2020"),
        (3, 2, "05-01-2020"),
    ]
    return (
        spark.createDataFrame(data, schema=columns)
        .withColumn("spark_date", to_date(col("timestamp"), "dd-MM-yyyy"))
        .withColumn("timestamp", unix_timestamp(col("spark_date")))
    )


@pytest.fixture(scope="module")
def dataframe_sessionizer_pandas():
    columns = ["user_id", "item_id", "timestamp"]
    data = [
        (1, 1, "01-01-2020"),
        (1, 2, "02-01-2020"),
        (1, 3, "03-01-2020"),
        (2, 1, "06-01-2020"),
        (2, 2, "07-01-2020"),
        (2, 3, "08-01-2020"),
        (2, 9, "09-01-2020"),
        (3, 1, "01-01-2020"),
        (3, 5, "02-01-2020"),
        (3, 3, "03-01-2020"),
        (3, 1, "04-01-2020"),
        (3, 2, "05-01-2020"),
    ]

    df = pd.DataFrame(data, columns=columns)
    df["pandas_date"] = pd.to_datetime(df["timestamp"], format="%d-%m-%Y")
    df["timestamp"] = [
        1577826000,
        1577912400,
        1577998800,
        1578258000,
        1578344400,
        1578430800,
        1578517200,
        1577826000,
        1577912400,
        1577998800,
        1578085200,
        1578171600,
    ]

    return df


@pytest.fixture(scope="module")
def session_dataset_pandas():
    data = {
        "user_id": [1, 1, 1, 2, 2, 2, 2, 3, 3],
        "item_id": [1, 2, 1, 3, 5, 6, 7, 8, 9],
        "timestamp": [10, 200, 220, 40, 55, 75, 100, 245, 350],
    }

    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def session_dataset_spark(spark, session_dataset_pandas):
    return spark.createDataFrame(session_dataset_pandas)


@pytest.fixture(scope="module")
def simple_data_to_filter_pandas():
    df = pd.DataFrame(
        {
            "query_id": [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 1, 0, 2, 2],
            "item_id": [
                10,
                10,
                11,
                12,
                15,
                10,
                11,
                11,
                13,
                14,
                15,
                13,
                10,
                11,
                11,
                14,
                10,
                12,
                13,
                15,
                15,
                11,
                11,
                10,
                11,
            ],
            "timestamp": [0, 1, 1, 2, 0, 2, 2, 3, 1, 1, 1, 1, 0, 2, 1, 3, 1, 0, 0, 0, 0, 2, 3, 3, 1],
        }
    )
    return df


@pytest.fixture(scope="module")
def simple_data_to_filter_polars(simple_data_to_filter_pandas):
    df = pl.DataFrame(simple_data_to_filter_pandas)
    return df


@pytest.fixture(scope="module")
def simple_data_to_filter_spark(simple_data_to_filter_pandas, spark):
    df = spark.createDataFrame(simple_data_to_filter_pandas)
    return df


@pytest.fixture(scope="module")
def interactions_100k_pandas():
    values = np.random.randint([1, 1], [1000, 10000], size=(int(1e5), 2))
    data = pd.DataFrame(values, columns=["user_id", "item_id"])
    return data


@pytest.fixture(scope="module")
def interactions_100k_polars(interactions_100k_pandas):
    return pl.from_pandas(interactions_100k_pandas)


@pytest.fixture(scope="module")
@pytest.mark.usefixtures("spark")
def interactions_100k_spark(spark, interactions_100k_pandas):
    return spark.createDataFrame(interactions_100k_pandas)


@pytest.fixture(scope="module")
def interactions_100k_pandas_with_nan(interactions_100k_pandas):
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    nan_index = np.random.choice(interactions_100k_pandas_with_nan.index, size=200, replace=False)
    interactions_100k_pandas_with_nan.loc[nan_index, ["user_id", "item_id"]] = np.nan
    return interactions_100k_pandas_with_nan


@pytest.fixture(scope="module")
def interactions_100k_polars_with_nan(interactions_100k_pandas_with_nan):
    return pl.from_pandas(interactions_100k_pandas_with_nan)


@pytest.fixture(scope="module")
@pytest.mark.usefixtures("spark")
def interactions_100k_spark_with_nan(spark, interactions_100k_pandas_with_nan):
    return spark.createDataFrame(interactions_100k_pandas_with_nan.values.tolist(), schema=["user_id", "item_id"])
