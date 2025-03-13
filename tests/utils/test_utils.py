import logging
import os
import tempfile
from datetime import datetime, timezone
from functools import partial

import pandas as pd
import polars as pl
import pytest
from polars.testing import assert_frame_equal as pl_assert_frame_equal

from tests.utils import sparkDataFrameEqual

pyspark = pytest.importorskip("pyspark")

import pyspark.sql.functions as sf
from pyspark.sql import SparkSession
from pyspark.sql.types import TimestampType

import replay.utils.session_handler
from replay.utils import (
    pandas_utils,
    polars_utils,
    spark_utils as utils,
)
from replay.utils.common import _check_if_dataframe
from replay.utils.time import get_item_recency

datetime = partial(datetime, tzinfo=timezone.utc)

different_timestamp_formats_data = [
    (
        [
            [1.0, 1],
            [300003.0, 300003],
            [0.0, 0],
        ],
        [
            [datetime(1970, 1, 1, 0, 0, 1), datetime(1970, 1, 1, 0, 0, 1)],
            [datetime(1970, 1, 4, 11, 20, 3), datetime(1970, 1, 4, 11, 20, 3)],
            [datetime(1970, 1, 1, 0, 0, 0), datetime(1970, 1, 1, 0, 0, 0)],
        ],
        ["float_", "int_"],
    ),
    (
        [
            [datetime(2021, 8, 22), "2021-08-22", "22.08.2021"],
            [
                datetime(2021, 8, 23, 11, 29, 29),
                "2021-08-23 11:29:29",
                "23.08.2021-11:29:29",
            ],
            [datetime(2021, 8, 27), "2021-08-27", "27.08.2021"],
        ],
        [
            [
                datetime(2021, 8, 22),
                datetime(2021, 8, 22),
                datetime(2021, 8, 22),
            ],
            [
                datetime(2021, 8, 23, 11, 29, 29),
                datetime(2021, 8, 23, 11, 29, 29),
                datetime(2021, 8, 23, 11, 29, 29),
            ],
            [
                datetime(2021, 8, 27),
                datetime(2021, 8, 27),
                datetime(2021, 8, 27),
            ],
        ],
        ["ts_", "str_", "str_format_"],
    ),
]


@pytest.mark.spark
@pytest.mark.parametrize("log_data, ground_truth_data, schema", different_timestamp_formats_data)
def test_process_timestamp(log_data, ground_truth_data, schema, spark):
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    log = spark.createDataFrame(data=log_data, schema=schema)
    ground_truth = spark.createDataFrame(data=ground_truth_data, schema=schema)
    for col in log.columns:
        kwargs = {"date_format": "dd.MM.yyyy[-HH:mm:ss[.SSS]]"} if col == "str_format_" else {}
        log = utils.process_timestamp_column(log, col, **kwargs)
        assert isinstance(log.schema[col].dataType, TimestampType)
    sparkDataFrameEqual(log, ground_truth)
    spark.conf.unset("spark.sql.session.timeZone")


@pytest.mark.spark
def test_get_spark_session():
    spark = replay.utils.session_handler.get_spark_session(1)
    assert isinstance(spark, SparkSession)
    assert spark.conf.get("spark.driver.memory") == "1g"
    assert replay.utils.session_handler.State(spark).session is spark
    assert replay.utils.session_handler.State().session is spark


@pytest.mark.spark
def test_convert():
    dataframe_pandas = pd.DataFrame([[1, "a", 3.0], [3, "b", 5.0]], columns=["a", "b", "c"])
    dataframe_polars = pl.DataFrame({"a": [1, 3], "b": ["a", "b"], "c": [3.0, 5.0]})

    spark_df = utils.convert2spark(dataframe_pandas)
    pd.testing.assert_frame_equal(dataframe_pandas, spark_df.toPandas())
    assert utils.convert2spark(spark_df) is spark_df

    spark_df = utils.convert2spark(dataframe_polars)
    pl_assert_frame_equal(dataframe_polars, pl.from_pandas(spark_df.toPandas()))

    pd.testing.assert_frame_equal(dataframe_pandas, dataframe_polars.to_pandas())


@pytest.mark.spark
@pytest.mark.parametrize("array", [None, [1, 2, 2, 3]])
def test_get_unique_entities(spark, array):
    log = spark.createDataFrame(data=[[1], [2], [3]], schema=["test"])
    assert sorted(utils.get_unique_entities(array or log, "test").toPandas()["test"]) == [1, 2, 3]


@pytest.mark.core
@pytest.mark.parametrize("array", [None, [1, 2, 2, 3]])
def test_get_unique_entities_polars(array):
    log = pl.DataFrame({"test": [1, 2, 3]})
    assert sorted(polars_utils.get_unique_entities(array or log, "test")["test"]) == [1, 2, 3]


@pytest.mark.core
@pytest.mark.parametrize("array", [None, [1, 2, 2, 3]])
def test_get_unique_entities_pandas(array):
    log = pd.DataFrame(data=[[1], [2], [3]], columns=["test"])
    assert sorted(pandas_utils.get_unique_entities(array or log, "test")["test"]) == [1, 2, 3]


@pytest.mark.spark
def test_utils_time_raise():
    d = {
        "item_idx": [1, 1, 2, 3, 3],
        "timestamp": ["2099-03-19", "2099-03-20", "2099-03-22", "2099-03-27", "2099-03-25"],
        "relevance": [1, 1, 1, 1, 1],
    }
    df = pd.DataFrame(d)
    with pytest.raises(ValueError, match="parameter kind must be one of [power, exp, linear]*"):
        get_item_recency(df, kind="fake_kind")


@pytest.mark.spark
def test_check_numeric_raise(spark):
    log = spark.createDataFrame(data=[["category1"], ["category2"], ["category3"]], schema=["test"])
    with pytest.raises(ValueError):
        utils.check_numeric(log)


@pytest.mark.spark
def test_join_or_return_return(spark):
    log = spark.createDataFrame(data=[["category1"], ["category2"], ["category3"]], schema=["test"])
    returned_log = utils.join_or_return(log, None, on="none", how="left")
    sparkDataFrameEqual(log, returned_log)


@pytest.mark.spark
def test_cache_if_exists(spark):
    log = spark.createDataFrame(data=[["category1"], ["category2"], ["category3"]], schema=["test"])
    returned_log = utils.cache_if_exists(log)
    sparkDataFrameEqual(log, returned_log)
    assert utils.cache_if_exists(None) is None


@pytest.mark.spark
def test_join_with_col_renaming(spark):
    log1 = spark.createDataFrame(
        data=[["category1", 1], ["category2", 2], ["category3", 3]], schema=["cat_feature", "num_feature1"]
    )
    log2 = spark.createDataFrame(data=[["category1", 1], ["category4", 4]], schema=["cat_feature", "num_feature2"])
    returned_log = utils.join_with_col_renaming(left=log1, right=log2, on_col_name="cat_feature", how="right")
    assert returned_log.count() == 2


@pytest.mark.spark
def test_process_timestamp_column_raise(spark):
    log = spark.createDataFrame(
        data=[["category1", 1], ["category2", 2], ["category3", 3]], schema=["cat_feature", "num_feature1"]
    )
    with pytest.raises(ValueError):
        utils.process_timestamp_column(dataframe=log, column_name="fake_column")


@pytest.mark.spark
def test_get_unique_entities_fake_column():
    log = 42
    with pytest.raises(ValueError, match="Wrong type <class 'int'>"):
        utils.get_unique_entities(df=log, column="fake_column")


@pytest.mark.core
def test_get_unique_entities_fake_pandas():
    log = 42
    with pytest.raises(ValueError, match="Wrong type <class 'int'>"):
        pandas_utils.get_unique_entities(df=log, column="fake_column")


@pytest.mark.core
def test_get_unique_entities_fake_polars():
    log = 42
    with pytest.raises(ValueError, match="Wrong type <class 'int'>"):
        polars_utils.get_unique_entities(df=log, column="fake_column")


@pytest.mark.spark
def test_assert_omp_single_thread(caplog):
    saved_omp = os.environ.get("OMP_NUM_THREADS", None)
    os.environ["OMP_NUM_THREADS"] = "42"
    with caplog.at_level(logging.WARNING):
        utils.assert_omp_single_thread()
        assert (
            'Environment variable "OMP_NUM_THREADS" is set to "42". '
            "Set it to 1 if the working process freezes." in caplog.text
        )

    if saved_omp is not None:
        os.environ["OMP_NUM_THREADS"] = saved_omp
    else:
        del os.environ["OMP_NUM_THREADS"]


@pytest.mark.spark
def test_sample_top_k(long_log_with_features):
    res = utils.sample_top_k_recs(long_log_with_features, 1, seed=123)
    assert res.count() == long_log_with_features.select("user_idx").distinct().count()
    test_rel = (
        res.withColumnRenamed("relevance", "predicted_relevance")
        .join(long_log_with_features, on=["user_idx", "item_idx"])
        .withColumn("wrong_rel", sf.col("relevance") != sf.col("predicted_relevance"))
    )
    assert test_rel.selectExpr("any(wrong_rel)").collect()[0][0] is False


@pytest.mark.parametrize(
    "data_dict",
    [
        pytest.param("fake_fit_queries", marks=pytest.mark.spark),
        pytest.param("fake_fit_queries_pandas", marks=pytest.mark.core),
        pytest.param("fake_fit_queries_polars", marks=pytest.mark.core),
    ],
)
def test_check_if_dataframe_true(data_dict, request):
    dataframe = request.getfixturevalue(data_dict)
    _check_if_dataframe(dataframe)


@pytest.mark.core
def test_check_if_dataframe_false():
    var = "Definitely not a dataframe"
    with pytest.raises(ValueError):
        _check_if_dataframe(var)


@pytest.mark.core
@pytest.mark.parametrize(
    "df, warm_df, expected_num_cold, expected_filtered_df",
    [
        (None, pl.DataFrame({"id": [1, 2]}), 0, None),
        (
            pl.DataFrame({"id": [1, 2], "value": [10, 20]}),
            pl.DataFrame({"id": [1, 2]}),
            0,
            pl.DataFrame({"id": [1, 2], "value": [10, 20]}),
        ),
        (
            pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}),
            pl.DataFrame({"id": [1, 2]}),
            1,
            pl.DataFrame({"id": [1, 2], "value": [10, 20]}),
        ),
    ],
)
def test_filter_cold(df, warm_df, expected_num_cold, expected_filtered_df):
    col_name = "id"
    num_cold, filtered_df = polars_utils.filter_cold(df, warm_df, col_name)

    assert num_cold == expected_num_cold
    if expected_filtered_df is not None:
        assert filtered_df.equals(expected_filtered_df)


@pytest.mark.spark
@pytest.mark.parametrize(
    "data, order_by_col, k, expected_data",
    [
        (
            {"user": [1, 1, 2, 2], "item": [10, 20, 30, 40], "score": [0.9, 0.8, 0.7, 0.6]},
            [("score", False)],
            1,
            {"user": [1, 2], "item": [10, 30], "score": [0.9, 0.7]},
        ),
        (
            {"user": [1, 1, 2, 2], "item": [10, 20, 30, 40], "score": [0.9, 0.8, 0.7, 0.6]},
            [("score", True)],
            2,
            {"user": [1, 1, 2, 2], "item": [20, 10, 40, 30], "score": [0.8, 0.9, 0.6, 0.7]},
        ),
    ],
)
def test_get_top_k_spark(data, order_by_col, k, expected_data):
    partition_by_col = "user"
    pd_dataframe = pd.DataFrame(data)
    spark_dataframe = utils.convert2spark(pd_dataframe)
    spark_order_cols = [
        sf.col(column).desc() if order is False else sf.col(column).asc() for column, order in order_by_col
    ]
    expected_df_pd = pd.DataFrame(expected_data)
    expected_df_spark = utils.convert2spark(expected_df_pd)
    result_df_pd = pandas_utils.get_top_k(pd_dataframe, partition_by_col, order_by_col, k)
    result_df_spark = utils.get_top_k(spark_dataframe, partition_by_col, spark_order_cols, k)
    assert result_df_pd.equals(expected_df_pd)
    assert result_df_spark.toPandas().equals(expected_df_spark.toPandas())
    assert result_df_pd.equals(result_df_spark.toPandas())


@pytest.mark.core
@pytest.mark.parametrize(
    "data, order_by_col, k, expected_data",
    [
        (
            {"user": [1, 1, 2, 2], "item": [10, 20, 30, 40], "score": [0.9, 0.8, 0.7, 0.6]},
            [("score", False)],
            1,
            {"user": [1, 2], "item": [10, 30], "score": [0.9, 0.7]},
        ),
        (
            {"user": [1, 1, 2, 2], "item": [10, 20, 30, 40], "score": [0.9, 0.8, 0.7, 0.6]},
            [("score", True)],
            2,
            {"user": [1, 1, 2, 2], "item": [20, 10, 40, 30], "score": [0.8, 0.9, 0.6, 0.7]},
        ),
    ],
)
def test_get_top_k_pandas_polars(data, order_by_col, k, expected_data):
    partition_by_col = "user"
    pd_dataframe = pd.DataFrame(data)
    pl_dataframe = pl.DataFrame(data)
    expected_df_pd = pd.DataFrame(expected_data)
    expected_df_pl = pl.DataFrame(expected_data)
    result_df_pd = pandas_utils.get_top_k(pd_dataframe, partition_by_col, order_by_col, k)
    result_df_pl = polars_utils.get_top_k(pl_dataframe, partition_by_col, order_by_col, k)
    assert result_df_pd.equals(expected_df_pd)
    assert result_df_pl.equals(expected_df_pl)
    assert result_df_pd.equals(result_df_pl.to_pandas())


@pytest.mark.spark
@pytest.mark.parametrize(
    "data, k, query_column, rating_column, expected_data",
    [
        (
            {"user": [1, 1, 2, 2], "item": [10, 20, 30, 40], "relevance": [0.9, 0.8, 0.7, 0.6]},
            1,
            "user",
            "relevance",
            {"user": [1, 2], "item": [10, 30], "relevance": [0.9, 0.7]},
        ),
        (
            {"user": [1, 1, 2, 2], "item": [10, 20, 30, 40], "relevance": [0.9, 0.8, 0.7, 0.6]},
            2,
            "user",
            "relevance",
            {"user": [1, 1, 2, 2], "item": [10, 20, 30, 40], "relevance": [0.9, 0.8, 0.7, 0.6]},
        ),
    ],
)
def test_get_top_k_recs_spark(data, k, query_column, rating_column, expected_data):
    query_column = "user"
    rating_column = "relevance"
    pd_dataframe = pd.DataFrame(data)
    spark_dataframe = utils.convert2spark(pd_dataframe)
    expected_df_pd = pd.DataFrame(expected_data)
    expected_df_spark = utils.convert2spark(expected_df_pd)
    result_df_pd = pandas_utils.get_top_k_recs(pd_dataframe, k, query_column, rating_column)
    result_df_spark = utils.get_top_k_recs(spark_dataframe, k, query_column, rating_column)
    assert result_df_pd.equals(expected_df_pd)
    assert result_df_spark.toPandas().equals(expected_df_spark.toPandas())
    assert result_df_pd.equals(result_df_spark.toPandas())


@pytest.mark.core
@pytest.mark.parametrize(
    "data, k, query_column, rating_column, expected_data",
    [
        (
            {"user": [1, 1, 2, 2], "item": [10, 20, 30, 40], "relevance": [0.9, 0.8, 0.7, 0.6]},
            1,
            "user",
            "relevance",
            {"user": [1, 2], "item": [10, 30], "relevance": [0.9, 0.7]},
        ),
        (
            {"user": [1, 1, 2, 2], "item": [10, 20, 30, 40], "relevance": [0.9, 0.8, 0.7, 0.6]},
            2,
            "user",
            "relevance",
            {"user": [1, 1, 2, 2], "item": [10, 20, 30, 40], "relevance": [0.9, 0.8, 0.7, 0.6]},
        ),
    ],
)
def test_get_top_k_recs_pandas_polars(data, k, query_column, rating_column, expected_data):
    query_column = "user"
    rating_column = "relevance"
    pd_dataframe = pd.DataFrame(data)
    pl_dataframe = pl.DataFrame(data)
    expected_df_pd = pd.DataFrame(expected_data)
    expected_df_pl = pl.DataFrame(expected_data)
    result_df_pd = pandas_utils.get_top_k_recs(
        pd_dataframe,
        k,
        query_column,
        rating_column,
    )
    result_df_pl = polars_utils.get_top_k_recs(pl_dataframe, k, query_column, rating_column)
    assert result_df_pd.equals(expected_df_pd)
    assert result_df_pl.equals(expected_df_pl)
    assert result_df_pd.equals(result_df_pl.to_pandas())


@pytest.mark.core
@pytest.mark.parametrize(
    "recs, recs_file_path, expected_result",
    [
        (
            pl.DataFrame({"user": [1, 2], "item": [10, 20], "relevance": [0.9, 0.8]}),
            None,
            pl.DataFrame({"user": [1, 2], "item": [10, 20], "relevance": [0.9, 0.8]}),
        ),
        (pl.DataFrame({"user": [1, 2], "item": [10, 20], "relevance": [0.9, 0.8]}), "/tmp/test.parquet", None),
    ],
)
def test_return_recs_polars(recs, recs_file_path, expected_result):
    if recs_file_path is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recs_file_path = os.path.join(temp_dir, "test.parquet")
            result = polars_utils.return_recs(recs, recs_file_path)
            assert result is None
            loaded_df = pl.read_parquet(recs_file_path)
            assert loaded_df.equals(recs)
    else:
        result = polars_utils.return_recs(recs, recs_file_path)
        assert result.equals(expected_result)


@pytest.mark.core
@pytest.mark.parametrize(
    "recs, recs_file_path, expected_result",
    [
        (
            pd.DataFrame({"user": [1, 2], "item": [10, 20], "relevance": [0.9, 0.8]}),
            None,
            pd.DataFrame({"user": [1, 2], "item": [10, 20], "relevance": [0.9, 0.8]}),
        ),
        (pd.DataFrame({"user": [1, 2], "item": [10, 20], "relevance": [0.9, 0.8]}), "/tmp/test.parquet", None),
    ],
)
def test_return_recs_pandas(recs, recs_file_path, expected_result):
    if recs_file_path is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recs_file_path = os.path.join(temp_dir, "test.parquet")
            result = pandas_utils.return_recs(recs, recs_file_path)
            assert result is None
            loaded_df = pd.read_parquet(recs_file_path)
            assert loaded_df.equals(recs)
    else:
        result = pandas_utils.return_recs(recs, recs_file_path)
        assert result.equals(expected_result)


@pytest.mark.spark
@pytest.mark.parametrize(
    "recs, recs_file_path, expected_result",
    [
        (
            {"user": [1, 2], "item": [10, 20], "relevance": [0.9, 0.8]},
            None,
            {"user": [1, 2], "item": [10, 20], "relevance": [0.9, 0.8]},
        ),
        ({"user": [1, 2], "item": [10, 20], "relevance": [0.9, 0.8]}, "/tmp/test.parquet", None),
    ],
)
def test_return_recs_spark(recs, recs_file_path, expected_result):
    recs = utils.convert2spark(pd.DataFrame(recs))
    expected_result = pd.DataFrame(expected_result)
    if recs_file_path is not None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recs_file_path = os.path.join(temp_dir, "test.parquet")
            result = utils.return_recs(recs, recs_file_path)
            assert result is None
            loaded_df = pd.read_parquet(recs_file_path)
            assert loaded_df.equals(recs.toPandas())
    else:
        result = utils.return_recs(recs, recs_file_path)
        assert result.toPandas().equals(expected_result)
