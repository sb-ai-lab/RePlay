import builtins
import importlib
import logging
import os
import sys
import types
from datetime import datetime, timezone
from functools import partial
from unittest.mock import Mock

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
from replay.utils import spark_utils as utils
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


def _make_builder_dummy_spark_session():
    class DummySparkSession:
        class _Builder:
            def __init__(self):
                self.configs = []

            def config(self, key, value):
                self.configs.append((key, value))
                return self

            def master(self, _value):
                return self

            def getOrCreate(self):
                return object()

        builder = _Builder()

    return DummySparkSession


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
def test_get_spark_session_spark4_warning(monkeypatch, caplog):
    dummy_spark_session = _make_builder_dummy_spark_session()

    monkeypatch.delenv("REPLAY_JAR_PATH", raising=False)
    monkeypatch.delenv("SCRIPT_ENV", raising=False)
    monkeypatch.setattr(replay.utils.session_handler, "SparkSession", dummy_spark_session)
    monkeypatch.setattr(replay.utils.session_handler, "pyspark_version", "4.0.0")

    with caplog.at_level(logging.WARNING):
        replay.utils.session_handler.get_spark_session(spark_memory=1, shuffle_partitions=1, core_count=1)

    assert "Spark 4.x detected. Replay Scala extensions are disabled by default" in caplog.text
    assert not any(key == "spark.jars" for key, _ in dummy_spark_session.builder.configs)


@pytest.mark.spark
def test_get_spark_session_sets_spark_jars_from_env(monkeypatch):
    dummy_spark_session = _make_builder_dummy_spark_session()

    monkeypatch.setenv("REPLAY_JAR_PATH", "/tmp/replay.jar")
    monkeypatch.delenv("SCRIPT_ENV", raising=False)
    monkeypatch.setattr(replay.utils.session_handler, "SparkSession", dummy_spark_session)
    monkeypatch.setattr(replay.utils.session_handler, "pyspark_version", "4.0.0")

    replay.utils.session_handler.get_spark_session(spark_memory=1, shuffle_partitions=1, core_count=1)

    assert ("spark.jars", "/tmp/replay.jar") in dummy_spark_session.builder.configs


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
def test_import_fallback_to_classic_column(monkeypatch):
    original_import = builtins.__import__
    error_msg = "simulated missing pyspark.sql.column"
    fake_classic_column = types.ModuleType("pyspark.sql.classic.column")
    fake_to_java_column = object()
    fake_to_seq = object()
    fake_classic_column._to_java_column = fake_to_java_column
    fake_classic_column._to_seq = fake_to_seq

    def patched_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        if name == "pyspark.sql.column":
            raise ImportError(error_msg)
        return original_import(name, globals_, locals_, fromlist, level)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", patched_import)
        m.setitem(sys.modules, "pyspark.sql.classic.column", fake_classic_column)
        reloaded_utils = importlib.reload(utils)
        assert reloaded_utils._to_java_column is fake_to_java_column
        assert reloaded_utils._to_seq is fake_to_seq

    importlib.reload(utils)


@pytest.mark.spark
def test_multiply_scala_udf_uses_python_fallback(monkeypatch, spark):
    del spark
    sentinel = object()
    called = {}

    def fake_vector_mult(scalar, vector):
        called["args"] = (scalar, vector)
        return sentinel

    monkeypatch.setattr(utils, "_to_java_column", None)
    monkeypatch.setattr(utils, "_to_seq", None)
    monkeypatch.setattr(utils, "vector_mult", fake_vector_mult)

    result = utils.multiply_scala_udf("s", "v")
    assert result is sentinel
    assert called["args"] == ("s", "v")


@pytest.mark.spark
def test_multiply_scala_udf_uses_scala_path(monkeypatch, spark):
    del spark
    sentinel = object()
    to_java_column = object()
    to_seq = Mock(return_value="seq")
    scala_apply = Mock(return_value="java_column")
    multiply_udf = Mock(return_value=types.SimpleNamespace(apply=scala_apply))
    column_ctor = Mock(return_value=sentinel)

    jvm = types.SimpleNamespace(
        org=types.SimpleNamespace(
            apache=types.SimpleNamespace(
                spark=types.SimpleNamespace(
                    replay=types.SimpleNamespace(
                        utils=types.SimpleNamespace(ScalaPySparkUDFs=types.SimpleNamespace(multiplyUDF=multiply_udf))
                    )
                )
            )
        )
    )
    spark_context = types.SimpleNamespace(_jvm=jvm)
    spark_instance = types.SimpleNamespace(sparkContext=spark_context)
    spark_session = types.SimpleNamespace(getActiveSession=lambda: spark_instance)

    monkeypatch.setattr(utils, "SparkSession", spark_session)
    monkeypatch.setattr(utils, "_to_java_column", to_java_column)
    monkeypatch.setattr(utils, "_to_seq", to_seq)
    monkeypatch.setattr(utils, "Column", column_ctor)

    result = utils.multiply_scala_udf("s", "v")
    assert result is sentinel
    to_seq.assert_called_once_with(spark_context, ["s", "v"], to_java_column)
    multiply_udf.assert_called_once_with()
    scala_apply.assert_called_once_with("seq")
    column_ctor.assert_called_once_with("java_column")


@pytest.mark.spark
@pytest.mark.parametrize("array", [None, [1, 2, 2, 3]])
def test_get_unique_entities(spark, array):
    log = spark.createDataFrame(data=[[1], [2], [3]], schema=["test"])
    assert sorted(utils.get_unique_entities(array or log, "test").toPandas()["test"]) == [1, 2, 3]


@pytest.mark.spark
def test_utils_time_raise():
    d = {
        "item_idx": [1, 1, 2, 3, 3],
        "timestamp": ["2099-03-19", "2099-03-20", "2099-03-22", "2099-03-27", "2099-03-25"],
        "relevance": [1, 1, 1, 1, 1],
    }
    df = pd.DataFrame(d)
    with pytest.raises(ValueError, match=r"parameter kind must be one of [power, exp, linear]*"):
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
    with pytest.raises(ValueError, match=r"Wrong type <class 'int'>"):
        utils.get_unique_entities(df=log, column="fake_column")


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
