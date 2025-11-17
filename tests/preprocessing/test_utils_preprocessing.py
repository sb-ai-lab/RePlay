import logging

import pandas as pd
import polars as pl
import pytest

from replay.preprocessing.utils import merge_subsets
from replay.utils import PYSPARK_AVAILABLE


@pytest.mark.core
def test_empty_input_raises():
    with pytest.raises(ValueError, match="At least one dataframe is required"):
        merge_subsets([])


@pytest.mark.core
def test_mixed_types_raise_typeerror():
    df_pd = pd.DataFrame({"a": [1], "b": [2]})
    df_pl = pl.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(TypeError, match="All input dataframes must be of the same type"):
        merge_subsets([df_pd, df_pl])


@pytest.mark.core
def test_unsupported_type_raises_notimplemented():
    with pytest.raises(NotImplementedError, match="Unsupported data frame type"):
        merge_subsets([42])


@pytest.mark.core
def test_pandas_ignore_duplicates():
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [2, 3], "b": ["y", "z"]})
    res = merge_subsets([df1, df2], on_duplicate="ignore")
    pd.testing.assert_frame_equal(res, pd.concat([df1, df2], ignore_index=True))


@pytest.mark.core
def test_pandas_error_on_duplicates():
    df1 = pd.DataFrame({"a": [1], "b": ["x"]})
    df2 = pd.DataFrame({"a": [1], "b": ["x"]})
    with pytest.raises(ValueError, match=r"Found 1 duplicate rows"):
        merge_subsets([df1, df2], on_duplicate="error")


@pytest.mark.core
def test_pandas_drop_duplicates_logs(caplog):
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [2, 2], "b": ["y", "y"]})
    with caplog.at_level(logging.WARNING, logger="replay"):
        res = merge_subsets([df1, df2], on_duplicate="drop")
        exp = pd.concat([df1, df2], ignore_index=True).drop_duplicates(["a", "b"], keep="first").reset_index(drop=True)
        pd.testing.assert_frame_equal(res, exp)
        assert "duplicate rows" in caplog.text


@pytest.mark.core
def test_pandas_subset_for_duplicates():
    df1 = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pd.DataFrame({"a": [2, 2], "b": ["z", "w"]})
    with pytest.raises(ValueError, match="duplicate"):
        merge_subsets([df1, df2], on_duplicate="error", subset_for_duplicates=["a"])


@pytest.mark.core
def test_pandas_check_columns_and_alignment():
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"b": [3], "a": [4]})
    res = merge_subsets([df1, df2], check_columns=True)
    assert list(res.columns) == ["a", "b"]
    pd.testing.assert_frame_equal(res, pd.DataFrame({"a": [1, 4], "b": [2, 3]}))


@pytest.mark.core
def test_pandas_columns_mismatch_raises():
    df1 = pd.DataFrame({"a": [1], "b": [2]})
    df2 = pd.DataFrame({"a": [3], "c": [4]})
    with pytest.raises(ValueError, match="Columns mismatch"):
        merge_subsets([df1, df2], check_columns=True)


@pytest.mark.core
def test_pandas_columns_param_select_subset_when_not_checking():
    df1 = pd.DataFrame({"a": [1], "b": [2], "c": [0]})
    df2 = pd.DataFrame({"b": [3], "a": [4], "c": [9]})
    res = merge_subsets([df1, df2], columns=["a", "b"], check_columns=False, on_duplicate="ignore")
    exp = pd.DataFrame({"a": [1, 4], "b": [2, 3]})
    pd.testing.assert_frame_equal(res, exp)


@pytest.mark.core
def test_polars_ignore_duplicates():
    df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pl.DataFrame({"a": [2, 3], "b": ["y", "z"]})
    res = merge_subsets([df1, df2], on_duplicate="ignore")
    exp = pl.concat([df1.select(["a", "b"]), df2.select(["a", "b"])], how="vertical")
    assert res.equals(exp)


@pytest.mark.core
def test_polars_error_on_duplicates():
    df1 = pl.DataFrame({"a": [1], "b": ["x"]})
    df2 = pl.DataFrame({"a": [1], "b": ["x"]})
    with pytest.raises(ValueError):
        merge_subsets([df1, df2], on_duplicate="error")


@pytest.mark.core
def test_polars_drop_duplicates_logs(caplog):
    df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pl.DataFrame({"a": [2, 2], "b": ["y", "y"]})
    with caplog.at_level(logging.WARNING, logger="replay"):
        res = merge_subsets([df1, df2], on_duplicate="drop")
        exp = pl.concat([df1, df2], how="vertical").unique(subset=["a", "b"], keep="first", maintain_order=True)
        assert res.equals(exp)
        assert "duplicate rows" in caplog.text


@pytest.mark.core
def test_polars_subset_for_duplicates():
    df1 = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    df2 = pl.DataFrame({"a": [2, 2], "b": ["z", "w"]})
    with pytest.raises(ValueError, match="duplicate"):
        merge_subsets([df1, df2], on_duplicate="error", subset_for_duplicates=["a"])


@pytest.mark.core
def test_polars_check_columns_and_alignment():
    df1 = pl.DataFrame({"a": [1], "b": [2]})
    df2 = pl.DataFrame({"b": [3], "a": [4]})
    res = merge_subsets([df1, df2], check_columns=True)
    assert res.columns == ["a", "b"]
    assert res.equals(pl.DataFrame({"a": [1, 4], "b": [2, 3]}))


@pytest.mark.core
def test_polars_columns_mismatch_raises():
    df1 = pl.DataFrame({"a": [1], "b": [2]})
    df2 = pl.DataFrame({"a": [3], "c": [4]})
    with pytest.raises(ValueError, match="Columns mismatch"):
        merge_subsets([df1, df2], check_columns=True)


@pytest.mark.core
def test_polars_columns_param_select_subset_when_not_checking():
    df1 = pl.DataFrame({"a": [1], "b": [2], "c": [0]})
    df2 = pl.DataFrame({"b": [3], "a": [4], "c": [9]})
    res = merge_subsets([df1, df2], columns=["a", "b"], check_columns=False, on_duplicate="ignore")
    exp = pl.DataFrame({"a": [1, 4], "b": [2, 3]})
    assert res.equals(exp)


@pytest.mark.spark
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark is not available")
def test_spark_ignore_duplicates(spark):
    df1 = spark.createDataFrame([(1, "x"), (2, "y")], schema=["a", "b"])
    df2 = spark.createDataFrame([(2, "y"), (3, "z")], schema=["a", "b"])
    res = merge_subsets([df1, df2], on_duplicate="ignore")
    assert res.count() == df1.count() + df2.count()


@pytest.mark.spark
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark is not available")
def test_spark_error_on_duplicates(spark):
    df1 = spark.createDataFrame([(1, "x")], schema=["a", "b"])
    df2 = spark.createDataFrame([(1, "x")], schema=["a", "b"])
    with pytest.raises(ValueError, match="Found duplicate rows"):
        merge_subsets([df1, df2], on_duplicate="error")


@pytest.mark.spark
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark is not available")
def test_spark_drop_duplicates_logs(spark, caplog):
    df1 = spark.createDataFrame([(1, "x"), (2, "y")], schema=["a", "b"])
    df2 = spark.createDataFrame([(2, "y"), (2, "y")], schema=["a", "b"])
    with caplog.at_level(logging.WARNING, logger="replay"):
        res = merge_subsets([df1, df2], on_duplicate="drop")
        assert res.dropDuplicates(["a", "b"]).count() == res.count()
        assert "duplicate rows" in caplog.text


@pytest.mark.spark
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark is not available")
def test_spark_subset_for_duplicates(spark):
    df1 = spark.createDataFrame([(1, "x"), (2, "y")], schema=["a", "b"])
    df2 = spark.createDataFrame([(2, "z"), (2, "w")], schema=["a", "b"])
    with pytest.raises(ValueError, match="duplicate"):
        merge_subsets([df1, df2], on_duplicate="error", subset_for_duplicates=["a"])


@pytest.mark.spark
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark is not available")
def test_spark_check_columns_alignment_and_mismatch(spark):
    df1 = spark.createDataFrame([(1, "x")], schema=["a", "b"])
    df2 = spark.createDataFrame([("y", 2)], schema=["b", "a"])  # same set, different order
    res = merge_subsets([df1, df2], check_columns=True)
    assert res.columns == ["a", "b"]

    df3 = spark.createDataFrame([(3,)], schema=["c"])  # mismatch columns
    with pytest.raises(ValueError, match="Columns mismatch"):
        merge_subsets([df1, df3], check_columns=True)


@pytest.mark.spark
@pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark is not available")
def test_spark_columns_param_select_subset_when_not_checking(spark):
    df1 = spark.createDataFrame([(1, "x", 0)], schema=["a", "b", "c"])
    df2 = spark.createDataFrame([(4, "q", 9)], schema=["a", "b", "c"])
    res = merge_subsets([df1, df2], columns=["a", "b"], check_columns=False, on_duplicate="ignore")
    assert res.columns == ["a", "b"]
    assert res.count() == 2
