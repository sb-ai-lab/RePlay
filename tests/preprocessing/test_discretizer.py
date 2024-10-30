import numpy as np
import pandas as pd
import polars as pl
import pytest

from replay.preprocessing import Discretizer, GreedyDiscretizingRule, QuantileDiscretizingRule


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
def test_greedy_descretizer_is_not_fitted(column, interactions_100k_pandas):
    rule = GreedyDiscretizingRule(column, n_bins=20, handle_invalid="error")
    discretizer = Discretizer([rule])
    with pytest.raises(RuntimeError):
        discretizer.transform(interactions_100k_pandas)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
def test_quantile_descretizer_is_not_fitted(column, interactions_100k_pandas):
    rule = QuantileDiscretizingRule(column, n_bins=20, handle_invalid="error")
    discretizer = Discretizer([rule])
    with pytest.raises(RuntimeError):
        discretizer.transform(interactions_100k_pandas)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_descretizer_partial_fit(column, interactions_100k_pandas):
    rule = GreedyDiscretizingRule(column, n_bins=20)
    discretizer = Discretizer([rule])
    discretizer = discretizer.partial_fit(interactions_100k_pandas)
    with pytest.raises(NotImplementedError):
        discretizer.partial_fit(interactions_100k_pandas)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_descretizer_partial_fit(column, interactions_100k_pandas):
    rule = QuantileDiscretizingRule(column, n_bins=20)
    discretizer = Discretizer([rule])
    discretizer = discretizer.partial_fit(interactions_100k_pandas)
    with pytest.raises(NotImplementedError):
        discretizer.partial_fit(interactions_100k_pandas)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_descretizer_repeated_fit(column, interactions_100k_pandas):
    rule = GreedyDiscretizingRule(column, n_bins=20)
    discretizer = Discretizer([rule])
    discretizer = discretizer.fit(interactions_100k_pandas)
    assert discretizer is discretizer.fit(interactions_100k_pandas)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_descretizer_repeated_fit(column, interactions_100k_pandas):
    rule = QuantileDiscretizingRule(column, n_bins=20)
    discretizer = Discretizer([rule])
    discretizer = discretizer.fit(interactions_100k_pandas)
    assert discretizer is discretizer.fit(interactions_100k_pandas)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_set_wrong_handle_invalid_greedy_rule(column):
    with pytest.raises(ValueError):
        GreedyDiscretizingRule(column, n_bins=20, handle_invalid="abc")


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
def test_change_wrong_handle_invalid_greedy_rule(column):
    rule = GreedyDiscretizingRule(column, n_bins=20, handle_invalid="error")
    with pytest.raises(ValueError):
        rule.set_handle_invalid("abc")


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_set_wrong_handle_invalid_quantile_rule(column):
    with pytest.raises(ValueError):
        QuantileDiscretizingRule(column, n_bins=20, handle_invalid="abc")


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
def test_change_wrong_handle_invalid_quantile_rule(column):
    rule = QuantileDiscretizingRule(column, n_bins=20, handle_invalid="error")
    with pytest.raises(ValueError):
        rule.set_handle_invalid("abc")


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
def test_change_wrong_handle_invalid_discretizer(column):
    rule = GreedyDiscretizingRule(column, n_bins=20, handle_invalid="error")
    discretizer = Discretizer([rule])
    with pytest.raises(ValueError):
        discretizer.set_handle_invalid({"item_id": "abc"})


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("column2", ["user_id"])
def test_change_wrong_column_in_handle_invalid_discretizer(column, column2):
    rule1 = QuantileDiscretizingRule(column, n_bins=20, handle_invalid="error")
    rule2 = GreedyDiscretizingRule(column2, n_bins=20, handle_invalid="error")
    discretizer = Discretizer([rule1, rule2])
    discretizer.set_handle_invalid({"item_id": "skip", "user_id": "skip"})

    with pytest.raises(ValueError):
        discretizer.set_handle_invalid({"item_id": "skip", "aaa": "skip"})

    with pytest.raises(ValueError):
        discretizer.set_handle_invalid({"aaa": "skip", "user_id": "skip"})


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_wrong_dataframe_type_greedy_discretizer(column, interactions_100k_pandas):
    rule = GreedyDiscretizingRule(column, n_bins=20, handle_invalid="error")
    discretizer = Discretizer([rule])
    with pytest.raises(NotImplementedError):
        discretizer.fit(interactions_100k_pandas.values)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_wrong_dataframe_type_quantile_discretizer(column, interactions_100k_pandas):
    rule = QuantileDiscretizingRule(column, n_bins=20, handle_invalid="error")
    discretizer = Discretizer([rule])
    with pytest.raises(NotImplementedError):
        discretizer.fit(interactions_100k_pandas.values)


# Test greedy
@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
def test_greedy_spark_without_nan_default(column, interactions_100k_spark):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(interactions_100k_spark)
    bucketed_data = discretizer.transform(interactions_100k_spark).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == rule.n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == rule.n_bins - 1
    assert vc.values.sum() == interactions_100k_spark.count()
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values > 0.9 * interactions_100k_spark.count() / (rule.n_bins))


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
def test_greedy_spark_without_nan_skip(column, interactions_100k_spark):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(interactions_100k_spark)
    bucketed_data = discretizer.transform(interactions_100k_spark).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == interactions_100k_spark.count()
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values > 0.9 * interactions_100k_spark.count() / (n_bins))


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
def test_greedy_spark_without_nan_error(column, interactions_100k_spark):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    discretizer = Discretizer([rule]).fit(interactions_100k_spark)
    bucketed_data = discretizer.transform(interactions_100k_spark).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == interactions_100k_spark.count()
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values > 0.9 * interactions_100k_spark.count() / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_few_rows(column, interactions_100k_pandas):
    n_bins = 20
    n_rows = 5
    few_rows_data = interactions_100k_pandas.copy().iloc[:n_rows]
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(few_rows_data)
    bucketed_data = discretizer.transform(few_rows_data)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_rows
    assert vc.index.min() == 0
    assert vc.index.max() == n_rows - 1
    assert vc.values.sum() == len(few_rows_data)
    assert len(bucketed_data.columns) == len(few_rows_data.columns)
    assert all(vc.values == 1)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
def test_greedy_lots_of_repetitions_1(column):
    data = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2], columns=["item_id"])
    rule = GreedyDiscretizingRule(column, n_bins=20)
    discretizer = Discretizer([rule]).fit(data)
    bucketed_data = discretizer.transform(data)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == 2
    assert vc.index.min() == 0
    assert vc.index.max() == 1
    assert vc.values.sum() == len(data)
    assert len(bucketed_data.columns) == len(data.columns)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
def test_greedy_lots_of_repetitions_2(column):
    data = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2], columns=["item_id"])
    rule = GreedyDiscretizingRule(column, n_bins=20, min_data_in_bin=12)
    discretizer = Discretizer([rule]).fit(data)
    bucketed_data = discretizer.transform(data)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == 1
    assert vc.index.min() == 0
    assert vc.values.sum() == len(data)
    assert len(bucketed_data.columns) == len(data.columns)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
def test_greedy_lots_of_repetitions_3(column):
    data = pd.DataFrame([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8], columns=["item_id"])
    rule = GreedyDiscretizingRule(column, n_bins=5)
    discretizer = Discretizer([rule]).fit(data)
    bucketed_data = discretizer.transform(data)
    vc = bucketed_data[column].value_counts()

    rule = GreedyDiscretizingRule(column, n_bins=5, min_data_in_bin=0)
    discretizer = Discretizer([rule]).fit(data)
    bucketed_data2 = discretizer.transform(data)
    vc2 = bucketed_data2[column].value_counts()
    assert len(vc) == len(vc2) == 5
    assert vc.index.min() == vc2.index.min() == 0
    assert vc.index.max() == vc2.index.max() == 4
    assert vc.values.sum() == vc2.values.sum() == len(data)
    assert len(bucketed_data.columns) == len(bucketed_data2.columns) == len(data.columns)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_pandas_without_nan_default(column, interactions_100k_pandas):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(interactions_100k_pandas)
    bucketed_data = discretizer.transform(interactions_100k_pandas)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_pandas_without_nan_skip(column, interactions_100k_pandas):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(interactions_100k_pandas)
    bucketed_data = discretizer.transform(interactions_100k_pandas)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_pandas_without_nan_error(column, interactions_100k_pandas):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    discretizer = Discretizer([rule]).fit(interactions_100k_pandas)
    bucketed_data = discretizer.transform(interactions_100k_pandas)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / n_bins)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_polars")
def test_greedy_polars_without_nan_default(column, interactions_100k_polars):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(interactions_100k_polars)
    bucketed_data = discretizer.transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_polars)
    assert len(bucketed_data.columns) == len(interactions_100k_polars.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_polars) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_polars")
def test_greedy_polars_without_nan_skip(column, interactions_100k_polars):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(interactions_100k_polars)
    bucketed_data = discretizer.transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_polars)
    assert len(bucketed_data.columns) == len(interactions_100k_polars.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_polars) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_polars")
def test_greedy_polars_without_nan_error(column, interactions_100k_polars):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    discretizer = Discretizer([rule]).fit(interactions_100k_polars)
    bucketed_data = discretizer.transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_polars)
    assert len(bucketed_data.columns) == len(interactions_100k_polars.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_polars) / n_bins)


# Test greedy with NaNs
@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
@pytest.mark.usefixtures("spark")
def test_greedy_spark_nan_default(column, interactions_100k_spark, spark):
    n_bins = 20
    interactions_100k_pandas_with_nan = interactions_100k_spark.toPandas()
    nan_index = np.random.choice(interactions_100k_pandas_with_nan.index, size=200, replace=False)
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    interactions_100k_spark_with_nan = spark.createDataFrame(
        interactions_100k_pandas_with_nan.values.tolist(), schema=["user_id", "item_id"]
    )
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(interactions_100k_spark_with_nan)
    bucketed_data = discretizer.transform(interactions_100k_spark_with_nan).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins + 1
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins
    assert vc.values.sum() == interactions_100k_spark.count()
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values[:-1] > 0.9 * interactions_100k_spark.count() / (n_bins))
    assert vc.values[-1] == 200


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
@pytest.mark.usefixtures("spark")
def test_greedy_spark_nan_skip(column, interactions_100k_spark, spark):
    n_bins = 20
    interactions_100k_pandas_with_nan = interactions_100k_spark.toPandas()
    nan_index = np.random.choice(interactions_100k_pandas_with_nan.index, size=200, replace=False)
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    interactions_100k_spark_with_nan = spark.createDataFrame(
        interactions_100k_pandas_with_nan.values.tolist(), schema=["user_id", "item_id"]
    )
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(interactions_100k_spark_with_nan)
    bucketed_data = discretizer.transform(interactions_100k_spark_with_nan).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == interactions_100k_spark.count() - 200
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values > 0.9 * interactions_100k_spark.count() / (n_bins))


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
@pytest.mark.usefixtures("spark")
def test_greedy_spark_nan_error(column, interactions_100k_spark, spark):
    n_bins = 20
    interactions_100k_pandas_with_nan = interactions_100k_spark.toPandas()
    nan_index = np.random.choice(interactions_100k_pandas_with_nan.index, size=200, replace=False)
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    interactions_100k_spark_with_nan = spark.createDataFrame(
        interactions_100k_pandas_with_nan.values.tolist(), schema=["user_id", "item_id"]
    )
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    with pytest.raises(ValueError):
        Discretizer([rule]).fit(interactions_100k_spark_with_nan)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_pandas_nan_default(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(interactions_100k_pandas_with_nan)
    bucketed_data = discretizer.transform(interactions_100k_pandas_with_nan)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins + 1
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values[:-1] > 0.9 * len(interactions_100k_pandas) / (n_bins))
    assert vc.values[-1] == 200


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_pandas_nan_skip(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(interactions_100k_pandas_with_nan)
    bucketed_data = discretizer.transform(interactions_100k_pandas_with_nan)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas) - 200
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_pandas_nan_error(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    with pytest.raises(ValueError):
        Discretizer([rule]).fit(interactions_100k_pandas_with_nan)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_polars_nan_default(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    interactions_100k_polars = pl.from_pandas(interactions_100k_pandas_with_nan)
    discretizer = Discretizer([rule]).fit(interactions_100k_polars)
    bucketed_data = discretizer.transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins + 1
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values[:-1] > 0.9 * len(interactions_100k_pandas) / (n_bins))
    assert vc.values[-1] == 200


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_polars_nan_skip(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    interactions_100k_polars = pl.from_pandas(interactions_100k_pandas_with_nan)
    discretizer = Discretizer([rule]).fit(interactions_100k_polars)
    bucketed_data = discretizer.transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas) - 200
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_polars_nan_error(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = GreedyDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    interactions_100k_polars = pl.from_pandas(interactions_100k_pandas_with_nan)
    with pytest.raises(ValueError):
        Discretizer([rule]).fit(interactions_100k_polars)


# Test quantile
@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
def test_quantile_spark_without_nan_default(column, interactions_100k_spark):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(interactions_100k_spark)
    bucketed_data = discretizer.transform(interactions_100k_spark).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == rule.n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == rule.n_bins - 1
    assert vc.values.sum() == interactions_100k_spark.count()
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values > 0.9 * interactions_100k_spark.count() / (rule.n_bins))


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
def test_quantile_spark_without_nan_skip(column, interactions_100k_spark):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(interactions_100k_spark)
    bucketed_data = discretizer.transform(interactions_100k_spark).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == interactions_100k_spark.count()
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values > 0.9 * interactions_100k_spark.count() / (n_bins))


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
def test_quantile_spark_without_nan_error(column, interactions_100k_spark):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    discretizer = Discretizer([rule]).fit(interactions_100k_spark)
    bucketed_data = discretizer.transform(interactions_100k_spark).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == interactions_100k_spark.count()
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values > 0.9 * interactions_100k_spark.count() / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_pandas_without_nan_default(column, interactions_100k_pandas):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(interactions_100k_pandas)
    bucketed_data = discretizer.transform(interactions_100k_pandas)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_pandas_without_nan_skip(column, interactions_100k_pandas):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(interactions_100k_pandas)
    bucketed_data = discretizer.transform(interactions_100k_pandas)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_pandas_without_nan_error(column, interactions_100k_pandas):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    discretizer = Discretizer([rule]).fit(interactions_100k_pandas)
    bucketed_data = discretizer.transform(interactions_100k_pandas)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / n_bins)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_polars")
def test_quantile_polars_without_nan_default(column, interactions_100k_polars):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(interactions_100k_polars)
    bucketed_data = discretizer.transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_polars)
    assert len(bucketed_data.columns) == len(interactions_100k_polars.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_polars) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_polars")
def test_quantile_polars_without_nan_skip(column, interactions_100k_polars):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(interactions_100k_polars)
    bucketed_data = discretizer.transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_polars)
    assert len(bucketed_data.columns) == len(interactions_100k_polars.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_polars) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_polars")
def test_quantile_polars_without_nan_error(column, interactions_100k_polars):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    discretizer = Discretizer([rule]).fit(interactions_100k_polars)
    bucketed_data = discretizer.transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_polars)
    assert len(bucketed_data.columns) == len(interactions_100k_polars.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_polars) / n_bins)


# Test quantile with NaNs
@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
@pytest.mark.usefixtures("spark")
def test_quantile_spark_nan_default(column, interactions_100k_spark, spark):
    n_bins = 20
    interactions_100k_pandas_with_nan = interactions_100k_spark.toPandas()
    nan_index = np.random.choice(interactions_100k_pandas_with_nan.index, size=200, replace=False)
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    interactions_100k_spark_with_nan = spark.createDataFrame(
        interactions_100k_pandas_with_nan.values.tolist(), schema=["user_id", "item_id"]
    )
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(interactions_100k_spark_with_nan)
    bucketed_data = discretizer.transform(interactions_100k_spark_with_nan).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins + 1
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins
    assert vc.values.sum() == interactions_100k_spark.count()
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values[:-1] > 0.9 * interactions_100k_spark.count() / (n_bins))
    assert vc.values[-1] == 200


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
@pytest.mark.usefixtures("spark")
def test_quantile_spark_nan_skip(column, interactions_100k_spark, spark):
    n_bins = 20
    interactions_100k_pandas_with_nan = interactions_100k_spark.toPandas()
    nan_index = np.random.choice(interactions_100k_pandas_with_nan.index, size=200, replace=False)
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    interactions_100k_spark_with_nan = spark.createDataFrame(
        interactions_100k_pandas_with_nan.values.tolist(), schema=["user_id", "item_id"]
    )
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(interactions_100k_spark_with_nan)
    bucketed_data = discretizer.transform(interactions_100k_spark_with_nan).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == interactions_100k_spark.count() - 200
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values > 0.9 * interactions_100k_spark.count() / (n_bins))


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
@pytest.mark.usefixtures("spark")
def test_quantile_spark_nan_error(column, interactions_100k_spark, spark):
    n_bins = 20
    interactions_100k_pandas_with_nan = interactions_100k_spark.toPandas()
    nan_index = np.random.choice(interactions_100k_pandas_with_nan.index, size=200, replace=False)
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    interactions_100k_spark_with_nan = spark.createDataFrame(
        interactions_100k_pandas_with_nan.values.tolist(), schema=["user_id", "item_id"]
    )
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    with pytest.raises(ValueError):
        Discretizer([rule]).fit(interactions_100k_spark_with_nan)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_pandas_nan_default(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(interactions_100k_pandas_with_nan)
    bucketed_data = discretizer.transform(interactions_100k_pandas_with_nan)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins + 1
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values[:-1] > 0.9 * len(interactions_100k_pandas) / (n_bins))
    assert vc.values[-1] == 200


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_pandas_nan_skip(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(interactions_100k_pandas_with_nan)
    bucketed_data = discretizer.transform(interactions_100k_pandas_with_nan)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas) - 200
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_pandas_nan_error(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    with pytest.raises(ValueError):
        Discretizer([rule]).fit(interactions_100k_pandas_with_nan)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_polars_nan_default(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    interactions_100k_polars = pl.from_pandas(interactions_100k_pandas_with_nan)
    discretizer = Discretizer([rule]).fit(interactions_100k_polars)
    bucketed_data = discretizer.transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins + 1
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values[:-1] > 0.9 * len(interactions_100k_pandas) / (n_bins))
    assert vc.values[-1] == 200


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_polars_nan_skip(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="skip")
    interactions_100k_polars = pl.from_pandas(interactions_100k_pandas_with_nan)
    discretizer = Discretizer([rule]).fit(interactions_100k_polars)
    bucketed_data = discretizer.transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas) - 200
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_polars_nan_error(column, interactions_100k_pandas):
    n_bins = 20
    nan_index = np.random.choice(interactions_100k_pandas.index, size=200, replace=False)
    interactions_100k_pandas_with_nan = interactions_100k_pandas.copy()
    interactions_100k_pandas_with_nan.loc[nan_index, column] = np.nan
    rule = QuantileDiscretizingRule(column, n_bins=n_bins, handle_invalid="error")
    interactions_100k_polars = pl.from_pandas(interactions_100k_pandas_with_nan)
    with pytest.raises(ValueError):
        Discretizer([rule]).fit(interactions_100k_polars)


# Test fit_transform
@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
def test_greedy_spark_fit_transform(column, interactions_100k_spark):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    bucketed_data = Discretizer([rule]).fit_transform(interactions_100k_spark).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == interactions_100k_spark.count()
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values > 0.9 * interactions_100k_spark.count() / (n_bins))


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
def test_quantile_spark_fit_transform(column, interactions_100k_spark):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    bucketed_data = Discretizer([rule]).fit_transform(interactions_100k_spark).toPandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == interactions_100k_spark.count()
    assert len(bucketed_data.columns) == len(interactions_100k_spark.toPandas().columns)
    assert all(vc.values > 0.9 * interactions_100k_spark.count() / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_pandas_fit_transform(column, interactions_100k_pandas):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    bucketed_data = Discretizer([rule]).fit_transform(interactions_100k_pandas)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_pandas_fit_transform(column, interactions_100k_pandas):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    bucketed_data = Discretizer([rule]).fit_transform(interactions_100k_pandas)
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_pandas)
    assert len(bucketed_data.columns) == len(interactions_100k_pandas.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_pandas) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_polars")
def test_greedy_polars_fit_transform(column, interactions_100k_polars):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    bucketed_data = Discretizer([rule]).fit_transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_polars)
    assert len(bucketed_data.columns) == len(interactions_100k_polars.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_polars) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_polars")
def test_quantile_polars_fit_transform(column, interactions_100k_polars):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    bucketed_data = Discretizer([rule]).fit_transform(interactions_100k_polars).to_pandas()
    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == len(interactions_100k_polars)
    assert len(bucketed_data.columns) == len(interactions_100k_polars.columns)
    assert all(vc.values > 0.9 * len(interactions_100k_polars) / (n_bins))


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_polars")
def test_quantile_discretizer_save_load_polars(column, interactions_100k_polars):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    d = Discretizer([rule]).fit(interactions_100k_polars)
    d.save("./")
    assert rule._bins == Discretizer.load("./Discretizer.replay").rules[0]._bins
    assert rule._handle_invalid == Discretizer.load("./Discretizer.replay").rules[0]._handle_invalid


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_polars")
def test_greedy_discretizer_save_load_polars(column, interactions_100k_polars):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    d = Discretizer([rule]).fit(interactions_100k_polars)
    d.save("./")
    assert rule._bins == Discretizer.load("./Discretizer.replay").rules[0]._bins
    assert rule._handle_invalid == Discretizer.load("./Discretizer.replay").rules[0]._handle_invalid


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_greedy_discretizer_save_load_pandas(column, interactions_100k_pandas):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    d = Discretizer([rule]).fit(interactions_100k_pandas)
    d.save("./")
    assert rule._bins == Discretizer.load("./Discretizer.replay").rules[0]._bins
    assert rule._handle_invalid == Discretizer.load("./Discretizer.replay").rules[0]._handle_invalid


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_pandas")
def test_quantile_discretizer_save_load_pandas(column, interactions_100k_pandas):
    n_bins = 20
    rule = QuantileDiscretizingRule(column, n_bins=n_bins)
    d = Discretizer([rule]).fit(interactions_100k_pandas)
    d.save("./")
    assert rule._bins == Discretizer.load("./Discretizer.replay").rules[0]._bins
    assert rule._handle_invalid == Discretizer.load("./Discretizer.replay").rules[0]._handle_invalid


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
def test_greedy_discretizer_save_load_spark(column, interactions_100k_spark):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    d = Discretizer([rule]).fit(interactions_100k_spark)
    d.save("./")
    assert rule._bins == Discretizer.load("./Discretizer.replay").rules[0]._bins
    assert rule._handle_invalid == Discretizer.load("./Discretizer.replay").rules[0]._handle_invalid


@pytest.mark.spark
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.usefixtures("interactions_100k_spark")
def test_greedy_discretizer_save_load_spark(column, interactions_100k_spark):
    n_bins = 20
    rule = GreedyDiscretizingRule(column, n_bins=n_bins)
    d = Discretizer([rule]).fit(interactions_100k_spark)
    d.save("./")
    assert rule._bins == Discretizer.load("./Discretizer.replay").rules[0]._bins
    assert rule._handle_invalid == Discretizer.load("./Discretizer.replay").rules[0]._handle_invalid
