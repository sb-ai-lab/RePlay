import pandas as pd
import pytest

from replay.preprocessing import Discretizer, GreedyDiscretizingRule, QuantileDiscretizingRule
from replay.utils import PolarsDataFrame, SparkDataFrame


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
def test_descretizer_is_not_fitted(column, discretizing_rule, interactions_100k_pandas):
    rule = discretizing_rule(column, n_bins=20, handle_invalid="error")
    discretizer = Discretizer([rule])
    with pytest.raises(RuntimeError):
        discretizer.transform(interactions_100k_pandas)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
def test_descretizer_partial_fit(column, discretizing_rule, interactions_100k_pandas):
    rule = discretizing_rule(column, n_bins=20)
    discretizer = Discretizer([rule])
    discretizer = discretizer.partial_fit(interactions_100k_pandas)
    with pytest.raises(NotImplementedError):
        discretizer.partial_fit(interactions_100k_pandas)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
def test_descretizer_repeated_fit(column, discretizing_rule, interactions_100k_pandas):
    rule = discretizing_rule(column, n_bins=20)
    discretizer = Discretizer([rule])
    discretizer = discretizer.fit(interactions_100k_pandas)
    assert discretizer is discretizer.fit(interactions_100k_pandas)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
def test_set_wrong_handle_invalid(column, discretizing_rule):
    with pytest.raises(ValueError):
        discretizing_rule(column, n_bins=20, handle_invalid="abc")


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
def test_change_wrong_handle_invalid(column, discretizing_rule):
    rule = discretizing_rule(column, n_bins=20, handle_invalid="error")
    with pytest.raises(ValueError):
        rule.set_handle_invalid("abc")


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
def test_change_wrong_handle_invalid_discretizer(column, discretizing_rule):
    rule = discretizing_rule(column, n_bins=20, handle_invalid="error")
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
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
def test_wrong_dataframe_type(column, discretizing_rule, interactions_100k_pandas):
    rule = discretizing_rule(column, n_bins=20, handle_invalid="error")
    discretizer = Discretizer([rule])
    with pytest.raises(NotImplementedError):
        discretizer.fit(interactions_100k_pandas.values)


@pytest.mark.core
@pytest.mark.parametrize("column", ["item_id"])
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


@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
@pytest.mark.parametrize("handle_invalid", ["keep", "skip", "error"])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("interactions_100k_polars", marks=pytest.mark.core),
        pytest.param("interactions_100k_pandas", marks=pytest.mark.core),
        pytest.param("interactions_100k_spark", marks=pytest.mark.spark),
    ],
)
def test_without_nan(column, discretizing_rule, handle_invalid, dataset, request):
    n_bins = 20
    dataset = request.getfixturevalue(dataset)
    rule = discretizing_rule(column, n_bins=n_bins, handle_invalid=handle_invalid)
    discretizer = Discretizer([rule]).fit(dataset)
    if isinstance(dataset, SparkDataFrame):
        bucketed_data = discretizer.transform(dataset).toPandas()
        n_rows = dataset.count()
    elif isinstance(dataset, PolarsDataFrame):
        bucketed_data = discretizer.transform(dataset).to_pandas()
        n_rows = dataset.shape[0]
    else:
        bucketed_data = discretizer.transform(dataset)
        n_rows = dataset.shape[0]

    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == n_rows
    assert len(bucketed_data.columns) == len(dataset.columns)
    assert all(vc.values > 0.9 * n_rows / (n_bins))
    assert rule.n_bins == n_bins


@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("interactions_100k_polars_with_nan", marks=pytest.mark.core),
        pytest.param("interactions_100k_pandas_with_nan", marks=pytest.mark.core),
        pytest.param("interactions_100k_spark_with_nan", marks=pytest.mark.spark),
    ],
)
def test_nan_default(column, discretizing_rule, dataset, request):
    n_bins = 20
    dataset = request.getfixturevalue(dataset)
    rule = discretizing_rule(column, n_bins=n_bins)
    discretizer = Discretizer([rule]).fit(dataset)
    if isinstance(dataset, SparkDataFrame):
        bucketed_data = discretizer.transform(dataset).toPandas()
        n_rows = dataset.count()
    elif isinstance(dataset, PolarsDataFrame):
        bucketed_data = discretizer.transform(dataset).to_pandas()
        n_rows = dataset.shape[0]
    else:
        bucketed_data = discretizer.transform(dataset)
        n_rows = dataset.shape[0]

    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins + 1
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins
    assert vc.values.sum() == n_rows
    assert len(bucketed_data.columns) == len(dataset.columns)
    assert all(vc.values[:-1] > 0.9 * n_rows / (n_bins))
    assert vc.values[-1] == 200


@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("interactions_100k_polars_with_nan", marks=pytest.mark.core),
        pytest.param("interactions_100k_pandas_with_nan", marks=pytest.mark.core),
        pytest.param("interactions_100k_spark_with_nan", marks=pytest.mark.spark),
    ],
)
def test_nan_skip(column, discretizing_rule, dataset, request):
    n_bins = 20
    dataset = request.getfixturevalue(dataset)
    rule = discretizing_rule(column, n_bins=n_bins, handle_invalid="skip")
    discretizer = Discretizer([rule]).fit(dataset)
    if isinstance(dataset, SparkDataFrame):
        bucketed_data = discretizer.transform(dataset).toPandas()
        n_rows = dataset.count()
    elif isinstance(dataset, PolarsDataFrame):
        bucketed_data = discretizer.transform(dataset).to_pandas()
        n_rows = dataset.shape[0]
    else:
        bucketed_data = discretizer.transform(dataset)
        n_rows = dataset.shape[0]

    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == n_rows - 200
    assert len(bucketed_data.columns) == len(dataset.columns)
    assert all(vc.values > 0.9 * n_rows / (n_bins))


@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("interactions_100k_polars_with_nan", marks=pytest.mark.core),
        pytest.param("interactions_100k_pandas_with_nan", marks=pytest.mark.core),
        pytest.param("interactions_100k_spark_with_nan", marks=pytest.mark.spark),
    ],
)
def test_nan_error(column, discretizing_rule, dataset, request):
    n_bins = 20
    rule = discretizing_rule(column, n_bins=n_bins, handle_invalid="error")
    with pytest.raises(ValueError):
        Discretizer([rule]).fit(request.getfixturevalue(dataset))


@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("interactions_100k_polars", marks=pytest.mark.core),
        pytest.param("interactions_100k_pandas", marks=pytest.mark.core),
        pytest.param("interactions_100k_spark", marks=pytest.mark.spark),
    ],
)
def test_fit_transform(column, discretizing_rule, dataset, request):
    n_bins = 20
    dataset = request.getfixturevalue(dataset)
    rule = discretizing_rule(column, n_bins=n_bins)
    if isinstance(dataset, SparkDataFrame):
        bucketed_data = Discretizer([rule]).fit_transform(dataset).toPandas()
        n_rows = dataset.count()
    elif isinstance(dataset, PolarsDataFrame):
        bucketed_data = Discretizer([rule]).fit_transform(dataset).to_pandas()
        n_rows = dataset.shape[0]
    else:
        bucketed_data = Discretizer([rule]).fit_transform(dataset)
        n_rows = dataset.shape[0]

    vc = bucketed_data[column].value_counts()
    assert len(vc) == n_bins
    assert vc.index.min() == 0
    assert vc.index.max() == n_bins - 1
    assert vc.values.sum() == n_rows
    assert len(bucketed_data.columns) == len(dataset.columns)
    assert all(vc.values > 0.9 * n_rows / (n_bins))


@pytest.mark.parametrize("column", ["item_id"])
@pytest.mark.parametrize("discretizing_rule", [GreedyDiscretizingRule, QuantileDiscretizingRule])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("interactions_100k_polars", marks=pytest.mark.core),
        pytest.param("interactions_100k_pandas", marks=pytest.mark.core),
        pytest.param("interactions_100k_spark", marks=pytest.mark.spark),
    ],
)
def test_discretizer_save_load(column, discretizing_rule, dataset, tmp_path, request):
    path = (tmp_path / "discretizer").resolve()
    n_bins = 20
    rule = discretizing_rule(column, n_bins=n_bins)
    d = Discretizer([rule]).fit(request.getfixturevalue(dataset))
    d.save(path)
    assert rule._bins == Discretizer.load(path).rules[0]._bins
    assert rule._handle_invalid == Discretizer.load(path).rules[0]._handle_invalid
