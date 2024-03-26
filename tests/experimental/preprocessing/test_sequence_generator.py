import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from replay.experimental.preprocessing import SequenceGenerator
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, SparkDataFrame
from tests.preprocessing.conftest import (
    columns,
    columns_target,
    columns_target_list_len,
    schema_target,
    schema_target_list_len,
    simple_dataframe,
    simple_dataframe_additional,
    simple_dataframe_additional_pandas,
    simple_dataframe_array,
    simple_dataframe_array_pandas,
    simple_dataframe_pandas,
    simple_dataframe_target,
    simple_dataframe_target_ordered,
    simple_dataframe_target_ordered_list_len,
    simple_dataframe_target_ordered_list_len_pandas,
    simple_dataframe_target_ordered_pandas,
    simple_dataframe_target_pandas,
)

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf


@pytest.mark.experimental
@pytest.mark.parametrize(
    "groupby_column, transform_columns, len_window, label_prefix", [("user_id", ["item_id", "timestamp"], 5, None)]
)
@pytest.mark.parametrize(
    "dataset, result",
    [
        pytest.param("simple_dataframe", "simple_dataframe_target"),
        pytest.param("simple_dataframe_pandas", "simple_dataframe_target_pandas"),
    ],
)
@pytest.mark.usefixtures("columns_target")
def test_target(groupby_column, transform_columns, len_window, label_prefix, columns_target, dataset, result, request):
    simple_dataframe = request.getfixturevalue(dataset)
    simple_dataframe_target = request.getfixturevalue(result)

    generator = SequenceGenerator(
        groupby_column=groupby_column,
        transform_columns=transform_columns,
        len_window=len_window,
        label_prefix=label_prefix,
    )
    sequences = generator.transform(simple_dataframe)

    if isinstance(sequences, SparkDataFrame):
        sequences = sequences.select(columns_target)
        assert sequences.toPandas().equals(simple_dataframe_target.toPandas())
    else:
        sequences = sequences[columns_target]
        assert sequences.equals(simple_dataframe_target)


@pytest.mark.experimental
@pytest.mark.parametrize(
    "groupby_column, orderby_column, transform_columns, len_window, label_prefix",
    [
        ("user_id", ["user_id", "timestamp"], ["item_id", "timestamp"], 5, None),
    ],
)
@pytest.mark.parametrize(
    "dataset, result",
    [
        pytest.param("simple_dataframe", "simple_dataframe_target_ordered"),
        pytest.param("simple_dataframe_pandas", "simple_dataframe_target_ordered_pandas"),
    ],
)
@pytest.mark.usefixtures("columns_target")
def test_target_ordered(
    groupby_column,
    orderby_column,
    transform_columns,
    len_window,
    label_prefix,
    columns_target,
    dataset,
    result,
    request,
):
    simple_dataframe = request.getfixturevalue(dataset)
    simple_dataframe_target_ordered = request.getfixturevalue(result)

    generator = SequenceGenerator(
        groupby_column=groupby_column,
        orderby_column=orderby_column,
        transform_columns=transform_columns,
        len_window=len_window,
        label_prefix=label_prefix,
    )

    sequences = generator.transform(simple_dataframe)

    if isinstance(sequences, SparkDataFrame):
        sequences = sequences.select(columns_target)
        assert sequences.toPandas().equals(simple_dataframe_target_ordered.toPandas())
    else:
        sequences = sequences[columns_target]
        assert sequences.equals(simple_dataframe_target_ordered)


@pytest.mark.experimental
@pytest.mark.parametrize(
    "groupby_column, orderby_column, transform_columns, len_window, label_prefix, get_list_len",
    [
        ("user_id", ["user_id", "timestamp"], ["item_id", "timestamp"], 5, None, True),
    ],
)
@pytest.mark.parametrize(
    "dataset, result",
    [
        pytest.param("simple_dataframe", "simple_dataframe_target_ordered_list_len"),
        pytest.param("simple_dataframe_pandas", "simple_dataframe_target_ordered_list_len_pandas"),
    ],
)
@pytest.mark.usefixtures("columns_target_list_len")
def test_target_ordered_list_len(
    groupby_column,
    orderby_column,
    transform_columns,
    len_window,
    label_prefix,
    get_list_len,
    columns_target_list_len,
    dataset,
    result,
    request,
):
    simple_dataframe = request.getfixturevalue(dataset)
    simple_dataframe_target_ordered_list_len = request.getfixturevalue(result)

    generator = SequenceGenerator(
        groupby_column=groupby_column,
        orderby_column=orderby_column,
        transform_columns=transform_columns,
        len_window=len_window,
        get_list_len=get_list_len,
        label_prefix=label_prefix,
    )

    sequences = generator.transform(simple_dataframe)

    if isinstance(sequences, SparkDataFrame):
        sequences = sequences.select(columns_target_list_len)
        assert sequences.toPandas().equals(simple_dataframe_target_ordered_list_len.toPandas())
    else:
        sequences = sequences[columns_target_list_len]
        assert sequences.equals(simple_dataframe_target_ordered_list_len)


@pytest.mark.experimental
@pytest.mark.parametrize(
    "user_column, item_column, time_column, len_window",
    [("user_id", "item_id", "timestamp", 5), ("usr", "item_id", "timestamp", 5), ("usr", "itms", "time", 5)],
)
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("simple_dataframe"),
        pytest.param("simple_dataframe_pandas"),
    ],
)
def test_sequence_generator_executor(user_column, item_column, time_column, len_window, dataset, request):
    simple_dataframe = request.getfixturevalue(dataset)

    if isinstance(simple_dataframe, SparkDataFrame):
        test_simple_dataframe = (
            simple_dataframe.withColumnRenamed("user_id", user_column)
            .withColumnRenamed("item_id", item_column)
            .withColumnRenamed("timestamp", time_column)
        )
    else:
        test_simple_dataframe = simple_dataframe.rename(
            columns={"user_id": user_column, "item_id": item_column, "timestamp": time_column}
        )

    generator = SequenceGenerator(
        groupby_column=user_column, transform_columns=[item_column, time_column], len_window=len_window
    )

    sequences = generator.transform(test_simple_dataframe)
    columns = sequences.columns
    assert user_column in columns
    assert f"{item_column}_list" in columns
    assert f"{time_column}_list" in columns


@pytest.mark.experimental
@pytest.mark.parametrize("user_column, len_window", [("user_id", 5)])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("simple_dataframe"),
        pytest.param("simple_dataframe_pandas"),
    ],
)
def test_default_parameters(user_column, len_window, dataset, request):
    simple_dataframe = request.getfixturevalue(dataset)

    generator = SequenceGenerator(groupby_column=user_column, len_window=len_window)
    sequences = generator.transform(simple_dataframe)
    columns = sequences.columns

    if isinstance(simple_dataframe, SparkDataFrame):
        assert 3 == sequences.select(user_column).distinct().count()
    else:
        assert 3 == len(sequences[user_column].unique())
    assert "item_id_list" in columns
    assert "label_item_id" in columns
    assert "timestamp_list" in columns
    assert "label_timestamp" in columns


@pytest.mark.experimental
@pytest.mark.parametrize(
    "user_column, item_column, time_column, len_window",
    [
        ("user_id", "item_id", "timestamp", 1),
        ("user_id", "item_id", "timestamp", 3),
        ("user_id", "item_id", "timestamp", 5),
    ],
)
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("simple_dataframe"),
        pytest.param("simple_dataframe_pandas"),
    ],
)
def test_window_parameters(user_column, item_column, time_column, len_window, dataset, request):
    simple_dataframe = request.getfixturevalue(dataset)

    generator = SequenceGenerator(
        groupby_column=user_column, transform_columns=[item_column, time_column], len_window=len_window
    )
    sequences = generator.transform(simple_dataframe)

    if isinstance(simple_dataframe, SparkDataFrame):
        agg_sequences = sequences.withColumn("LEN", sf.size(f"{item_column}_list"))
        max_len = agg_sequences.agg(sf.max("LEN")).first()[0]
    else:
        max_len = sequences[f"{item_column}_list"].str.len().max()

    assert len_window == max_len


@pytest.mark.experimental
@pytest.mark.parametrize(
    "user_column, item_column, len_window", [("user_id", "item_id", 5), ("user_id", ["item_id"], 5)]
)
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("simple_dataframe"),
        pytest.param("simple_dataframe_pandas"),
    ],
)
def test_only_item_processing(user_column, item_column, len_window, dataset, request):
    simple_dataframe = request.getfixturevalue(dataset)

    generator = SequenceGenerator(groupby_column=user_column, transform_columns=item_column, len_window=len_window)
    sequences = generator.transform(simple_dataframe)
    columns = sequences.columns
    item_column_str = item_column if isinstance(item_column, str) else item_column[0]
    assert f"{item_column_str}_list" in columns
    assert f"label_{item_column_str}" in columns
    assert "user_id" in columns
    assert "timestamp" not in columns


@pytest.mark.experimental
@pytest.mark.parametrize(
    "user_column, time_column, sequence_prefix, sequence_suffix, label_prefix, label_suffix, len_window",
    [("user_id", "timestamp", "preffix_augm_", "_suffix_augm", "preffix_label_", "_suffix_label", 5)],
)
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("simple_dataframe"),
        pytest.param("simple_dataframe_pandas"),
    ],
)
def test_return_column_names(
    user_column, time_column, sequence_prefix, sequence_suffix, label_prefix, label_suffix, len_window, dataset, request
):
    simple_dataframe = request.getfixturevalue(dataset)

    generator = SequenceGenerator(
        groupby_column=user_column,
        transform_columns=time_column,
        sequence_prefix=sequence_prefix,
        sequence_suffix=sequence_suffix,
        label_prefix=label_prefix,
        label_suffix=label_suffix,
        len_window=len_window,
    )
    sequences = generator.transform(simple_dataframe)
    columns = sequences.columns
    assert f"preffix_augm_{time_column}_suffix_augm" in columns
    assert f"preffix_label_{time_column}_suffix_label" in columns
    assert "user_id" in columns
    assert "item_id" not in columns


@pytest.mark.experimental
@pytest.mark.usefixtures("columns")
@pytest.mark.parametrize(
    "is_spark",
    [
        pytest.param(True),
        pytest.param(False),
    ],
)
def test_with_string_values(is_spark, request, columns):
    string_data = [
        ("1", "2", "19842"),
        ("1", "4", "19844"),
        ("1", "3", "19843"),
        ("1", "5", "19845"),
        ("1", "6", "19846"),
        ("1", "7", "19847"),
        ("2", "1", "19841"),
        ("2", "2", "19842"),
        ("2", "3", "19843"),
        ("2", "4", "19844"),
        ("3", "10", "19844"),
        ("4", "11", "19843"),
        ("4", "12", "19845"),
        ("1", "1", "19841"),
    ]

    if is_spark is True:
        spark = request.getfixturevalue("spark")
        string_simple_dataframe = spark.createDataFrame(string_data, schema=columns)
    else:
        string_simple_dataframe = PandasDataFrame(string_data, columns=columns)

    generator = SequenceGenerator(groupby_column="user_id", transform_columns=["timestamp", "item_id"])
    _ = generator.transform(string_simple_dataframe)


@pytest.mark.experimental
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("simple_dataframe_additional"),
        pytest.param("simple_dataframe_additional_pandas"),
    ],
)
def test_groupby_multiple_columns(dataset, request):
    simple_dataframe_additional = request.getfixturevalue(dataset)

    generator = SequenceGenerator(
        groupby_column=["user_id", "other_column"], transform_columns=["item_id", "timestamp"]
    )
    sequences = generator.transform(simple_dataframe_additional)
    columns = sequences.columns
    if isinstance(sequences, SparkDataFrame):
        assert sequences.select(sf.countDistinct(*["user_id", "other_column"])).first()[0] == 4
    else:
        assert len(pd.unique(sequences[["user_id", "other_column"]].values.ravel("K"))) == 4
    assert "item_id_list" in columns
    assert "label_item_id" in columns
    assert "timestamp_list" in columns
    assert "label_timestamp" in columns
    assert "user_id" in columns
    assert "other_column" in columns


@pytest.mark.experimental
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param("simple_dataframe_array"),
        pytest.param("simple_dataframe_array_pandas"),
    ],
)
def test_array_columns(dataset, request):
    simple_dataframe_array = request.getfixturevalue(dataset)

    generator = SequenceGenerator(groupby_column=["user_id"], transform_columns=["item_id", "timestamp"])
    sequences = generator.transform(simple_dataframe_array)
    columns = sequences.columns
    assert "item_id_list" in columns
    assert "label_item_id" in columns
    assert "timestamp_list" in columns
    assert "label_timestamp" in columns
    assert "user_id" in columns
