import pytest
from pyspark.sql import DataFrame as SparkDataFrame

from replay.experimental.preprocessing import Padder
from tests.preprocessing.conftest import *


@pytest.mark.parametrize("pad_columns, padding_value, array_size", [("item_id", 0, 5)])
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe", "dataframe_only_item"),
        ("dataframe_pandas", "dataframe_only_item_pandas"),
    ],
)
def test_padder_only_item(pad_columns, padding_value, array_size, request, dataset, result):
    dataframe = request.getfixturevalue(dataset)
    dataframe_only_item = request.getfixturevalue(result)
    is_spark = isinstance(dataframe, SparkDataFrame)

    padder = Padder(pad_columns=pad_columns, padding_value=padding_value, array_size=array_size)
    padder_interactions = padder.transform(dataframe)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_only_item.toPandas())
    else:
        assert padder_interactions.equals(dataframe_only_item)


@pytest.mark.parametrize("pad_columns, padding_value, array_size", [(["item_id", "timestamp"], 0, 5)])
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe", "dataframe_two_columns_zeros"),
        ("dataframe_pandas", "dataframe_two_columns_zeros_pandas"),
    ],
)
def test_padder_two_columns_same_value(pad_columns, padding_value, array_size, dataset, result, request):
    dataframe = request.getfixturevalue(dataset)
    dataframe_two_columns_zeros = request.getfixturevalue(result)
    is_spark = isinstance(dataframe, SparkDataFrame)

    padder = Padder(pad_columns=pad_columns, padding_value=padding_value, array_size=array_size)
    padder_interactions = padder.transform(dataframe)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_two_columns_zeros.toPandas())
    else:
        assert padder_interactions.equals(dataframe_two_columns_zeros)


@pytest.mark.parametrize("pad_columns, padding_value, array_size", [(["item_id", "timestamp"], [0, -1], 5)])
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe", "dataframe_two_columns"),
        ("dataframe_pandas", "dataframe_two_columns_pandas"),
    ],
)
def test_padder_two_columns(pad_columns, padding_value, array_size, dataset, result, request):
    dataframe = request.getfixturevalue(dataset)
    dataframe_two_columns = request.getfixturevalue(result)
    is_spark = isinstance(dataframe, SparkDataFrame)

    padder = Padder(pad_columns=pad_columns, padding_value=padding_value, array_size=array_size)
    padder_interactions = padder.transform(dataframe)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_two_columns.toPandas())
    else:
        assert padder_interactions.equals(dataframe_two_columns)


@pytest.mark.parametrize(
    "pad_columns, padding_value, array_size, cut_side", [(["item_id", "timestamp"], [0, -1], 5, "left")]
)
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe", "dataframe_two_columns_cut_left"),
        ("dataframe_pandas", "dataframe_two_columns_cut_left_pandas"),
    ],
)
def test_padder_two_columns_cut_left(pad_columns, padding_value, array_size, cut_side, dataset, result, request):
    dataframe = request.getfixturevalue(dataset)
    dataframe_two_columns_cut_left = request.getfixturevalue(result)
    is_spark = isinstance(dataframe, SparkDataFrame)

    padder = Padder(pad_columns=pad_columns, padding_value=padding_value, array_size=array_size, cut_side=cut_side)

    padder_interactions = padder.transform(dataframe)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_two_columns_cut_left.toPandas())
    else:
        assert padder_interactions.equals(dataframe_two_columns_cut_left)


@pytest.mark.parametrize(
    "pad_columns, padding_value, array_size, cut_array", [(["item_id", "timestamp"], [0, -1], 5, False)]
)
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe", "dataframe_two_columns_no_cut"),
        ("dataframe_pandas", "dataframe_two_columns_no_cut_pandas"),
    ],
)
def test_padder_two_columns_no_cut(pad_columns, padding_value, array_size, cut_array, dataset, result, request):
    dataframe = request.getfixturevalue(dataset)
    dataframe_two_columns_no_cut = request.getfixturevalue(result)
    is_spark = isinstance(dataframe, SparkDataFrame)

    padder = Padder(pad_columns=pad_columns, padding_value=padding_value, array_size=array_size, cut_array=cut_array)

    padder_interactions = padder.transform(dataframe)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_two_columns_no_cut.toPandas())
    else:
        assert padder_interactions.equals(dataframe_two_columns_no_cut)


@pytest.mark.parametrize("pad_columns, padding_value, array_size, padding_side", [("item_id", 0, 5, "left")])
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe", "dataframe_only_item_left"),
        ("dataframe_pandas", "dataframe_only_item_left_pandas"),
    ],
)
def test_padder_only_item_left(pad_columns, padding_value, array_size, padding_side, dataset, result, request):
    dataframe = request.getfixturevalue(dataset)
    dataframe_only_item_left = request.getfixturevalue(result)
    is_spark = isinstance(dataframe, SparkDataFrame)

    padder = Padder(
        pad_columns=pad_columns, padding_value=padding_value, array_size=array_size, padding_side=padding_side
    )

    padder_interactions = padder.transform(dataframe)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_only_item_left.toPandas())
    else:
        assert padder_interactions.equals(dataframe_only_item_left)


@pytest.mark.parametrize(
    "pad_columns, padding_value, array_size, padding_side", [(["item_id", "timestamp"], [0, -1], 5, "left")]
)
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe", "dataframe_two_columns_left"),
        ("dataframe_pandas", "dataframe_two_columns_left_pandas"),
    ],
)
def test_padder_two_columns_left(pad_columns, padding_value, array_size, padding_side, dataset, result, request):
    dataframe = request.getfixturevalue(dataset)
    dataframe_two_columns_left = request.getfixturevalue(result)
    is_spark = isinstance(dataframe, SparkDataFrame)

    padder = Padder(
        pad_columns=pad_columns, padding_value=padding_value, array_size=array_size, padding_side=padding_side
    )

    padder_interactions = padder.transform(dataframe)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_two_columns_left.toPandas())
    else:
        assert padder_interactions.equals(dataframe_two_columns_left)


@pytest.mark.parametrize("pad_columns, padding_value, array_size", [("item_id", "[PAD]", 5)])
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe", "dataframe_string"),
        ("dataframe_pandas", "dataframe_string_pandas"),
    ],
)
def test_padder_only_item_string(pad_columns, padding_value, array_size, dataset, result, request):
    dataframe = request.getfixturevalue(dataset)
    dataframe_string = request.getfixturevalue(result)
    is_spark = isinstance(dataframe, SparkDataFrame)

    padder = Padder(pad_columns=pad_columns, padding_value=padding_value, array_size=array_size)

    padder_interactions = padder.transform(dataframe)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_string.toPandas())
    else:
        assert padder_interactions.equals(dataframe_string)


@pytest.mark.parametrize("pad_columns, padding_value, array_size", [(["item_id", "timestamp"], [0, -1], 2)])
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe", "dataframe_two_columns_len_two"),
        ("dataframe_pandas", "dataframe_two_columns_len_two_pandas"),
    ],
)
def test_padder_two_columns_len_two(pad_columns, padding_value, array_size, dataset, result, request):
    dataframe = request.getfixturevalue(dataset)
    dataframe_two_columns_len_two = request.getfixturevalue(result)
    is_spark = isinstance(dataframe, SparkDataFrame)

    padder = Padder(pad_columns=pad_columns, padding_value=padding_value, array_size=array_size)

    padder_interactions = padder.transform(dataframe)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_two_columns_len_two.toPandas())
    else:
        assert padder_interactions.equals(dataframe_two_columns_len_two)


@pytest.mark.parametrize("pad_columns, padding_value, array_size", [(["item_id"], 0, None)])
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe", "dataframe_only_item_none"),
        ("dataframe_pandas", "dataframe_only_item_none_pandas"),
    ],
)
def test_padder_only_item_none(pad_columns, padding_value, array_size, dataset, result, request):
    dataframe = request.getfixturevalue(dataset)
    dataframe_two_columns_len_two = request.getfixturevalue(result)
    is_spark = isinstance(dataframe, SparkDataFrame)

    padder = Padder(pad_columns=pad_columns, padding_value=padding_value, array_size=array_size)

    padder_interactions = padder.transform(dataframe)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_two_columns_len_two.toPandas())
    else:
        assert padder_interactions.equals(dataframe_two_columns_len_two)


@pytest.mark.parametrize("pad_columns, padding_value, array_size", [(["item_id", "timestamp"], 0, None)])
@pytest.mark.parametrize(
    "dataset, result",
    [
        ("dataframe_special", "dataframe_two_columns_none"),
        ("dataframe_special_pandas", "dataframe_two_columns_none_pandas"),
    ],
)
def test_padder_two_columns_none(pad_columns, padding_value, array_size, dataset, result, request):
    dataframe_special = request.getfixturevalue(dataset)
    dataframe_two_columns_none = request.getfixturevalue(result)
    is_spark = isinstance(dataframe_special, SparkDataFrame)

    padder = Padder(pad_columns=pad_columns, padding_value=padding_value, array_size=array_size)

    padder_interactions = padder.transform(dataframe_special)
    columns = padder_interactions.collect()[0].asDict().keys() if is_spark else padder_interactions.columns
    assert "user_id" in columns
    assert "item_id" in columns
    assert "timestamp" in columns

    if is_spark is True:
        assert padder_interactions.toPandas().equals(dataframe_two_columns_none.toPandas())
    else:
        assert padder_interactions.equals(dataframe_two_columns_none)


@pytest.mark.parametrize("pad_columns, padding_side", [(["item_id", "timestamp"], "smth")])
@pytest.mark.usefixtures("dataframe")
def test_wrong_padding_side(pad_columns, padding_side, dataframe):
    with pytest.raises(ValueError):
        padder = Padder(pad_columns=pad_columns, padding_side=padding_side)
        padder.transform(dataframe)


@pytest.mark.parametrize("pad_columns, padding_value", [(["item_id", "timestamp"], [0, 1, 0])])
@pytest.mark.usefixtures("dataframe")
def test_different_pad_columns_padding_value(pad_columns, padding_value, dataframe):
    with pytest.raises(ValueError):
        padder = Padder(pad_columns=pad_columns, padding_value=padding_value)
        padder.transform(dataframe)


@pytest.mark.parametrize(
    "pad_columns, array_size",
    [(["item_id", "timestamp"], -1), (["item_id", "timestamp"], 0), (["item_id", "timestamp"], 0.25)],
)
@pytest.mark.usefixtures("dataframe")
def test_wrong_array_size(pad_columns, array_size, dataframe):
    with pytest.raises(ValueError):
        padder = Padder(pad_columns=pad_columns, array_size=array_size)
        padder.transform(dataframe)


@pytest.mark.parametrize("pad_columns", [("smth")])
@pytest.mark.usefixtures("dataframe")
def test_unknown_column(pad_columns, dataframe):
    with pytest.raises(ValueError):
        padder = Padder(pad_columns=pad_columns)
        padder.transform(dataframe)


@pytest.mark.parametrize("pad_columns", [("user_id")])
@pytest.mark.usefixtures("dataframe")
def test_not_array_column(pad_columns, dataframe):
    with pytest.raises(ValueError):
        padder = Padder(pad_columns=pad_columns)
        padder.transform(dataframe)


@pytest.mark.parametrize("pad_columns", [("user_id")])
@pytest.mark.usefixtures("dataframe_pandas")
def test_invalid_column_dtype_pandas(pad_columns, dataframe_pandas):
    with pytest.raises(ValueError):
        Padder(pad_columns=pad_columns).transform(dataframe_pandas)
