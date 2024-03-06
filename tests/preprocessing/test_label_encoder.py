import pandas as pd
import pytest

from replay.preprocessing import LabelEncoder, LabelEncodingRule
from replay.utils import PYSPARK_AVAILABLE

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as F


@pytest.mark.spark
@pytest.mark.parametrize("column", ["user_id"])
@pytest.mark.usefixtures("simple_dataframe")
def test_label_encoder_spark(column, simple_dataframe):
    rule = LabelEncodingRule(column)
    encoder = LabelEncoder([rule]).fit(simple_dataframe)

    mapped_data = encoder.transform(simple_dataframe)
    rebuild_original_cols = encoder.inverse_transform(mapped_data).withColumn(column, F.col(column))

    columns_order = ["user_id", "item_id", "timestamp"]
    df1 = simple_dataframe.orderBy(*columns_order).toPandas()[columns_order]
    df2 = rebuild_original_cols.orderBy(*columns_order).toPandas()[columns_order]

    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.spark
@pytest.mark.parametrize("column", ["user_id"])
@pytest.mark.usefixtures("simple_dataframe")
def test_label_encoder_load_rule_spark(column, simple_dataframe):
    rule = LabelEncodingRule(column)
    encoder = LabelEncoder([rule])
    mapped_data = encoder.fit_transform(simple_dataframe)
    mapping = encoder.mapping[column]

    new_encoder = LabelEncoder([LabelEncodingRule(column, mapping)])
    new_encoder.fit(simple_dataframe)

    rebuild_original_cols = new_encoder.inverse_transform(mapped_data).withColumn(column, F.col(column))

    columns_order = ["user_id", "item_id", "timestamp"]
    df1 = simple_dataframe.orderBy(*columns_order).toPandas()[columns_order]
    df2 = rebuild_original_cols.orderBy(*columns_order).toPandas()[columns_order]

    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.core
@pytest.mark.parametrize("column", ["user_id", "item_id"])
@pytest.mark.parametrize(
    "dataframe",
    [
        ("simple_dataframe_pandas"),
        ("simple_dataframe_polars"),
    ],
)
def test_label_encoder_pandas_polars(column, dataframe, request):
    df = request.getfixturevalue(dataframe)
    rule = LabelEncodingRule(column, default_value="last")
    encoder = LabelEncoder([rule]).fit(df)

    mapped_data = encoder.transform(df)
    assert isinstance(encoder.inverse_mapping, dict)
    assert list(encoder.inverse_mapping[column].items())[-1][0] + 1 == len(encoder.mapping[column])
    rebuild_original_cols = encoder.inverse_transform(mapped_data)
    changed_interactions = df[rebuild_original_cols.columns]

    assert changed_interactions.equals(rebuild_original_cols)


@pytest.mark.core
@pytest.mark.parametrize("column", ["user_id", "item_id"])
@pytest.mark.parametrize(
    "dataframe",
    [
        ("simple_dataframe_pandas"),
        ("simple_dataframe_polars"),
    ],
)
def test_label_encoder_load_rule_pandas_polars(column, dataframe, request):
    df = request.getfixturevalue(dataframe)
    rule = LabelEncodingRule(column)
    encoder = LabelEncoder([rule])
    mapped_data = encoder.fit_transform(df)
    mapping = encoder.mapping[column]

    new_encoder = LabelEncoder([LabelEncodingRule(column, mapping)])
    new_encoder.fit(df)

    rebuild_original_cols = new_encoder.inverse_transform(mapped_data)
    changed_interactions = df[rebuild_original_cols.columns]

    assert changed_interactions.equals(rebuild_original_cols)


@pytest.mark.core
@pytest.mark.parametrize(
    "dataframe",
    [
        ("simple_dataframe_pandas"),
        ("simple_dataframe_polars"),
    ],
)
def test_label_encoder_is_not_fitted(dataframe, request):
    df = request.getfixturevalue(dataframe)
    encoder = LabelEncoder([LabelEncodingRule("user_id")])

    with pytest.raises(RuntimeError):
        encoder.mapping()
    with pytest.raises(RuntimeError):
        encoder.inverse_mapping()
    with pytest.raises(RuntimeError):
        encoder.transform(df)
    with pytest.raises(RuntimeError):
        encoder.inverse_transform(df)


@pytest.mark.core
@pytest.mark.parametrize("column", ["user_id", "item_id"])
@pytest.mark.parametrize(
    "dataframe",
    [
        ("simple_dataframe_pandas"),
        ("simple_dataframe_polars"),
    ],
)
def test_label_encoder_pandas_polars_wrong_inplace_transform(column, dataframe, request):
    df = request.getfixturevalue(dataframe)
    rule = LabelEncodingRule(column)
    encoder = LabelEncoder([rule]).fit(df)

    mapped_data = encoder.transform(df)

    assert id(mapped_data) != id(df)


@pytest.mark.core
@pytest.mark.usefixtures(
    "pandas_df_for_labelencoder",
    "pandas_df_for_labelencoder_modified",
)
@pytest.mark.parametrize(
    "df_for_labelencoder, df_for_labelencoder_modified",
    [
        ("pandas_df_for_labelencoder", "pandas_df_for_labelencoder_modified"),
        ("polars_df_for_labelencoder", "polars_df_for_labelencoder_modified"),
    ],
)
def test_label_encoder_with_handled_null_values_pandas_polars(
    df_for_labelencoder,
    df_for_labelencoder_modified,
    request,
):
    df_labelencoder = request.getfixturevalue(df_for_labelencoder)
    df_labelencoder_modified = request.getfixturevalue(df_for_labelencoder_modified)
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(df_labelencoder)
    encoder.set_handle_unknowns({"item1": "use_default_value", "item2": "use_default_value"})
    encoder.set_default_values({"item1": "last", "item2": 5})
    mapped_interactions = encoder.transform(df_labelencoder_modified)
    assert mapped_interactions.tail(1)["item1"].to_list()[0] == 2
    assert mapped_interactions.tail(1)["item2"].to_list()[0] == 5


@pytest.mark.core
@pytest.mark.parametrize(
    "df_for_labelencoder, df_for_labelencoder_modified",
    [
        ("pandas_df_for_labelencoder", "pandas_df_for_labelencoder_modified"),
        ("polars_df_for_labelencoder", "polars_df_for_labelencoder_modified"),
    ],
)
def test_none_type_passed_as_default_value_pandas_polars(
    df_for_labelencoder,
    df_for_labelencoder_modified,
    request,
):
    df_labelencoder = request.getfixturevalue(df_for_labelencoder)
    df_labelencoder_modified = request.getfixturevalue(df_for_labelencoder_modified)
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(df_labelencoder)
    encoder.set_handle_unknowns({"item1": "use_default_value", "item2": "use_default_value"})
    encoder.set_default_values({"item1": "last", "item2": None})
    mapped_interactions = encoder.transform(df_labelencoder_modified)

    assert mapped_interactions.tail(1)["item2"].to_list()[0] is None


@pytest.mark.spark
@pytest.mark.usefixtures(
    "spark_df_for_labelencoder",
    "spark_df_for_labelencoder_modified",
)
def test_label_encoder_with_handled_null_values_spark(
    spark_df_for_labelencoder,
    spark_df_for_labelencoder_modified,
):
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(spark_df_for_labelencoder)
    encoder.set_handle_unknowns({"item1": "use_default_value", "item2": "use_default_value"})
    encoder.set_default_values({"item1": None, "item2": "last"})
    mapped_interactions = encoder.transform(spark_df_for_labelencoder_modified).toPandas().sort_index()
    mapped_interactions.sort_values("user_id", inplace=True)

    assert str(mapped_interactions.iloc[-1]["item1"]) == "nan"
    assert str(mapped_interactions.iloc[-1]["item2"]) == "2"


@pytest.mark.spark
@pytest.mark.usefixtures(
    "spark_df_for_labelencoder",
    "spark_df_for_labelencoder_modified",
)
def test_label_encoder_with_null_values_spark(
    spark_df_for_labelencoder,
    spark_df_for_labelencoder_modified,
):
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(spark_df_for_labelencoder)
    encoder.set_default_values({"item1": None, "item2": "last"})
    with pytest.raises(ValueError):
        encoder.transform(spark_df_for_labelencoder_modified)


@pytest.mark.core
@pytest.mark.parametrize(
    "df_for_labelencoder, df_for_labelencoder_modified",
    [
        ("pandas_df_for_labelencoder", "pandas_df_for_labelencoder_modified"),
        ("polars_df_for_labelencoder", "polars_df_for_labelencoder_modified"),
    ],
)
def test_label_encoder_with_null_values_pandas_polars(
    df_for_labelencoder,
    df_for_labelencoder_modified,
    request,
):
    df_labelencoder = request.getfixturevalue(df_for_labelencoder)
    df_labelencoder_modified = request.getfixturevalue(df_for_labelencoder_modified)
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(df_labelencoder)
    encoder.set_default_values({"item1": "last", "item2": 5})
    with pytest.raises(ValueError):
        encoder.transform(df_labelencoder_modified)


@pytest.mark.core
@pytest.mark.parametrize(
    "df_for_labelencoder",
    [
        ("pandas_df_for_labelencoder"),
        ("polars_df_for_labelencoder"),
    ],
)
def test_label_encoder_with_default_value_in_seen_labels(
    df_for_labelencoder,
    request,
):
    df_labelencoder = request.getfixturevalue(df_for_labelencoder)
    encoder = LabelEncoder([LabelEncodingRule("item1", handle_unknown="use_default_value", default_value=1)])
    with pytest.raises(ValueError):
        encoder.fit(df_labelencoder)

    encoder = LabelEncoder([LabelEncodingRule("item1", handle_unknown="use_default_value", default_value=-1)])
    encoder.fit(df_labelencoder)


@pytest.mark.spark
@pytest.mark.usefixtures(
    "spark",
)
def test_label_encoder_undetectable_type_spark(spark):
    data = []

    for i in range(1000):
        gg = 1
        if i < 500:
            gg = None
        else:
            gg = 1
        data.append([(gg, str(1000 - i)), i])

    df = spark.createDataFrame(data, schema=["user_id", "item_id"])
    encoder = LabelEncoder([LabelEncodingRule("user_id"), LabelEncodingRule("item_id")])
    encoder.fit_transform(df)


@pytest.mark.core
def test_label_encoder_value_errors():
    with pytest.raises(ValueError):
        LabelEncoder([LabelEncodingRule("item1", handle_unknown="qwerty", default_value="some_text")])

    with pytest.raises(ValueError):
        LabelEncoder([LabelEncodingRule("item1", handle_unknown="use_default_value", default_value="some_text")])

    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])

    with pytest.raises(ValueError):
        encoder.set_default_values({"item1": "some_text", "item2": None})

    with pytest.raises(ValueError):
        encoder.set_default_values({"item3": "some_text"})

    with pytest.raises(ValueError):
        encoder.set_handle_unknowns({"item2": "some_text"})

    with pytest.raises(ValueError):
        encoder.set_handle_unknowns({"item3": "some_text"})


@pytest.mark.core
@pytest.mark.parametrize(
    "df_for_labelencoder, df_for_labelencoder_modified",
    [
        ("pandas_df_for_labelencoder", "pandas_df_for_labelencoder_modified"),
        ("polars_df_for_labelencoder", "polars_df_for_labelencoder_modified"),
    ],
)
def test_pandas_polars_partial_fit(
    df_for_labelencoder,
    df_for_labelencoder_modified,
    request,
):
    df = request.getfixturevalue(df_for_labelencoder)
    new_df = request.getfixturevalue(df_for_labelencoder_modified)

    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(df)
    encoder.partial_fit(new_df)
    transformed = encoder.transform(new_df)

    assert sorted(transformed["item1"].to_list()) == [0, 1, 2]
    assert sorted(transformed["item2"].to_list()) == [0, 1, 2]
    assert "item_1" in encoder.mapping["item1"]
    assert "item_2" in encoder.mapping["item1"]
    assert "item_3" in encoder.mapping["item1"]
    assert "item_1" in encoder.mapping["item2"]
    assert "item_2" in encoder.mapping["item2"]
    assert "item_3" in encoder.mapping["item2"]


@pytest.mark.spark
@pytest.mark.usefixtures(
    "spark_df_for_labelencoder",
    "spark_df_for_labelencoder_modified",
)
def test_spark_partial_fit(spark_df_for_labelencoder, spark_df_for_labelencoder_modified):
    df = spark_df_for_labelencoder
    new_df = spark_df_for_labelencoder_modified

    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(df)
    encoder.partial_fit(new_df)
    transformed = encoder.transform(new_df)

    item1_encoded = [x["item1"] for x in transformed.select("item1").collect()]
    item2_encoded = [x["item2"] for x in transformed.select("item2").collect()]

    assert sorted(item1_encoded) == [0, 1, 2]
    assert sorted(item2_encoded) == [0, 1, 2]
    assert "item_1" in encoder.mapping["item1"]
    assert "item_2" in encoder.mapping["item1"]
    assert "item_3" in encoder.mapping["item1"]
    assert "item_1" in encoder.mapping["item2"]
    assert "item_2" in encoder.mapping["item2"]
    assert "item_3" in encoder.mapping["item2"]


@pytest.mark.core
@pytest.mark.parametrize(
    "df_for_labelencoder",
    [
        ("pandas_df_for_labelencoder"),
        ("polars_df_for_labelencoder"),
    ],
)
def test_partial_fit_to_unfitted_encoder(
    df_for_labelencoder,
    request,
):
    df = request.getfixturevalue(df_for_labelencoder)
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.partial_fit(df)
    transformed = encoder.transform(df)

    assert sorted(transformed["item1"].to_list()) == [0, 1]
    assert sorted(transformed["item2"].to_list()) == [0, 1]
    assert "item_1" in encoder.mapping["item1"]
    assert "item_2" in encoder.mapping["item1"]
    assert "item_1" in encoder.mapping["item2"]
    assert "item_2" in encoder.mapping["item2"]


@pytest.mark.core
@pytest.mark.usefixtures(
    "pandas_df_for_labelencoder",
    "pandas_df_for_labelencoder_modified",
    "pandas_df_for_labelencoder_new_data",
)
@pytest.mark.parametrize(
    "df_for_labelencoder, df_for_labelencoder_modified, df_for_labelencoder_new_data",
    [
        ("pandas_df_for_labelencoder", "pandas_df_for_labelencoder_modified", "pandas_df_for_labelencoder_new_data"),
        ("polars_df_for_labelencoder", "polars_df_for_labelencoder_modified", "polars_df_for_labelencoder_new_data"),
    ],
)
def test_default_value_after_partial_fit(
    df_for_labelencoder,
    df_for_labelencoder_modified,
    df_for_labelencoder_new_data,
    request,
):
    df = request.getfixturevalue(df_for_labelencoder)
    df_modified = request.getfixturevalue(df_for_labelencoder_modified)
    df_new_data = request.getfixturevalue(df_for_labelencoder_new_data)
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.set_handle_unknowns({"item1": "use_default_value", "item2": "use_default_value"})
    encoder.set_default_values({"item1": "last", "item2": 5})
    encoder.fit(df)
    after_fit = encoder.transform(df_modified)

    encoder.partial_fit(df_modified)
    after_partial_fit = encoder.transform(df_new_data)

    assert after_fit["item1"].to_list()[-1] == 2
    assert after_fit["item2"].to_list()[-1] == 5
    assert after_partial_fit["item1"].to_list()[-1] == 3
    assert after_partial_fit["item2"].to_list()[-1] == 5


@pytest.mark.core
@pytest.mark.usefixtures("simple_dataframe_pandas")
def test_label_encoder_pandas_transform_optimization(simple_dataframe_pandas):
    rule = LabelEncodingRule("user_id", default_value="last")
    encoder = LabelEncoder([rule]).fit(simple_dataframe_pandas)

    mapped_data = encoder.transform(simple_dataframe_pandas)
    rule._TRANSFORM_PERFORMANCE_THRESHOLD_FOR_PANDAS = 1

    encoder_mod = LabelEncoder([rule]).fit(simple_dataframe_pandas)
    mapped_data_mod = encoder_mod.transform(simple_dataframe_pandas)

    assert mapped_data.equals(mapped_data_mod)


@pytest.mark.core
@pytest.mark.usefixtures("dataframe_not_implemented")
def test_label_encoder_not_implemented_df(dataframe_not_implemented):
    rule = LabelEncodingRule("user_id", default_value="last")
    with pytest.raises(NotImplementedError):
        LabelEncoder([rule]).fit(dataframe_not_implemented)

    rule._mapping = {"fake": "mapping"}
    encoder = LabelEncoder([rule])
    with pytest.raises(NotImplementedError):
        encoder.transform(dataframe_not_implemented)

    with pytest.raises(NotImplementedError):
        encoder.partial_fit(dataframe_not_implemented)

    with pytest.raises(NotImplementedError):
        encoder.inverse_transform(dataframe_not_implemented)
