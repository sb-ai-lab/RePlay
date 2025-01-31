import numpy as np
import pandas as pd
import pytest

from replay.preprocessing import LabelEncoder, LabelEncodingRule, SequenceEncodingRule
from replay.utils import PYSPARK_AVAILABLE, PandasDataFrame, PolarsDataFrame
from tests.utils import sparkDataFrameEqual

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as F


@pytest.mark.spark
@pytest.mark.parametrize(
    "column, df_name, is_grouped_encoder",
    [
        pytest.param("user_id", "simple_dataframe", False),
        pytest.param("item_id", "simple_dataframe_array", True),
        pytest.param("item_id_list", "simple_dataframe_target_ordered", True),
    ],
)
def test_label_encoder_spark(column, df_name, is_grouped_encoder, request):
    df = request.getfixturevalue(df_name)
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    rule = rule_class(column)
    encoder = LabelEncoder([rule]).fit(df)

    mapped_data = encoder.transform(df)
    rebuild_original_cols = encoder.inverse_transform(mapped_data).withColumn(column, F.col(column))

    columns_order = ["user_id", "item_id", "timestamp"]
    df1 = df.orderBy(*columns_order).toPandas()[columns_order]
    df2 = rebuild_original_cols.orderBy(*columns_order).toPandas()[columns_order]

    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.spark
@pytest.mark.parametrize(
    "column, df_name, is_grouped_encoder",
    [
        pytest.param("user_id", "simple_dataframe", False),
        pytest.param("item_id", "simple_dataframe_array", True),
        pytest.param("item_id_list", "simple_dataframe_target_ordered", True),
    ],
)
def test_label_encoder_load_rule_spark(column, df_name, is_grouped_encoder, request):
    df = request.getfixturevalue(df_name)
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    rule = rule_class(column)
    encoder = LabelEncoder([rule])
    mapped_data = encoder.fit_transform(df)
    mapping = encoder.mapping[column]

    trained_rule = SequenceEncodingRule(column, mapping) if is_grouped_encoder else LabelEncodingRule(column, mapping)
    new_encoder = LabelEncoder([trained_rule])
    new_encoder.fit(df)
    rebuild_original_cols = new_encoder.inverse_transform(mapped_data).withColumn(column, F.col(column))

    columns_order = ["user_id", "item_id", "timestamp"]
    df1 = df.orderBy(*columns_order).toPandas()[columns_order]
    df2 = rebuild_original_cols.orderBy(*columns_order).toPandas()[columns_order]

    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.spark
@pytest.mark.parametrize("column", ["random_string"])
def test_label_encoder_determinism(column, random_string_spark_df):
    # При репартиционировании датафрейма, label encoder, обученный через spark выдает разные маппинги
    df1 = random_string_spark_df.repartition(13)
    rule_1 = LabelEncodingRule(column)
    encoder_1 = LabelEncoder([rule_1])
    encoder_1.fit(df1)
    mapping_1 = encoder_1.mapping[column]

    df2 = random_string_spark_df.repartition(11)
    rule_2 = LabelEncodingRule(column)
    encoder_2 = LabelEncoder([rule_2])
    encoder_2.fit(df2)
    mapping_2 = encoder_2.mapping[column]

    df3 = random_string_spark_df.repartition(20)
    rule_3 = LabelEncodingRule(column)
    encoder_3 = LabelEncoder([rule_3])
    encoder_3.fit(df3)
    mapping_3 = encoder_3.mapping[column]

    assert mapping_1 == mapping_2, "LabelEncoder.fit работает недетерминировано (сравнение на запуске 1 и 2)"
    assert mapping_1 == mapping_3, "LabelEncoder.fit работает недетерминировано (сравнение на запуске 1 и 3)"
    assert mapping_2 == mapping_3, "LabelEncoder.fit работает недетерминировано (сравнение на запуске 2 и 3)"


@pytest.mark.spark
@pytest.mark.parametrize("column", ["random_string"])
def test_label_encoder_partial_fit_determinism(column, random_string_spark_df, static_string_spark_df):
    # При репартиционировании датафрейма, label encoder, обученный через spark выдает разные маппинги
    df1 = random_string_spark_df.repartition(13)
    rule_1 = LabelEncodingRule(column)
    encoder_1 = LabelEncoder([rule_1])
    encoder_1.fit(static_string_spark_df)
    encoder_1.partial_fit(df1)
    mapping_1 = encoder_1.mapping[column]

    df2 = random_string_spark_df.repartition(11)
    rule_2 = LabelEncodingRule(column)
    encoder_2 = LabelEncoder([rule_2])
    encoder_2.fit(static_string_spark_df)
    encoder_2.partial_fit(df2)
    mapping_2 = encoder_2.mapping[column]

    df3 = random_string_spark_df.repartition(20)
    rule_3 = LabelEncodingRule(column)
    encoder_3 = LabelEncoder([rule_3])
    encoder_3.fit(static_string_spark_df)
    encoder_3.partial_fit(df3)
    mapping_3 = encoder_3.mapping[column]

    assert mapping_1 == mapping_2, "LabelEncoder.fit работает недетерминировано (сравнение на запуске 1 и 2)"
    assert mapping_1 == mapping_3, "LabelEncoder.fit работает недетерминировано (сравнение на запуске 1 и 3)"
    assert mapping_2 == mapping_3, "LabelEncoder.fit работает недетерминировано (сравнение на запуске 2 и 3)"


@pytest.mark.core
@pytest.mark.parametrize(
    "column, df_name, is_grouped_encoder",
    [
        pytest.param("user_id", "simple_dataframe_pandas", False),
        pytest.param("user_id", "simple_dataframe_polars", False),
        pytest.param("item_id", "simple_dataframe_array_pandas", True),
        pytest.param("item_id", "simple_dataframe_array_polars", True),
    ],
)
def test_label_encoder_pandas_polars(column, df_name, is_grouped_encoder, request):
    df = request.getfixturevalue(df_name)
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    rule = rule_class(column, default_value="last")
    encoder = LabelEncoder([rule]).fit(df)

    mapped_data = encoder.transform(df)
    assert isinstance(encoder.inverse_mapping, dict)
    assert list(encoder.inverse_mapping[column].items())[-1][0] + 1 == len(encoder.mapping[column])
    rebuild_original_cols = encoder.inverse_transform(mapped_data)
    changed_interactions = df[rebuild_original_cols.columns]

    assert changed_interactions.equals(rebuild_original_cols)


@pytest.mark.core
@pytest.mark.parametrize(
    "column, df_name, is_grouped_encoder",
    [
        pytest.param("user_id", "simple_dataframe_pandas", False),
        pytest.param("user_id", "simple_dataframe_polars", False),
        pytest.param("item_id", "simple_dataframe_array_pandas", True),
        pytest.param("item_id", "simple_dataframe_array_polars", True),
    ],
)
def test_label_encoder_load_rule_pandas_polars(column, df_name, is_grouped_encoder, request):
    df = request.getfixturevalue(df_name)
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    rule = rule_class(column)
    encoder = LabelEncoder([rule])
    mapped_data = encoder.fit_transform(df)
    mapping = encoder.mapping[column]

    trained_rule = SequenceEncodingRule(column, mapping) if is_grouped_encoder else LabelEncodingRule(column, mapping)
    new_encoder = LabelEncoder([trained_rule])
    new_encoder.fit(df)

    rebuild_original_cols = new_encoder.inverse_transform(mapped_data)
    changed_interactions = df[rebuild_original_cols.columns]

    assert changed_interactions.equals(rebuild_original_cols)


@pytest.mark.core
@pytest.mark.parametrize(
    "column, df_name, is_grouped_encoder",
    [
        pytest.param("user_id", "simple_dataframe_pandas", False),
        pytest.param("user_id", "simple_dataframe_polars", False),
        pytest.param("item_id", "simple_dataframe_array_pandas", True),
        pytest.param("item_id", "simple_dataframe_array_polars", True),
    ],
)
def test_label_encoder_is_not_fitted(column, df_name, is_grouped_encoder, request):
    df = request.getfixturevalue(df_name)
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    rule = rule_class(column)
    encoder = LabelEncoder([rule])

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
        (
            "pandas_df_for_grouped_labelencoder",
            "pandas_df_for_grouped_labelencoder_modified",
        ),
        (
            "polars_df_for_grouped_labelencoder",
            "polars_df_for_grouped_labelencoder_modified",
        ),
    ],
)
def test_grouped_label_encoder_with_handled_null_values_pandas_polars(
    df_for_labelencoder,
    df_for_labelencoder_modified,
    request,
):
    df_labelencoder = request.getfixturevalue(df_for_labelencoder)
    df_labelencoder_modified = request.getfixturevalue(df_for_labelencoder_modified)
    encoder = LabelEncoder([SequenceEncodingRule("item1"), SequenceEncodingRule("item2")])
    encoder.fit(df_labelencoder)
    encoder.set_handle_unknowns({"item1": "use_default_value", "item2": "use_default_value"})
    encoder.set_default_values({"item1": "last", "item2": 5})
    mapped_interactions = encoder.transform(df_labelencoder_modified)

    if isinstance(mapped_interactions, PandasDataFrame):
        items1 = mapped_interactions["item1"].explode().unique().tolist()
        items2 = mapped_interactions["item2"].explode().unique().tolist()
    else:
        items1 = mapped_interactions["item1"].explode().unique().to_list()
        items2 = mapped_interactions["item2"].explode().unique().to_list()

    assert set(items1) == {0, 1, 2}
    assert set(items2) == {0, 1, 5}


@pytest.mark.spark
def test_grouped_label_encoder_with_handled_null_values_spark(
    spark_df_for_grouped_labelencoder,
    spark_df_for_grouped_labelencoder_modified,
):
    encoder = LabelEncoder([SequenceEncodingRule("item1"), SequenceEncodingRule("item2")])
    encoder.fit(spark_df_for_grouped_labelencoder)
    encoder.set_handle_unknowns({"item1": "use_default_value", "item2": "use_default_value"})
    encoder.set_default_values({"item1": "last", "item2": 5})
    mapped_interactions = encoder.transform(spark_df_for_grouped_labelencoder_modified)

    items1 = [x[0] for x in mapped_interactions.select(F.explode("item1")).collect()]
    items2 = [x[0] for x in mapped_interactions.select(F.explode("item2")).collect()]

    assert set(items1) == {0, 1, 2}
    assert set(items2) == {0, 1, 5}


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
@pytest.mark.parametrize(
    "df_name, modified_df_name, is_grouped_encoder",
    [
        pytest.param("spark_df_for_labelencoder", "spark_df_for_labelencoder_modified", False),
        pytest.param("spark_df_for_grouped_labelencoder", "spark_df_for_grouped_labelencoder_modified", True),
    ],
)
def test_label_encoder_with_null_values_spark(df_name, modified_df_name, is_grouped_encoder, request):
    df = request.getfixturevalue(df_name)
    df_modified = request.getfixturevalue(modified_df_name)
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    encoder = LabelEncoder([rule_class("item1"), rule_class("item2")])
    encoder.fit(df)
    encoder.set_default_values({"item1": None, "item2": "last"})
    with pytest.raises(ValueError):
        encoder.transform(df_modified)


@pytest.mark.core
@pytest.mark.parametrize(
    "df_name, modified_df_name, is_grouped_encoder",
    [
        ("pandas_df_for_labelencoder", "pandas_df_for_labelencoder_modified", False),
        ("polars_df_for_labelencoder", "polars_df_for_labelencoder_modified", False),
        ("pandas_df_for_grouped_labelencoder", "pandas_df_for_grouped_labelencoder_modified", True),
        ("polars_df_for_grouped_labelencoder", "polars_df_for_grouped_labelencoder_modified", True),
    ],
)
def test_label_encoder_with_null_values_pandas_polars(
    df_name,
    modified_df_name,
    is_grouped_encoder,
    request,
):
    df = request.getfixturevalue(df_name)
    df_modified = request.getfixturevalue(modified_df_name)
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    encoder = LabelEncoder([rule_class("item1"), rule_class("item2")])
    encoder.fit(df)
    encoder.set_default_values({"item1": "last", "item2": 5})
    with pytest.raises(ValueError):
        encoder.transform(df_modified)


@pytest.mark.core
@pytest.mark.parametrize(
    "df_name, is_grouped_encoder",
    [
        ("pandas_df_for_labelencoder", False),
        ("polars_df_for_labelencoder", False),
        ("polars_df_for_grouped_labelencoder", True),
        ("polars_df_for_grouped_labelencoder", True),
    ],
)
def test_label_encoder_with_default_value_in_seen_labels(df_name, is_grouped_encoder, request):
    df_labelencoder = request.getfixturevalue(df_name)
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    encoder = LabelEncoder([rule_class("item1", handle_unknown="use_default_value", default_value=1)])
    with pytest.raises(ValueError):
        encoder.fit(df_labelencoder)

    encoder = LabelEncoder([rule_class("item1", handle_unknown="use_default_value", default_value=-1)])
    encoder.fit(df_labelencoder)


@pytest.mark.spark
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
    "df_name, modified_df_name, is_grouped_encoder",
    [
        ("pandas_df_for_labelencoder", "pandas_df_for_labelencoder_modified", False),
        ("polars_df_for_labelencoder", "polars_df_for_labelencoder_modified", False),
        ("pandas_df_for_grouped_labelencoder", "pandas_df_for_grouped_labelencoder_modified", True),
        ("polars_df_for_grouped_labelencoder", "polars_df_for_grouped_labelencoder_modified", True),
    ],
)
def test_pandas_polars_partial_fit(
    df_name,
    modified_df_name,
    is_grouped_encoder,
    request,
):
    df = request.getfixturevalue(df_name)
    new_df = request.getfixturevalue(modified_df_name)
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule

    encoder = LabelEncoder([rule_class("item1"), rule_class("item2")])
    encoder.fit(df)
    encoder.partial_fit(new_df)

    mapped_data = encoder.transform(new_df)
    rebuild_original_cols = encoder.inverse_transform(mapped_data)
    changed_interactions = new_df[rebuild_original_cols.columns]
    assert changed_interactions.equals(rebuild_original_cols)

    mapped_data = encoder.transform(df)
    rebuild_original_cols = encoder.inverse_transform(mapped_data)
    changed_interactions = df[rebuild_original_cols.columns]
    assert changed_interactions.equals(rebuild_original_cols)


@pytest.mark.spark
@pytest.mark.parametrize(
    "df_name, modified_df_name, is_grouped_encoder",
    [
        ("spark_df_for_labelencoder", "spark_df_for_labelencoder_modified", False),
        ("spark_df_for_grouped_labelencoder", "spark_df_for_grouped_labelencoder_modified", True),
    ],
)
def test_spark_partial_fit(df_name, modified_df_name, is_grouped_encoder, request):
    df = request.getfixturevalue(df_name)
    new_df = request.getfixturevalue(modified_df_name)

    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    encoder = LabelEncoder([rule_class("item1"), rule_class("item2")])
    encoder.fit(df)
    encoder.partial_fit(new_df)
    transformed = encoder.transform(df)

    item1_encoded = np.array([x["item1"] for x in transformed.select("item1").collect()])
    item2_encoded = np.array([x["item2"] for x in transformed.select("item2").collect()])

    assert np.equal(np.unique(item1_encoded), [0, 1]).all()
    assert np.equal(np.unique(item2_encoded), [0, 1]).all()
    assert "item_1" in encoder.mapping["item1"]
    assert "item_2" in encoder.mapping["item1"]
    assert "item_3" in encoder.mapping["item1"]
    assert "item_1" in encoder.mapping["item2"]
    assert "item_2" in encoder.mapping["item2"]
    assert "item_3" in encoder.mapping["item2"]


@pytest.mark.core
@pytest.mark.parametrize(
    "df_name, is_grouped_encoder",
    [
        ("pandas_df_for_labelencoder", False),
        ("polars_df_for_labelencoder", False),
        ("pandas_df_for_grouped_labelencoder", True),
        ("polars_df_for_grouped_labelencoder", True),
    ],
)
def test_partial_fit_to_unfitted_encoder(
    df_name,
    is_grouped_encoder,
    request,
):
    df = request.getfixturevalue(df_name)
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    encoder = LabelEncoder([rule_class("item1"), rule_class("item2")])
    encoder.partial_fit(df)
    mapped_data = encoder.transform(df)
    rebuild_original_cols = encoder.inverse_transform(mapped_data)
    changed_interactions = df[rebuild_original_cols.columns]
    assert changed_interactions.equals(rebuild_original_cols)


@pytest.mark.core
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
def test_label_encoder_pandas_transform_optimization(simple_dataframe_pandas):
    rule = LabelEncodingRule("user_id", default_value="last")
    encoder = LabelEncoder([rule]).fit(simple_dataframe_pandas)

    mapped_data = encoder.transform(simple_dataframe_pandas)
    rule._TRANSFORM_PERFORMANCE_THRESHOLD_FOR_PANDAS = 1

    encoder_mod = LabelEncoder([rule]).fit(simple_dataframe_pandas)
    mapped_data_mod = encoder_mod.transform(simple_dataframe_pandas)

    assert mapped_data.equals(mapped_data_mod)


@pytest.mark.core
@pytest.mark.parametrize("is_grouped_encoder", [False, True])
def test_label_encoder_not_implemented_df(is_grouped_encoder, dataframe_not_implemented):
    column, default_value = "user_id", "last"
    rule_class = SequenceEncodingRule if is_grouped_encoder else LabelEncodingRule
    rule = rule_class(column, default_value=default_value)
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


@pytest.mark.parametrize(
    "df_for_labelencoder, df_for_labelencoder_modified",
    [
        pytest.param("pandas_df_for_labelencoder", "pandas_df_for_labelencoder_modified", marks=pytest.mark.core),
        pytest.param("polars_df_for_labelencoder", "polars_df_for_labelencoder_modified", marks=pytest.mark.core),
        pytest.param("spark_df_for_labelencoder", "spark_df_for_labelencoder_modified", marks=pytest.mark.spark),
    ],
)
def test_label_encoder_drop_strategy(request, df_for_labelencoder, df_for_labelencoder_modified):
    df = request.getfixturevalue(df_for_labelencoder)
    df_modified = request.getfixturevalue(df_for_labelencoder_modified)

    encoder = LabelEncoder([LabelEncodingRule("item1", handle_unknown="drop")])
    encoder.fit(df)
    transformed = encoder.transform(df_modified)
    inversed = encoder.inverse_transform(transformed)

    if isinstance(inversed, PandasDataFrame):
        items = inversed["item1"].tolist()
    elif isinstance(inversed, PolarsDataFrame):
        items = inversed["item1"].to_list()
    else:
        items = [x.item1 for x in inversed.select("item1").collect()]

    assert "item_1" in items
    assert "item_2" in items
    assert "item_3" not in items


@pytest.mark.parametrize(
    "df_for_labelencoder, df_for_labelencoder_modified",
    [
        pytest.param(
            "pandas_df_for_grouped_labelencoder", "pandas_df_for_grouped_labelencoder_modified", marks=pytest.mark.core
        ),
        pytest.param(
            "polars_df_for_grouped_labelencoder", "polars_df_for_grouped_labelencoder_modified", marks=pytest.mark.core
        ),
        pytest.param(
            "spark_df_for_grouped_labelencoder", "spark_df_for_grouped_labelencoder_modified", marks=pytest.mark.spark
        ),
    ],
)
def test_grouped_label_encoder_drop_strategy(request, df_for_labelencoder, df_for_labelencoder_modified):
    df = request.getfixturevalue(df_for_labelencoder)
    df_modified = request.getfixturevalue(df_for_labelencoder_modified)

    encoder = LabelEncoder([SequenceEncodingRule("item1", handle_unknown="drop")])
    encoder.fit(df)
    transformed = encoder.transform(df_modified)
    inversed = encoder.inverse_transform(transformed)

    if isinstance(inversed, PandasDataFrame):
        items = inversed["item1"].explode().unique().tolist()
    elif isinstance(inversed, PolarsDataFrame):
        items = inversed["item1"].explode().unique().to_list()
    else:
        items = [x[0] for x in inversed.select(F.explode("item1")).collect()]

    assert "item_1" in items
    assert "item_2" in items
    assert "item_3" not in items


@pytest.mark.parametrize(
    "df_for_labelencoder, df_for_labelencoder_new_data",
    [
        pytest.param("pandas_df_for_labelencoder", "pandas_df_for_labelencoder_new_data", marks=pytest.mark.core),
        pytest.param("polars_df_for_labelencoder", "polars_df_for_labelencoder_new_data", marks=pytest.mark.core),
        pytest.param("spark_df_for_labelencoder", "spark_df_for_labelencoder_new_data", marks=pytest.mark.spark),
    ],
)
def test_label_encoder_drop_strategy_empty_dataset(request, df_for_labelencoder, df_for_labelencoder_new_data):
    df = request.getfixturevalue(df_for_labelencoder)
    df_new = request.getfixturevalue(df_for_labelencoder_new_data)

    encoder = LabelEncoder([LabelEncodingRule("item1", handle_unknown="drop")])
    encoder.fit(df)
    transformed = encoder.transform(df_new)

    if isinstance(transformed, PandasDataFrame):
        assert transformed.empty
    elif isinstance(transformed, PolarsDataFrame):
        assert transformed.is_empty()
    else:
        assert transformed.rdd.isEmpty()


@pytest.mark.parametrize(
    "df_for_labelencoder, df_for_labelencoder_new_data",
    [
        pytest.param(
            "pandas_df_for_grouped_labelencoder", "pandas_df_for_grouped_labelencoder_new_data", marks=pytest.mark.core
        ),
        pytest.param(
            "polars_df_for_grouped_labelencoder", "polars_df_for_grouped_labelencoder_new_data", marks=pytest.mark.core
        ),
        pytest.param(
            "spark_df_for_grouped_labelencoder", "spark_df_for_grouped_labelencoder_new_data", marks=pytest.mark.spark
        ),
    ],
)
def test_grouped_label_encoder_drop_strategy_empty_dataset(request, df_for_labelencoder, df_for_labelencoder_new_data):
    df = request.getfixturevalue(df_for_labelencoder)
    df_new = request.getfixturevalue(df_for_labelencoder_new_data)

    encoder = LabelEncoder([SequenceEncodingRule("item1", handle_unknown="drop")])
    encoder.fit(df)
    transformed = encoder.transform(df_new)

    if isinstance(transformed, PandasDataFrame):
        assert transformed["item1"].apply(len).max() == 0
    elif isinstance(transformed, PolarsDataFrame):
        assert transformed["item1"].list.len().max() == 0
    else:
        assert transformed.select(F.max(F.size("item1"))).first()[0] == 0


@pytest.mark.core
@pytest.mark.parametrize("col_type", ["string", "float", "int"])
def test_label_encoder_save_load(simple_dataframe_pandas, col_type, tmp_path):
    path = (tmp_path / "encoder").resolve()
    simple_dataframe_pandas["user_id"] = simple_dataframe_pandas["user_id"].astype(col_type)
    rule = LabelEncodingRule("user_id", default_value="last")
    encoder = LabelEncoder([rule]).fit(simple_dataframe_pandas)
    mapping = encoder.mapping
    encoder.save(path)
    assert mapping == LabelEncoder.load(path).mapping


@pytest.mark.core
@pytest.mark.parametrize("dataset", ["simple_dataframe_pandas", "simple_dataframe_polars"])
def test_label_encoder_save_load_inverse_transform(dataset, tmp_path, request):
    dataset = request.getfixturevalue(dataset)
    path = (tmp_path / "encoder").resolve()
    rule = LabelEncodingRule("user_id", default_value="last")
    encoder = LabelEncoder([rule]).fit(dataset)
    encoded_data = encoder.transform(dataset)
    encoder.save(path)
    assert dataset["user_id"].equals(LabelEncoder.load(path).inverse_transform(encoded_data)["user_id"])


@pytest.mark.spark
@pytest.mark.parametrize("column", ["user_id"])
def test_label_encoder_save_load_inverse_transform_spark(column, simple_dataframe, tmp_path):
    path = (tmp_path / "encoder").resolve()
    rule = LabelEncodingRule(column)
    encoder = LabelEncoder([rule]).fit(simple_dataframe)
    encoder.save(path)
    encoded_data = encoder.transform(simple_dataframe)
    rebuild_original_cols = LabelEncoder.load(path).inverse_transform(encoded_data)
    sparkDataFrameEqual(simple_dataframe, rebuild_original_cols)
