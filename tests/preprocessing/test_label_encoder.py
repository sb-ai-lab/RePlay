import pytest

from replay.preprocessing import GroupedLabelEncodingRule, LabelEncoder, LabelEncodingRule
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
    rule_class = GroupedLabelEncodingRule if is_grouped_encoder else LabelEncodingRule
    rule = rule_class(column)
    encoder = LabelEncoder([rule]).fit(df)

    mapped_data = encoder.transform(df)
    rebuild_original_cols = encoder.inverse_transform(mapped_data)

    sparkDataFrameEqual(df, rebuild_original_cols)


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
    rule_class = GroupedLabelEncodingRule if is_grouped_encoder else LabelEncodingRule
    rule = rule_class(column)
    encoder = LabelEncoder([rule])
    mapped_data = encoder.fit_transform(df)
    mapping = encoder.mapping[column]

    trained_rule = (
        GroupedLabelEncodingRule(column, mapping) if is_grouped_encoder else LabelEncodingRule(column, mapping)
    )
    new_encoder = LabelEncoder([trained_rule])
    new_encoder.fit(df)
    rebuild_original_cols = new_encoder.inverse_transform(mapped_data).withColumn(column, F.col(column))

    sparkDataFrameEqual(df, rebuild_original_cols)


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
    rule_class = GroupedLabelEncodingRule if is_grouped_encoder else LabelEncodingRule
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
    rule_class = GroupedLabelEncodingRule if is_grouped_encoder else LabelEncodingRule
    rule = rule_class(column)
    encoder = LabelEncoder([rule])
    mapped_data = encoder.fit_transform(df)
    mapping = encoder.mapping[column]

    trained_rule = (
        GroupedLabelEncodingRule(column, mapping) if is_grouped_encoder else LabelEncodingRule(column, mapping)
    )
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
    rule_class = GroupedLabelEncodingRule if is_grouped_encoder else LabelEncodingRule
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
    rule_class = GroupedLabelEncodingRule if is_grouped_encoder else LabelEncodingRule
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
    rule_class = GroupedLabelEncodingRule if is_grouped_encoder else LabelEncodingRule

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

    rule_class = GroupedLabelEncodingRule if is_grouped_encoder else LabelEncodingRule
    encoder = LabelEncoder([rule_class("item1"), rule_class("item2")])
    encoder.fit(df)
    encoder.partial_fit(new_df)

    for dataset in [df, new_df]:
        mapped_data = encoder.transform(dataset)
        rebuild_original_cols = encoder.inverse_transform(mapped_data)
        sparkDataFrameEqual(dataset.sort("user_id"), rebuild_original_cols.sort("user_id"))


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
    rule_class = GroupedLabelEncodingRule if is_grouped_encoder else LabelEncodingRule
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
    rule_class = GroupedLabelEncodingRule if is_grouped_encoder else LabelEncodingRule
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
