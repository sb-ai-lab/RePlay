import pandas as pd
import pytest

from replay.preprocessing import LabelEncoder, LabelEncodingRule
from replay.utils import PYSPARK_AVAILABLE

if PYSPARK_AVAILABLE:
    import pyspark.sql.functions as F


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


@pytest.mark.parametrize("column", ["user_id", "item_id"])
@pytest.mark.usefixtures("simple_dataframe_pandas")
def test_label_encoder_pandas(column, simple_dataframe_pandas):
    rule = LabelEncodingRule(column, default_value="last")
    encoder = LabelEncoder([rule]).fit(simple_dataframe_pandas)

    mapped_data = encoder.transform(simple_dataframe_pandas)
    assert isinstance(encoder.inverse_mapping, dict)
    assert list(encoder.inverse_mapping[column].items())[-1][0] + 1 == len(encoder.mapping[column])
    rebuild_original_cols = encoder.inverse_transform(mapped_data)
    changed_interactions = simple_dataframe_pandas[rebuild_original_cols.columns]

    assert changed_interactions.equals(rebuild_original_cols)


@pytest.mark.parametrize("column", ["user_id", "item_id"])
@pytest.mark.usefixtures("simple_dataframe_pandas")
def test_label_encoder_load_rule_pandas(column, simple_dataframe_pandas):
    rule = LabelEncodingRule(column)
    encoder = LabelEncoder([rule])
    mapped_data = encoder.fit_transform(simple_dataframe_pandas)
    mapping = encoder.mapping[column]

    new_encoder = LabelEncoder([LabelEncodingRule(column, mapping)])
    new_encoder.fit(simple_dataframe_pandas)

    rebuild_original_cols = new_encoder.inverse_transform(mapped_data)
    changed_interactions = simple_dataframe_pandas[rebuild_original_cols.columns]

    assert changed_interactions.equals(rebuild_original_cols)


@pytest.mark.usefixtures("simple_dataframe_pandas")
def test_label_encoder_is_not_fitted(simple_dataframe_pandas):
    encoder = LabelEncoder([LabelEncodingRule("user_id")])

    with pytest.raises(RuntimeError):
        encoder.mapping()
    with pytest.raises(RuntimeError):
        encoder.inverse_mapping()
    with pytest.raises(RuntimeError):
        encoder.transform(simple_dataframe_pandas)
    with pytest.raises(RuntimeError):
        encoder.inverse_transform(simple_dataframe_pandas)


@pytest.mark.parametrize("column", ["user_id", "item_id"])
@pytest.mark.usefixtures("simple_dataframe_pandas")
def test_label_encoder_pandas_wrong_inplace_transform(column, simple_dataframe_pandas):
    rule = LabelEncodingRule(column)
    encoder = LabelEncoder([rule]).fit(simple_dataframe_pandas)

    mapped_data = encoder.transform(simple_dataframe_pandas)

    assert id(mapped_data) != id(simple_dataframe_pandas)


@pytest.mark.usefixtures(
    "pandas_df_for_labelencoder",
    "pandas_df_for_labelencoder_modified",
)
def test_label_encoder_with_handled_null_values_pandas(
    pandas_df_for_labelencoder,
    pandas_df_for_labelencoder_modified,
):
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(pandas_df_for_labelencoder)
    encoder.set_handle_unknowns({"item1": "use_default_value", "item2": "use_default_value"})
    encoder.set_default_values({"item1": "last", "item2": 5})
    mapped_interactions = encoder.transform(pandas_df_for_labelencoder_modified)
    assert str(mapped_interactions.iloc[-1]["item1"]) == "2"
    assert str(mapped_interactions.iloc[-1]["item2"]) == "5"


@pytest.mark.usefixtures(
    "pandas_df_for_labelencoder",
    "pandas_df_for_labelencoder_modified",
)
def test_none_type_passed_as_default_value_pandas(
    pandas_df_for_labelencoder,
    pandas_df_for_labelencoder_modified,
):
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(pandas_df_for_labelencoder)
    encoder.set_handle_unknowns({"item1": "use_default_value", "item2": "use_default_value"})
    encoder.set_default_values({"item1": "last", "item2": None})
    mapped_interactions = encoder.transform(pandas_df_for_labelencoder_modified)

    assert mapped_interactions.loc[2, "item2"] is None


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


@pytest.mark.usefixtures(
    "pandas_df_for_labelencoder",
    "pandas_df_for_labelencoder_modified",
)
def test_label_encoder_with_null_values_pandas(
    pandas_df_for_labelencoder,
    pandas_df_for_labelencoder_modified,
):
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(pandas_df_for_labelencoder)
    encoder.set_default_values({"item1": "last", "item2": 5})
    with pytest.raises(ValueError):
        encoder.transform(pandas_df_for_labelencoder_modified)


@pytest.mark.usefixtures(
    "pandas_df_for_labelencoder",
)
def test_label_encoder_with_default_value_in_seen_labels(
    pandas_df_for_labelencoder,
):
    encoder = LabelEncoder([LabelEncodingRule("item1", handle_unknown="use_default_value", default_value=1)])
    with pytest.raises(ValueError):
        encoder.fit(pandas_df_for_labelencoder)

    encoder = LabelEncoder([LabelEncodingRule("item1", handle_unknown="use_default_value", default_value=-1)])
    encoder.fit(pandas_df_for_labelencoder)


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


@pytest.mark.usefixtures(
    "pandas_df_for_labelencoder",
    "pandas_df_for_labelencoder_modified",
)
def test_pandas_partial_fit(pandas_df_for_labelencoder, pandas_df_for_labelencoder_modified):
    df = pandas_df_for_labelencoder
    new_df = pandas_df_for_labelencoder_modified

    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.fit(df)
    encoder.partial_fit(new_df)
    transformed = encoder.transform(new_df)

    assert (transformed["item1"] == [0, 1, 2]).all()
    assert (transformed["item2"] == [0, 1, 2]).all()
    assert "item_1" in encoder.mapping["item1"]
    assert "item_2" in encoder.mapping["item1"]
    assert "item_3" in encoder.mapping["item1"]
    assert "item_1" in encoder.mapping["item2"]
    assert "item_2" in encoder.mapping["item2"]
    assert "item_3" in encoder.mapping["item2"]


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


@pytest.mark.usefixtures("pandas_df_for_labelencoder")
def test_partial_fit_to_unfitted_encoder(pandas_df_for_labelencoder):
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.partial_fit(pandas_df_for_labelencoder)
    transformed = encoder.transform(pandas_df_for_labelencoder)

    assert (transformed["item1"] == [0, 1]).all()
    assert (transformed["item2"] == [0, 1]).all()
    assert "item_1" in encoder.mapping["item1"]
    assert "item_2" in encoder.mapping["item1"]
    assert "item_1" in encoder.mapping["item2"]
    assert "item_2" in encoder.mapping["item2"]


@pytest.mark.usefixtures(
    "pandas_df_for_labelencoder",
    "pandas_df_for_labelencoder_modified",
    "pandas_df_for_labelencoder_new_data",
)
def test_default_value_after_partial_fit(
    pandas_df_for_labelencoder, pandas_df_for_labelencoder_modified, pandas_df_for_labelencoder_new_data
):
    encoder = LabelEncoder([LabelEncodingRule("item1"), LabelEncodingRule("item2")])
    encoder.set_handle_unknowns({"item1": "use_default_value", "item2": "use_default_value"})
    encoder.set_default_values({"item1": "last", "item2": 5})
    encoder.fit(pandas_df_for_labelencoder)
    after_fit = encoder.transform(pandas_df_for_labelencoder_modified)

    encoder.partial_fit(pandas_df_for_labelencoder_modified)
    after_partial_fit = encoder.transform(pandas_df_for_labelencoder_new_data)

    assert after_fit["item1"].tolist()[-1] == 2
    assert after_fit["item2"].tolist()[-1] == 5
    assert after_partial_fit["item1"].tolist()[-1] == 3
    assert after_partial_fit["item2"].tolist()[-1] == 5
