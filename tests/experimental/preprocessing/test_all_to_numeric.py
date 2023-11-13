# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

from pyspark.sql import functions as sf

from replay.experimental.preprocessing.data_preparator import ToNumericFeatureTransformer

from tests.utils import (
    item_features,
    spark,
    sparkDataFrameEqual,
)


def test_all_to_numeric_big_threshold(item_features):
    processor = ToNumericFeatureTransformer()
    processor.fit(item_features.filter(sf.col("class") != "dog"))
    transformed = processor.transform(item_features)
    assert "iq" in transformed.columns and "color" not in transformed.columns
    assert (
        "ohe_class_dog" not in transformed.columns
        and "ohe_class_cat" in transformed.columns
    )
    assert sorted(transformed.columns) == [
        "iq",
        "item_idx",
        "ohe_class_cat",
        "ohe_class_mouse",
        "ohe_color_black",
        "ohe_color_yellow",
    ]


def test_all_to_numeric_threshold(item_features):
    processor = ToNumericFeatureTransformer(threshold=1)
    processor.fit(item_features.filter(sf.col("class") != "dog"))
    transformed = processor.transform(item_features)
    assert "iq" in transformed.columns and "color" not in transformed.columns
    assert sorted(transformed.columns) == ["iq", "item_idx"]


def test_all_to_numeric_only_numeric(item_features):
    processor = ToNumericFeatureTransformer(threshold=1)
    processor.fit(item_features.select("item_idx", "iq"))
    transformed = processor.transform(item_features.select("item_idx", "iq"))
    sparkDataFrameEqual(item_features.select("item_idx", "iq"), transformed)


def test_all_to_numeric_numeric_and_greater_threshold(
    item_features,
):
    processor = ToNumericFeatureTransformer(threshold=0)
    processor.fit(item_features)
    transformed = processor.transform(item_features)
    sparkDataFrameEqual(item_features.select("item_idx", "iq"), transformed)


def test_all_to_numeric_empty(item_features):
    processor = ToNumericFeatureTransformer(threshold=1)
    processor.fit(None)
    assert processor.transform(item_features) is None

    processor.fit(item_features.select("item_idx", "class"))
    assert processor.cat_feat_transformer is None
    transformed = processor.transform(
        item_features.select("item_idx", "class")
    )
    assert transformed.columns == ["item_idx"]
