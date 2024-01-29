# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import pytest
import logging

from replay.data import FeatureHint, FeatureInfo, FeatureSchema, FeatureType
from replay.models import CatPopRec
from tests.utils import create_dataset, spark, sparkDataFrameEqual

pyspark = pytest.importorskip("pyspark")
from pyspark.sql import functions as sf


@pytest.fixture
def cat_tree(spark):
    return spark.createDataFrame(
        data=[
            [None, "healthy_food"],
            [None, "groceries"],
            ["groceries", "fruits"],
            ["fruits", "apples"],
            ["fruits", "bananas"],
            ["apples", "red_apples"],
        ],
        schema="parent_cat string, category string",
    )


@pytest.fixture
def cat_log(spark):
    # assume item 1 is an apple-banana mix and item 2 is a banana
    return spark.createDataFrame(
        data=[
            [1, 1, "red_apples", 5],
            [1, 2, "bananas", 1],
            [2, 1, "healthy_food", 3],
            [3, 1, "bananas", 2],
        ],
        schema="user_idx int, item_idx int, category string, relevance int",
    )


@pytest.fixture
def requested_cats(spark):
    return spark.createDataFrame(
        data=[
            ["healthy_food"],
            ["fruits"],
            ["red_apples"],
        ],
        schema="category string",
    )


@pytest.fixture
def cold_items(spark):
    # assume item 1 is an apple-banana mix and item 2 is a banana
    return spark.createDataFrame(
        data=[
            [1],
            [1],
            [2],
            [3],
        ],
        schema="item_idx int",
    )


@pytest.fixture
def model(cat_tree):
    return CatPopRec(cat_tree)


def test_cat_tree(model):
    mapping = model.leaf_cat_mapping.orderBy("category")
    mapping.show()
    assert mapping.count() == 8
    assert mapping.filter(sf.col("category") == "healthy_food").count() == 1
    assert (
        mapping.filter(sf.col("category") == "healthy_food")
        .select("leaf_cat")
        .collect()[0][0]
        == "healthy_food"
    )

    assert mapping.filter(sf.col("category") == "groceries").count() == 2
    assert sorted(
        mapping.filter(sf.col("category") == "groceries")
        .select("leaf_cat")
        .toPandas()["leaf_cat"]
        .tolist()
    ) == ["bananas", "red_apples"]


def test_works_no_rel(spark, cat_log, requested_cats, model):
    ground_thuth = spark.createDataFrame(
        data=[
            ["red_apples", 1, 1.0],
            ["healthy_food", 1, 1.0],
            ["fruits", 1, 2 / 3],
            ["fruits", 2, 1 / 3],
        ],
        schema="category string, item_idx int, relevance double",
    )
    feature_schema = FeatureSchema(
        [
            FeatureInfo(
                column="user_idx",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.QUERY_ID,
            ),
            FeatureInfo(
                column="item_idx",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
            ),
            FeatureInfo(
                column="category",
                feature_type=FeatureType.CATEGORICAL,
            ),
        ]
    )
    dataset = create_dataset(cat_log.drop("relevance"), feature_schema=feature_schema)
    model.fit(dataset)
    sparkDataFrameEqual(model.predict(requested_cats, k=3), ground_thuth)


def test_works_rel(spark, cat_log, requested_cats, model):
    ground_thuth = spark.createDataFrame(
        data=[
            ["red_apples", 1, 1.0],
            ["healthy_food", 1, 1.0],
            ["fruits", 1, 7 / 8],
            ["fruits", 2, 1 / 8],
        ],
        schema="category string, item_idx int, relevance double",
    )
    feature_schema = FeatureSchema(
        [
            FeatureInfo(
                column="user_idx",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.QUERY_ID,
            ),
            FeatureInfo(
                column="item_idx",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
            ),
            FeatureInfo(
                column="category",
                feature_type=FeatureType.CATEGORICAL,
            ),
            FeatureInfo(
                column="relevance",
                feature_type=FeatureType.NUMERICAL,
                feature_hint=FeatureHint.RATING,
            ),
        ]
    )
    dataset = create_dataset(cat_log, feature_schema=feature_schema)
    model.fit(dataset)
    sparkDataFrameEqual(model.predict(requested_cats, k=3), ground_thuth)
    model._clear_cache()


def test_set_cat_tree(model, cat_tree):
    mapping = model.leaf_cat_mapping
    model.set_cat_tree(cat_tree)
    sparkDataFrameEqual(model.leaf_cat_mapping, mapping)


def test_max_iter_warning(cat_tree, caplog):
    with caplog.at_level(logging.WARNING):
        CatPopRec(cat_tree, max_iter=1)
        assert (
            "Category tree was not fully processed in 1 iterations. "
            "Increase the `max_iter` value or check the category tree structure."
            "It must not have loops and each category should have only one parent."
            in caplog.text
        )


def test_predict_cold_items(cat_log, requested_cats, model, cold_items ,caplog):
    caplog.set_level(logging.INFO, logger="replay")
    feature_schema = FeatureSchema(
        [
            FeatureInfo(
                column="user_idx",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.QUERY_ID,
            ),
            FeatureInfo(
                column="item_idx",
                feature_type=FeatureType.CATEGORICAL,
                feature_hint=FeatureHint.ITEM_ID,
            ),
            FeatureInfo(
                column="category",
                feature_type=FeatureType.CATEGORICAL,
            ),
        ]
    )
    dataset = create_dataset(cat_log.drop("relevance"), feature_schema=feature_schema)
    model.fit(dataset)
    model.predict(requested_cats, k=3, items=cold_items)

    assert (
        f"{model} model can't predict cold items, they will be ignored"
        in caplog.text
    )
