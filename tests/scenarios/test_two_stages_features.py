# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import math
from datetime import datetime

import pytest

from pyspark.sql import functions as sf

from replay.scenarios.two_stages.feature_processor import (
    SecondLevelFeaturesProcessor,
    FirstLevelFeaturesProcessor,
)
from tests.utils import (
    spark,
    sparkDataFrameEqual,
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
)

simple_u_columns = ["u_log_ratings_count", "u_mean_log_items_ratings_count"]
simple_i_columns = ["i_log_ratings_count", "i_mean_log_users_ratings_count"]
time_columns = ["min_rating_date", "max_rating_date", "log_rating_dates_count"]
relevance_columns = ["std", "mean", "quantile_05", "quantile_5", "quantile_95"]
abnormality_columns = ["abnormality", "abnormalityCR"]


@pytest.fixture
def log_second_stage(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i1", date, 1.0],
            ["u1", "i4", datetime(2019, 1, 5), 3.0],
            ["u2", "i1", date, 1.0],
            ["u2", "i3", datetime(2018, 1, 1), 4.0],
            ["u3", "i3", date, 3.0],
            ["u3", "i2", date, 2.0],
            ["u3", "i4", datetime(2020, 3, 1), 1.0],
        ],
        schema=["user_idx", "item_idx", "timestamp", "relevance"],
    )


@pytest.fixture
def user_features_second_stage(spark):
    return spark.createDataFrame(
        [("u1", 2.0, 3.0, "M"), ("u2", 1.0, 4.0, "F")]
    ).toDF("user_idx", "user_feature_1", "user_feature_2", "gender")


@pytest.fixture
def item_features_second_stage(spark):
    return spark.createDataFrame(
        [("i1", 4.0, "cat"), ("i2", 10.0, "dog"), ("i4", 0.0, "cat")]
    ).toDF("item_idx", "item_feature_1", "class")


@pytest.fixture
def second_stage_fp():
    return SecondLevelFeaturesProcessor()


def test_second_level_fp_log_features_unary_relevance_no_timestamp(
    spark, second_stage_fp, log_second_stage
):
    user_log_features, item_log_features = second_stage_fp._calc_log_features(
        log_second_stage.withColumn("relevance", sf.lit(1)).withColumn(
            "timestamp", sf.to_timestamp(sf.lit(datetime(2019, 1, 1)))
        )
    )
    assert sorted(user_log_features.columns) == sorted(
        ["user_idx"] + simple_u_columns
    )
    gt_user_features = spark.createDataFrame(
        data=[
            ["u1", math.log(2), (math.log(2) + math.log(2)) / 2],
            ["u2", math.log(2), (math.log(2) + math.log(2)) / 2],
            ["u3", math.log(3), (math.log(2) + math.log(2) + math.log(1)) / 3],
        ],
        schema=["user_idx"] + simple_u_columns,
    )
    sparkDataFrameEqual(gt_user_features, user_log_features)

    assert sorted(item_log_features.columns) == sorted(
        ["item_idx"] + simple_i_columns
    )
    gt_item_features = spark.createDataFrame(
        data=[
            ["i1", math.log(2), (math.log(2) + math.log(2)) / 2],
            ["i2", math.log(1), math.log(3)],
            ["i3", math.log(2), (math.log(3) + math.log(2)) / 2],
            ["i4", math.log(2), (math.log(3) + math.log(2)) / 2],
        ],
        schema=["item_idx"] + simple_i_columns,
    )
    sparkDataFrameEqual(gt_item_features, item_log_features)


def test_second_level_fp_log_features_unary_relevance_timestamp(
    spark, second_stage_fp, log_second_stage
):
    user_log_features, item_log_features = second_stage_fp._calc_log_features(
        log_second_stage.withColumn("relevance", sf.lit(1))
    )

    assert sorted(user_log_features.columns) == sorted(
        ["user_idx"] + simple_u_columns + ["u_" + col for col in time_columns]
    )
    assert sorted(item_log_features.columns) == sorted(
        ["item_idx"] + simple_i_columns + ["i_" + col for col in time_columns]
    )
    date = datetime(2019, 1, 1)
    gt_user_time_features = spark.createDataFrame(
        data=[
            ["u1", date, datetime(2019, 1, 5), math.log(2)],
            ["u2", datetime(2018, 1, 1), date, math.log(2)],
            ["u3", date, datetime(2020, 3, 1), math.log(2)],
        ],
        schema=["user_idx"] + ["u_" + col for col in time_columns],
    )
    sparkDataFrameEqual(
        gt_user_time_features,
        user_log_features.select(
            *(["user_idx"] + ["u_" + col for col in time_columns])
        ),
    )


def test_second_level_fp_log_features_relevance(
    spark, second_stage_fp, log_second_stage
):
    user_log_features, item_log_features = second_stage_fp._calc_log_features(
        log_second_stage
    )

    assert sorted(user_log_features.columns) == sorted(
        ["user_idx"]
        + simple_u_columns
        + ["u_" + col for col in time_columns + relevance_columns]
        + abnormality_columns
    )
    assert sorted(item_log_features.columns) == sorted(
        ["item_idx"]
        + simple_i_columns
        + ["i_" + col for col in time_columns + relevance_columns]
    )

    gt_item_rel_features = spark.createDataFrame(
        data=[
            ["i1", 0.0, 1.0, 1.0, 1.0, 1.0],
            ["i2", 0.0, 2.0, 2.0, 2.0, 2.0],
            [
                "i3",
                math.sqrt(((4.0 - 3.5) ** 2 + (3.0 - 3.5) ** 2) / (2.0 - 1)),
                3.5,
                3.0,
                3.0,
                4.0,
            ],
            [
                "i4",
                math.sqrt(((1.0 - 2) ** 2 + (3.0 - 2) ** 2) / (2.0 - 1)),
                2.0,
                1.0,
                1.0,
                3.0,
            ],
        ],
        schema=["item_idx"] + ["i_" + col for col in relevance_columns],
    )
    sparkDataFrameEqual(
        gt_item_rel_features,
        item_log_features.select(
            *(["item_idx"] + ["i_" + col for col in relevance_columns])
        ),
    )

    gt_user_abnorm_features = spark.createDataFrame(
        data=[
            ["u1", (abs(1.0 - 1.0) + abs(2.0 - 3.0)) / 2, 0.0],
            ["u2", (abs(1.0 - 1.0) + abs(4.0 - 3.5)) / 2, 0.03125],
            [
                "u3",
                (abs(3.0 - 3.5) + abs(2.0 - 2.0) + abs(1.0 - 2.0)) / 3,
                0.020833333333333332,
            ],
        ],
        schema=["user_idx"] + abnormality_columns,
    )
    sparkDataFrameEqual(
        gt_user_abnorm_features,
        user_log_features.select(*(["user_idx"] + abnormality_columns)),
    )


def test_conditional_features(
    spark, second_stage_fp, log_second_stage, user_features_second_stage
):

    item_cond_dist_cat_features = second_stage_fp._add_cond_distr_feat(
        log=log_second_stage,
        features_df=user_features_second_stage,
        cat_cols=["gender"],
    )
    gt_item_feat = spark.createDataFrame(
        data=[["i1", "M", 0.5], ["i1", "F", 0.5], ["i2", None, 1.0]],
        schema=["item_idx", "gender", "item_pop_by_gender"],
    )
    sparkDataFrameEqual(
        gt_item_feat,
        item_cond_dist_cat_features["gender"].filter(
            sf.col("item_idx").isin(["i1", "i2"])
        ),
    )


def test_second_level_fp_fit_transform(
    second_stage_fp,
    log_second_stage,
    user_features_second_stage,
    item_features_second_stage,
):
    second_stage_fp.fit(
        log_second_stage,
        user_features_second_stage,
        item_features_second_stage,
        user_cat_features_list=["gender"],
        item_cat_features_list=["class"],
    )
    assert second_stage_fp.user_log_features_cached is not None
    assert second_stage_fp.item_cond_dist_cat_feat_c is not None
    assert (
        "item_pop_by_gender"
        in second_stage_fp.item_cond_dist_cat_feat_c["gender"].columns
    )
    res = second_stage_fp.transform(
        (
            log_second_stage.join(
                user_features_second_stage, on="user_idx"
            ).join(item_features_second_stage, on="item_idx")
        )
    )
    assert "gender" in res.columns
    assert "item_pop_by_gender" in res.columns


def test_second_level_fp_fit_transform_one_features_df(
    second_stage_fp, log_second_stage, user_features_second_stage
):
    second_stage_fp.fit(
        log_second_stage,
        user_features_second_stage,
        item_features=None,
        user_cat_features_list=["gender"],
    )
    assert second_stage_fp.item_cond_dist_cat_feat_c is not None
    assert second_stage_fp.user_cond_dist_cat_feat_c is None
    assert (
        "item_pop_by_gender"
        in second_stage_fp.item_cond_dist_cat_feat_c["gender"].columns
    )
    res = second_stage_fp.transform(
        log_second_stage.join(user_features_second_stage, on="user_idx")
    )
    assert "item_pop_by_gender" in res.columns


def test_first_level_features_processor(item_features):
    processor = FirstLevelFeaturesProcessor(threshold=3)
    processor.fit(item_features.filter(sf.col("class") != "dog"))
    transformed = processor.transform(item_features)
    assert "iq" in transformed.columns and "color" not in transformed.columns
    assert (
        "ohe_class_dog" not in transformed.columns
        and "ohe_class_cat" in transformed.columns
    )
    assert sorted(transformed.columns) == [
        "iq",
        "item_id",
        "ohe_class_cat",
        "ohe_class_mouse",
        "ohe_color_black",
        "ohe_color_yellow",
    ]


def test_first_level_features_processor_threshold(item_features):
    # в категориальных колонках больше значений, чем threshold
    processor = FirstLevelFeaturesProcessor(threshold=1)
    processor.fit(item_features.filter(sf.col("class") != "dog"))
    transformed = processor.transform(item_features)
    assert "iq" in transformed.columns and "color" not in transformed.columns
    assert sorted(transformed.columns) == ["iq", "item_id"]


def test_first_level_features_processor_empty(item_features):
    # обработка None и случаев, когда все колонки отфильтровались
    processor = FirstLevelFeaturesProcessor(threshold=1)
    processor.fit(None)
    assert processor.transform(item_features) is None

    processor.fit(item_features.select("item_id", "class"))
    transformed = processor.transform(item_features.select("item_id", "class"))
    assert transformed.columns == ["item_id"]
