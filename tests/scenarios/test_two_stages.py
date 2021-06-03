# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import math
from datetime import datetime

import pytest

from pyspark.sql import functions as sf

from replay.scenarios.two_stages.feature_processor import (
    TwoStagesFeaturesProcessor,
)
from tests.utils import spark, sparkDataFrameEqual

simple_u_columns = ["u_log_ratings_count", "u_mean_log_items_ratings_count"]
simple_i_columns = ["i_log_ratings_count", "i_mean_log_users_ratings_count"]
time_columns = ["min_rating_date", "max_rating_date", "log_rating_dates_count"]
relevance_columns = ["std", "mean", "quantile_05", "quantile_5", "quantile_95"]
abnormality_columns = ["abnormality", "abnormalityCR"]


@pytest.fixture
def log(spark):
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
def user_features(spark):
    return spark.createDataFrame(
        [("u1", 2.0, 3.0, "M"), ("u2", 1.0, 4.0, "F")]
    ).toDF("user_idx", "user_feature_1", "user_feature_2", "gender")


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame(
        [("i1", 4.0, "cat"), ("i2", 10.0, "dog"), ("i4", 0.0, "cat")]
    ).toDF("item_idx", "item_feature_1", "class")


@pytest.fixture
def two_stages_fp():
    return TwoStagesFeaturesProcessor()


def test_log_features_unary_relevance_no_timestamp(spark, two_stages_fp, log):
    user_log_features, item_log_features = two_stages_fp._calc_log_features(
        log.withColumn("relevance", sf.lit(1)).withColumn(
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


def test_log_features_unary_relevance_timestamp(spark, two_stages_fp, log):
    user_log_features, item_log_features = two_stages_fp._calc_log_features(
        log.withColumn("relevance", sf.lit(1))
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


def test_log_features_relevance(spark, two_stages_fp, log):
    user_log_features, item_log_features = two_stages_fp._calc_log_features(
        log
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
    item_log_features.show()
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
    gt_user_abnorm_features.show()
    user_log_features.select(*(["user_idx"] + abnormality_columns)).show()
    sparkDataFrameEqual(
        gt_user_abnorm_features,
        user_log_features.select(*(["user_idx"] + abnormality_columns)),
    )


def test_conditional_features(spark, two_stages_fp, log, user_features):

    item_cond_dist_cat_features = two_stages_fp._add_cond_distr_feat(
        log=log, features_df=user_features, cat_cols=["gender"]
    )
    item_cond_dist_cat_features["gender"].show()
    gt_item_feat = spark.createDataFrame(
        data=[["i1", "M", 0.5], ["i1", "F", 0.5], ["i2", None, 1.0],],
        schema=["item_idx", "gender", "item_pop_by_gender"],
    )
    sparkDataFrameEqual(
        gt_item_feat,
        item_cond_dist_cat_features["gender"].filter(
            sf.col("item_idx").isin(["i1", "i2"])
        ),
    )


def test_fit_transform(two_stages_fp, log, user_features, item_features):
    two_stages_fp.fit(
        log,
        user_features,
        item_features,
        user_cat_features_list=["gender"],
        item_cat_features_list=["class"],
    )
    assert two_stages_fp.user_log_features is not None
    assert two_stages_fp.item_cond_dist_cat_features is not None
    assert (
        "item_pop_by_gender"
        in two_stages_fp.item_cond_dist_cat_features["gender"].columns
    )
    res = two_stages_fp.transform(log, user_features, item_features)
    assert "gender" in res.columns
    assert "item_pop_by_gender" in res.columns


def test_fit_transform_one_features_df(two_stages_fp, log, user_features):
    two_stages_fp.fit(
        log,
        user_features,
        item_features=None,
        user_cat_features_list=["gender"],
    )
    assert two_stages_fp.item_cond_dist_cat_features is not None
    assert two_stages_fp.user_cond_dist_cat_features is None
    assert (
        "item_pop_by_gender"
        in two_stages_fp.item_cond_dist_cat_features["gender"].columns
    )
    res = two_stages_fp.transform(log, user_features, None)
    assert "item_pop_by_gender" in res.columns
