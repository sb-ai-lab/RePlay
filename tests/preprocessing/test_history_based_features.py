import math
from datetime import datetime

import pytest

from replay.preprocessing.history_based_fp import (
    ConditionalPopularityProcessor,
    EmptyFeatureProcessor,
    HistoryBasedFeaturesProcessor,
    LogStatFeaturesProcessor,
)
from replay.utils import PYSPARK_AVAILABLE
from tests.utils import sparkDataFrameEqual

if PYSPARK_AVAILABLE:
    from pyspark.sql import functions as sf

simple_u_columns = ["u_log_num_interact", "u_mean_i_log_num_interact"]
simple_i_columns = ["i_log_num_interact", "i_mean_u_log_num_interact"]
time_columns = [
    "log_interact_days_count",
    "max_interact_date",
    "min_interact_date",
    "history_length_days",
    "last_interaction_gap_days",
]
relevance_columns = ["std", "mean", "quantile_05", "quantile_5", "quantile_95"]
abnormality_columns = ["abnormality", "abnormalityCR"]


@pytest.fixture
def log_for_feature_gen(spark):
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
    return spark.createDataFrame([("u1", 2.0, 3.0, "M"), ("u2", 1.0, 4.0, "F")]).toDF(
        "user_idx", "user_feature_1", "user_feature_2", "gender"
    )


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame([("i1", 4.0, "cat"), ("i2", 10.0, "dog"), ("i4", 0.0, "cat")]).toDF(
        "item_idx", "item_feature_1", "class"
    )


@pytest.fixture()
def log_proc():
    return LogStatFeaturesProcessor()


@pytest.mark.spark
def test_log_proc_no_ts_rel(spark, log_proc, log_for_feature_gen):
    log_proc.fit(
        log_for_feature_gen.withColumn("relevance", sf.lit(1)).withColumn(
            "timestamp", sf.to_timestamp(sf.lit(datetime(2019, 1, 1)))
        )
    )

    assert sorted(log_proc.user_log_features.columns) == sorted(["user_idx", *simple_u_columns])
    gt_user_features = spark.createDataFrame(
        data=[
            ["u1", math.log(2), (math.log(2) + math.log(2)) / 2],
            ["u2", math.log(2), (math.log(2) + math.log(2)) / 2],
            ["u3", math.log(3), (math.log(2) + math.log(2) + math.log(1)) / 3],
        ],
        schema=["user_idx", *simple_u_columns],
    )
    sparkDataFrameEqual(gt_user_features, log_proc.user_log_features)

    assert sorted(log_proc.item_log_features.columns) == sorted(["item_idx", *simple_i_columns])
    gt_item_features = spark.createDataFrame(
        data=[
            ["i1", math.log(2), (math.log(2) + math.log(2)) / 2],
            ["i2", math.log(1), math.log(3)],
            ["i3", math.log(2), (math.log(3) + math.log(2)) / 2],
            ["i4", math.log(2), (math.log(3) + math.log(2)) / 2],
        ],
        schema=["item_idx", *simple_i_columns],
    )
    sparkDataFrameEqual(gt_item_features, log_proc.item_log_features)


@pytest.mark.spark
def test_log_proc_ts_no_rel(spark, log_proc, log_for_feature_gen):
    log_proc.fit(log_for_feature_gen.withColumn("relevance", sf.lit(1)))

    assert sorted(log_proc.user_log_features.columns) == sorted(
        ["user_idx"] + simple_u_columns + ["u_" + col for col in time_columns]
    )
    assert sorted(log_proc.item_log_features.columns) == sorted(
        ["item_idx"] + simple_i_columns + ["i_" + col for col in time_columns]
    )
    date = datetime(2019, 1, 1)
    gt_user_time_features = (
        spark.createDataFrame(
            data=[
                ["u1", math.log(2), datetime(2019, 1, 5), date, 4, 421],
                ["u2", math.log(2), date, datetime(2018, 1, 1), 365, 425],
                ["u3", math.log(2), datetime(2020, 3, 1), date, 425, 0],
            ],
            schema=["user_idx"] + ["u_" + col for col in time_columns],
        )
        .withColumn(
            "u_history_length_days",
            sf.col("u_history_length_days").astype("int"),
        )
        .withColumn(
            "u_last_interaction_gap_days",
            sf.col("u_last_interaction_gap_days").astype("int"),
        )
    )
    sparkDataFrameEqual(
        gt_user_time_features,
        log_proc.user_log_features.select(*(["user_idx"] + ["u_" + col for col in time_columns])),
    )


@pytest.mark.spark
def test_log_proc_relevance_ts(spark, log_proc, log_for_feature_gen):
    log_proc.fit(log_for_feature_gen)

    assert sorted(log_proc.user_log_features.columns) == sorted(
        ["user_idx"] + simple_u_columns + ["u_" + col for col in time_columns + relevance_columns] + abnormality_columns
    )
    assert sorted(log_proc.item_log_features.columns) == sorted(
        ["item_idx"] + simple_i_columns + ["i_" + col for col in time_columns + relevance_columns]
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
        log_proc.item_log_features.select(*(["item_idx"] + ["i_" + col for col in relevance_columns])),
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
        schema=["user_idx", *abnormality_columns],
    )
    sparkDataFrameEqual(
        gt_user_abnorm_features,
        log_proc.user_log_features.select(["user_idx", *abnormality_columns]),
    )


@pytest.mark.spark
def test_conditional_features(spark, log_for_feature_gen, user_features):
    cond_pop_proc = ConditionalPopularityProcessor(cat_features_list=["gender"])
    cond_pop_proc.fit(log=log_for_feature_gen, features=user_features)

    gt_item_feat = spark.createDataFrame(
        data=[["i1", "M", 0.5], ["i1", "F", 0.5], ["i2", None, 1.0]],
        schema=["item_idx", "gender", "i_pop_by_gender"],
    )
    sparkDataFrameEqual(
        gt_item_feat,
        cond_pop_proc.conditional_pop_dict["gender"].filter(sf.col("item_idx").isin(["i1", "i2"])),
    )


@pytest.mark.spark
def test_conditional_features_raise(spark, log_for_feature_gen, user_features):
    cond_pop_proc = ConditionalPopularityProcessor(cat_features_list=["gender", "fake_feature"])
    with pytest.raises(ValueError):
        cond_pop_proc.fit(log=log_for_feature_gen, features=user_features)


@pytest.mark.spark
def test_history_based_fp_fit_transform(
    log_for_feature_gen,
    user_features,
    item_features,
):
    history_based_fp = HistoryBasedFeaturesProcessor(
        user_cat_features_list=["gender"], item_cat_features_list=["class"]
    )

    history_based_fp.fit(log_for_feature_gen, user_features, item_features)
    assert history_based_fp.log_processor.user_log_features is not None
    assert history_based_fp.user_cond_pop_proc.conditional_pop_dict is not None
    assert "i_pop_by_gender" in history_based_fp.user_cond_pop_proc.conditional_pop_dict["gender"].columns
    res = history_based_fp.transform(
        (log_for_feature_gen.join(user_features, on="user_idx").join(item_features, on="item_idx"))
    )
    assert "gender" in res.columns
    assert "i_pop_by_gender" in res.columns
    assert "na_u_pop_by_class" in res.columns


@pytest.mark.spark
def test_history_based_fp_one_features_df(log_for_feature_gen, user_features):
    history_based_fp = HistoryBasedFeaturesProcessor(user_cat_features_list=["gender"])
    history_based_fp.fit(log=log_for_feature_gen, user_features=user_features)
    assert isinstance(history_based_fp.item_cond_pop_proc, EmptyFeatureProcessor)
    assert "i_pop_by_gender" in history_based_fp.user_cond_pop_proc.conditional_pop_dict["gender"].columns
    res = history_based_fp.transform(log_for_feature_gen.join(user_features, on="user_idx"))
    assert "i_pop_by_gender" in res.columns


@pytest.mark.spark
def test_history_based_fp_transform_raise(log_for_feature_gen, user_features):
    history_based_fp = HistoryBasedFeaturesProcessor(user_cat_features_list=["gender"])
    with pytest.raises(AttributeError, match="Call fit before running transform"):
        history_based_fp.transform(log_for_feature_gen.join(user_features, on="user_idx"))
