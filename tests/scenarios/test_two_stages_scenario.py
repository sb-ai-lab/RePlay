# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
# import math
from datetime import datetime

import pytest

from pyspark.sql import functions as sf

from lightautoml.automl.presets.tabular_presets import TabularAutoML

from replay.models import ALSWrap, KNN, PopRec, LightFMWrap
from replay.scenarios.two_stages.feature_processor import (
    TwoStagesFeaturesProcessor,
)
from replay.scenarios.two_stages.two_stages_scenario import TwoStagesScenario

from tests.utils import spark, sparkDataFrameEqual


@pytest.fixture
def two_stages_kwargs():
    return {
        "first_level_models": [ALSWrap(rank=4), KNN(num_neighbours=4)],
        # , LightFMWrap(no_components=4)],
        "use_first_level_models_feat": True,
        "num_negatives": 2,
        "negatives_type": "first_level",
        "use_generated_features": True,
        "user_cat_features_list": ["gender"],
        "item_cat_features_list": ["class"],
        "custom_features_processor": None,
    }


@pytest.fixture
def log(spark):
    date = datetime(2019, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i1", date, 1.0],
            ["u1", "i4", datetime(2019, 1, 5), 3.0],
            ["u1", "i2", date, 2.0],
            ["u1", "i5", date, 4.0],
            ["u2", "i1", date, 1.0],
            ["u2", "i3", datetime(2018, 1, 1), 2.0],
            ["u2", "i7", datetime(2019, 1, 1), 4.0],
            ["u2", "i8", datetime(2020, 1, 1), 4.0],
            ["u3", "i9", date, 3.0],
            ["u3", "i2", date, 2.0],
            ["u3", "i6", datetime(2020, 3, 1), 1.0],
            ["u3", "i7", date, 5.0],
        ],
        schema=["user_id", "item_id", "timestamp", "relevance"],
    )


@pytest.fixture
def pairs(spark):
    date = datetime(2021, 1, 1)
    return spark.createDataFrame(
        data=[
            ["u1", "i3", date, 1.0],
            ["u1", "i7", datetime(2019, 1, 5), 3.0],
            ["u2", "i2", date, 1.0],
            ["u2", "i10", datetime(2018, 1, 1), 2.0],
            ["u3", "i8", date, 3.0],
            ["u3", "i1", date, 2.0],
            ["u4", "i7", date, 5.0],
        ],
        schema=["user_id", "item_id", "timestamp", "relevance"],
    )


@pytest.fixture
def user_features(spark):
    return spark.createDataFrame(
        [("u1", 20.0, -3.0, "M"), ("u2", 30.0, 4.0, "F")]
    ).toDF("user_id", "age", "mood", "gender")


@pytest.fixture
def item_features(spark):
    return spark.createDataFrame(
        [
            ("i1", 4.0, "cat"),
            ("i2", 10.0, "dog"),
            ("i3", 7.0, "mouse"),
            ("i4", -1.0, "cat"),
            ("i5", 11.0, "dog"),
            ("i6", 0.0, "mouse"),
        ]
    ).toDF("item_id", "iq", "class")


def test_init(two_stages_kwargs):

    two_stages = TwoStagesScenario(**two_stages_kwargs)
    assert isinstance(two_stages.fallback_model, PopRec)
    assert isinstance(two_stages.second_stage_model, TabularAutoML)
    assert two_stages.use_first_level_models_feat == [True, True, True]

    two_stages_kwargs["use_first_level_models_feat"] = [True]
    with pytest.raises(
        ValueError, match="Для каждой модели из first_level_models укажите.*"
    ):
        two_stages = TwoStagesScenario(**two_stages_kwargs)

    two_stages_kwargs["use_first_level_models_feat"] = True
    two_stages_kwargs["negatives_type"] = "abs"
    with pytest.raises(ValueError, match="Некорректное значение.*"):
        two_stages = TwoStagesScenario(**two_stages_kwargs)


def test_fit(log, pairs, user_features, item_features, two_stages_kwargs):
    two_stages_kwargs["use_first_level_models_feat"] = [True, False]
    two_stages = TwoStagesScenario(**two_stages_kwargs)

    two_stages.fit(log, user_features, item_features.filter(sf.col("iq") > 4))
    # features_proc = TwoStagesFeaturesProcessor(
    #     use_log_features=True, use_conditional_popularity=False
    # )
    res = two_stages.add_features_for_second_level(
        pairs, log, user_features, item_features
    )
    print(res.count())
    print(two_stages.first_level_item_features_transformer)
    two_stages.first_level_item_features_transformer.transform(
        item_features
    ).show()
    res.select("user_id", "item_id", "rel_1_KNN", "rel_0_ALSWrap").show()
    print(
        "two_stages.first_level_item_indexer_len",
        two_stages.first_level_item_indexer_len,
    )
    print(
        "two_stages.first_level_user_indexer_len",
        two_stages.first_level_user_indexer_len,
    )
