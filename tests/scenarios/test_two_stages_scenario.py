# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import
# import math
from datetime import datetime

import pytest

# from pyspark.sql import functions as sf

from lightautoml.automl.presets.tabular_presets import TabularAutoML

from replay.models import ALSWrap, KNN, PopRec
from replay.scenarios.two_stages.feature_processor import (
    TwoStagesFeaturesProcessor,
)
from replay.scenarios.two_stages.two_stages_scenario import TwoStagesScenario

# from tests.utils import spark, sparkDataFrameEqual


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


def test_add_features(log, user_features, item_features):
    features_proc = TwoStagesFeaturesProcessor(
        use_log_features=True, use_conditional_popularity=False
    )
    two_stages = TwoStagesScenario(
        first_level_models=[ALSWrap(rank=10), KNN(num_neighbours=10)],
        use_first_level_features=True,
        num_negatives=2,
        negatives_type="first-level",
        use_generated_features=True,
        user_cat_features_list=["gender"],
        item_cat_features_list=["class"],
        custom_features_processor=features_proc,
    )
    assert isinstance(two_stages.fallback_model, PopRec)
    assert isinstance(two_stages.second_stage_model, TabularAutoML)
    two_stages.fit(log, user_features, item_features)
