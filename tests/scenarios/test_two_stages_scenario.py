# pylint: disable=redefined-outer-name, missing-function-docstring, unused-import

import pytest
from pyspark.sql import functions as sf

from lightautoml.automl.presets.tabular_presets import TabularAutoML

from replay.models import ALSWrap, KNN, PopRec, LightFMWrap
from replay.scenarios.two_stages.feature_processor import (
    SecondLevelFeaturesProcessor,
    FirstLevelFeaturesProcessor,
)
from replay.scenarios import TwoStagesScenario
from replay.splitters import UserSplitter

from tests.utils import (
    spark,
    sparkDataFrameEqual,
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
)


@pytest.fixture
def two_stages_kwargs():
    return {
        "first_level_models": [
            ALSWrap(rank=4),
            KNN(num_neighbours=4),
            LightFMWrap(no_components=4),
        ],
        "train_splitter": UserSplitter(
            item_test_size=0.3, shuffle=False, seed=42
        ),
        "use_first_level_models_feat": True,
        "second_model_params": {
            "timeout": 30,
            "general_params": {"use_algos": ["lgb"]},
        },
        "num_negatives": 6,
        "negatives_type": "first_level",
        "use_generated_features": True,
        "user_cat_features_list": ["gender"],
        "item_cat_features_list": ["class"],
        "custom_features_processor": None,
    }


def test_init(two_stages_kwargs):

    two_stages = TwoStagesScenario(**two_stages_kwargs)
    assert isinstance(two_stages.fallback_model, PopRec)
    assert isinstance(two_stages.second_stage_model, TabularAutoML)
    assert isinstance(
        two_stages.features_processor, SecondLevelFeaturesProcessor
    )
    assert isinstance(
        two_stages.first_level_item_features_transformer,
        FirstLevelFeaturesProcessor,
    )
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


def test_fit(
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
    two_stages_kwargs,
):
    two_stages_kwargs["use_first_level_models_feat"] = [True, False, True]
    two_stages = TwoStagesScenario(**two_stages_kwargs)

    two_stages.fit(
        long_log_with_features,
        user_features,
        item_features.filter(sf.col("iq") > 4),
    )
    assert two_stages.first_level_item_indexer_len == 6
    assert two_stages.first_level_user_indexer_len == 3

    res = two_stages._add_features_for_second_level(
        log_to_add_features=two_stages._convert_index(short_log_with_features),
        log_for_first_level_models=two_stages._convert_index(
            long_log_with_features
        ),
        user_features=two_stages._convert_index(user_features),
        item_features=two_stages._convert_index(item_features),
    )
    assert res.count() == short_log_with_features.count()
    assert "rel_0_ALSWrap" in res.columns
    assert "m_2_fm_0" in res.columns
    assert "user_pop_by_class" in res.columns
    assert "age" in res.columns

    two_stages.first_level_item_features_transformer.transform(
        item_features.withColumnRenamed("item_id", "item_idx")
    )


def test_predict(
    long_log_with_features, user_features, item_features, two_stages_kwargs,
):
    two_stages = TwoStagesScenario(**two_stages_kwargs)

    two_stages.fit(
        long_log_with_features,
        user_features,
        item_features.filter(sf.col("iq") > 4),
    )
    pred = two_stages.predict(
        log=long_log_with_features,
        k=2,
        user_features=user_features,
        item_features=item_features,
    )
    assert pred.count() == 6
    assert sorted(pred.select(sf.collect_set("user_id")).collect()[0][0]) == [
        "u1",
        "u2",
        "u3",
    ]


def test_optimize(
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
    two_stages_kwargs,
):
    two_stages = TwoStagesScenario(**two_stages_kwargs)
    param_grid = [{"rank": [1, 10]}, {}, {"no_components": [1, 10]}, None]
    # with fallback
    first_level_params, fallback_params = two_stages.optimize(
        train=long_log_with_features,
        test=short_log_with_features,
        user_features=user_features,
        item_features=item_features,
        param_grid=param_grid,
        k=1,
        budget=1,
    )
    assert len(first_level_params) == 3
    assert first_level_params[1] is None
    assert list(first_level_params[0].keys()) == ["rank"]
    assert fallback_params is None

    # no fallback works
    two_stages.fallback_model = None
    two_stages.optimize(
        train=long_log_with_features,
        test=short_log_with_features,
        user_features=user_features,
        item_features=item_features,
        param_grid=param_grid[:3],
        k=1,
        budget=1,
    )
