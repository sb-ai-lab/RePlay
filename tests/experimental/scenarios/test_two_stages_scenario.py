import pytest

pyspark = pytest.importorskip("pyspark")
torch = pytest.importorskip("torch")

from pyspark.sql import functions as sf

from replay.experimental.models import (
    LightFMWrap,
    ScalaALSWrap as ALSWrap,
)
from replay.experimental.preprocessing.data_preparator import ToNumericFeatureTransformer
from replay.experimental.scenarios import TwoStagesScenario
from replay.experimental.scenarios.two_stages.reranker import LamaWrap
from replay.models import ItemKNN, PopRec
from replay.preprocessing.history_based_fp import HistoryBasedFeaturesProcessor
from replay.splitters import TimeSplitter


@pytest.fixture
def two_stages_kwargs():
    return {
        "first_level_models": [
            ALSWrap(rank=4),
            ItemKNN(num_neighbours=4),
            LightFMWrap(no_components=4),
        ],
        "train_splitter": TimeSplitter(time_threshold=0.1),
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


@pytest.mark.experimental
def test_init(two_stages_kwargs):
    two_stages = TwoStagesScenario(**two_stages_kwargs)
    assert isinstance(two_stages.fallback_model, PopRec)
    assert isinstance(two_stages.second_stage_model, LamaWrap)
    assert isinstance(two_stages.features_processor, HistoryBasedFeaturesProcessor)
    assert isinstance(
        two_stages.first_level_item_features_transformer,
        ToNumericFeatureTransformer,
    )
    assert two_stages.use_first_level_models_feat == [True, True, True]

    two_stages_kwargs["use_first_level_models_feat"] = [True]
    with pytest.raises(ValueError, match="For each model from first_level_models specify.*"):
        TwoStagesScenario(**two_stages_kwargs)

    two_stages_kwargs["use_first_level_models_feat"] = True
    two_stages_kwargs["negatives_type"] = "abs"
    with pytest.raises(ValueError, match="Invalid negatives_type value.*"):
        TwoStagesScenario(**two_stages_kwargs)


@pytest.mark.xfail
@pytest.mark.experimental
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
    assert two_stages.first_level_item_len == 8
    assert two_stages.first_level_user_len == 3

    res = two_stages._add_features_for_second_level(
        log_to_add_features=short_log_with_features,
        log_for_first_level_models=long_log_with_features,
        user_features=user_features,
        item_features=item_features,
    )
    assert res.count() == short_log_with_features.count()
    assert "rel_0_ScalaALSWrap" in res.columns
    assert "m_2_fm_0" in res.columns
    assert "u_pop_by_class" in res.columns
    assert "age" in res.columns

    two_stages.first_level_item_features_transformer.transform(item_features)


@pytest.mark.xfail
@pytest.mark.experimental
def test_predict(
    long_log_with_features,
    user_features,
    item_features,
    two_stages_kwargs,
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
    assert sorted(pred.select(sf.collect_set("user_idx")).collect()[0][0]) == [
        0,
        1,
        2,
    ]


@pytest.mark.xfail
@pytest.mark.experimental
def test_optimize(
    long_log_with_features,
    short_log_with_features,
    user_features,
    item_features,
    two_stages_kwargs,
):
    two_stages = TwoStagesScenario(**two_stages_kwargs)
    param_borders = [{"rank": [1, 10]}, {}, {"no_components": [1, 10]}, None]
    # with fallback
    first_level_params, fallback_params = two_stages.optimize(
        train=long_log_with_features,
        test=short_log_with_features,
        user_features=user_features,
        item_features=item_features,
        param_borders=param_borders,
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
        param_borders=param_borders[:3],
        k=1,
        budget=1,
    )
